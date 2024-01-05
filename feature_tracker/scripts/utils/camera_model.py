import os
import rospy 
import numpy as np
import cv2
import math
from scipy.linalg import eig

class CameraModel():
    def __init__(self, params):
        self.support_camera_types = ["PINHOLE", "KANNALA_BRANDT"]
        self.params = params
        self.model_type = self.params["model_type"]
        self.checkCameraType()
        # rospy.loginfo(params.keys())
    
    def checkCameraType(self):
        if self.model_type not in self.support_camera_types:
            raise ValueError(
                "[Error] The camera type selection '%s' is not supported.", 
                self.model_type)

    def generateCameraModel(self):
        if self.model_type == "PINHOLE":
            camera_model = PinholeCamera(self.params["distortion_parameters"], self.params["projection_parameters"])
            return camera_model
        elif self.model_type == "KANNALA_BRANDT":
            camera_model = KannalabrantCamera(self.params["projection_parameters"], self.params["image_width"], self.params["image_height"])
            return camera_model
        else:
            raise ValueError(
                "[Error] The camera type selection '%s' is not supported.", 
                self.model_type)
        
class PinholeCamera:

    def __init__(self, distortion_parameters, projection_parameters):

        self.fx = projection_parameters["fx"]
        self.fy = projection_parameters["fy"]
        self.cx = projection_parameters["cx"]
        self.cy = projection_parameters["cy"]
        self.d = list(distortion_parameters.values())
        self.K = [[self.fx, 0, self.cx],
            [0, self.fy, self.cy],
            [0, 0, 1]]
        

    def distortion(self, p_u):
        k1 = self.d[0]
        k2 = self.d[1]
        p1 = self.d[2]
        p2 = self.d[3]

        mx2_u = p_u[0] * p_u[0]
        my2_u = p_u[1] * p_u[1]
        mxy_u = p_u[0] * p_u[1]
        rho2_u = mx2_u + my2_u
        rad_dist_u = k1 * rho2_u + k2 * rho2_u * rho2_u

        d_u0 = p_u[0] * rad_dist_u + 2.0 * p1 * mxy_u + p2 * (rho2_u + 2.0 * mx2_u)
        d_u1 = p_u[1] * rad_dist_u + 2.0 * p2 * mxy_u + p1 * (rho2_u + 2.0 * my2_u)

        return (d_u0, d_u1)

    def liftProjective(self, p):
        mx_d = (p[0]-self.cx)/self.fx
        my_d = (p[1]-self.cy)/self.fy

        return (mx_d, my_d, 1.0)
    
    def undistortImg(self, img):
        # mapx, mapy = cv2.initUndistortRectifyMap(np.array(self.K), np.array(self.d), None, np.array(self.K), (600,400), 5)
        img_distort = cv2.undistort(img, np.array(self.K), np.array(self.d))
        # img_distort = cv2.remap(img, mapx, mapy, cv2.INTER_LINEAR)
        return img_distort
    
class KannalabrantCamera():
    def __init__(self, projection_parameters, w, h):
        self.k2 = projection_parameters["k2"]
        self.k3 = projection_parameters["k3"]
        self.k4 = projection_parameters["k4"]
        self.k5 = projection_parameters["k5"]
        self.mu = projection_parameters["mu"]
        self.mv = projection_parameters["mv"]
        self.u0 = projection_parameters["u0"]
        self.v0 = projection_parameters["v0"]
        self.m_inv_K11 = 1.0 / self.mu
        self.m_inv_K13 = -self.u0 / self.mu
        self.m_inv_K22 = 1.0 / self.mv
        self.m_inv_K23 = -self.v0 / self.mv
        self.npow = 9
        self.tol = 1e-10
        DIM = (w, h)
        K = np.array([[self.mu, 0.0, self.mv], [0.0, self.u0, self.v0], [0.0, 0.0, 1.0]])
        D = np.array([self.k2,self.k3,self.k4,self.k5])
        self.map_x, self.map_y = cv2.fisheye.initUndistortRectifyMap(K, D, np.eye(3), K, DIM, cv2.CV_16SC2)
        
    def liftProjective(self, p):
        p_u = np.array([self.m_inv_K11 * p[0] + self.m_inv_K13, self.m_inv_K22 * p[1] + self.m_inv_K23])
        theta, phi = self.backProjectSymmetric(p_u)
        P0 = math.sin(theta) * math.cos(phi)
        P1 = math.sin(theta) * math.sin(phi)
        P2 = math.cos(theta)
        return P0, P1, P2

    def backProjectSymmetric(self, p_u):
        
        p_u_norm = np.linalg.norm(p_u)
        if (p_u_norm < 1e-10):
            phi = 0.0
        else:    
            phi = math.atan2(p_u[1], p_u[0])
        coeffs = np.zeros((self.npow + 1, 1))
        coeffs[0] = -p_u_norm
        coeffs[1] = 1.0
        coeffs[3] = self.k2
        coeffs[5] = self.k3
        coeffs[7] = self.k4
        coeffs[9] = self.k5
        A = np.zeros((self.npow, self.npow))
        A[1:, :-1] = np.eye(self.npow - 1)
        A[:, -1] = -coeffs[:-1, 0] / coeffs[-1, 0]
        eigval, _ = eig(A)  # calculate eigenvalues

        # extract proper eigenvalue
        thetas = []
        for t in eigval:
            if abs(t.imag) > self.tol:
                continue
            t = t.real
            if t < -self.tol:
                continue
            elif t < 0.0:
                t = 0.0
            thetas.append(t)
        if not thetas:
            theta = p_u_norm
        else:
            theta = min(thetas)
        return theta, phi
        
    def undistortImg(self, img):
        undistorted_img = cv2.remap(img, self.map_x, self.map_y, cv2.INTER_LINEAR)
        return undistorted_img