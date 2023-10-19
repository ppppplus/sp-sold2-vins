#pragma once
#include <ros/ros.h>
#include <opencv2/highgui/highgui.hpp>
#include<Eigen/Core>
#include<Eigen/Dense>

extern std::string MAP_TOPIC;
extern std::string PTS_TOPIC;
extern int GRID_SIZE;
extern float NMS_THRESH;
extern float PTS_SELECT_THRESH;
extern float DETECT_THRESH;
extern double INLIER_THRESH;
extern int NUM_SAMPLES;
extern int NMS_DIST;
extern int TOPK;
extern int CANDK;
extern int Height;
extern int Width;
extern int Hc;
extern int Wc;
extern int Junction_Channel;
extern int Desc_Dim;
extern Eigen::MatrixXd sampler;
extern Eigen::MatrixXd sampler1;
void readPostprocessParameters(ros::NodeHandle &n);
