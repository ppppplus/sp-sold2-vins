#pragma once
#include <ros/ros.h>
#include <opencv2/highgui/highgui.hpp>

extern std::string MAP_TOPIC;
extern int GRID_SIZE;
extern float NMS_THRESH;
extern float PTS_SELECT_THRESH;
extern int NMS_DIST;
extern int TOPK;
extern int Height;
extern int Width;
extern int Hc;
extern int Wc; 
extern int Junction_Channnel;
extern int Desc_Dim;
void readPostprocessParameters(ros::NodeHandle &n);
