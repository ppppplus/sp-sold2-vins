#include "postprocess_parameters.h"

std::string MAP_TOPIC;
int GRID_SIZE;
int Height;
int Width;
int Desc_Dim;
int Hc;
int Wc;
int Junction_Channel;
float NMS_THRESH;
float PTS_SELECT_THRESH;
int NMS_DIST;
int TOPK;
template <typename T>
T readParam(ros::NodeHandle &n, std::string name)
{
    T ans;
    if (n.getParam(name, ans))
    {
        ROS_INFO_STREAM("Loaded " << name << ": " << ans);
    }
    else
    {
        ROS_ERROR_STREAM("Failed to load " << name);
        n.shutdown();
    }
    return ans;
}

void readPostprocessParameters(ros::NodeHandle &n)
{
    std::string config_file;
    config_file = readParam<std::string>(n, "config_file");
    // std::cout<<config_file<<std::endl;
    cv::FileStorage fsSettings(config_file, cv::FileStorage::READ);
    if(!fsSettings.isOpened())
    {
        std::cerr << "ERROR: Wrong path to settings" << std::endl;
    }

    fsSettings["map_topic"] >> MAP_TOPIC;

    GRID_SIZE = fsSettings["grid_size"];
    NMS_THRESH = fsSettings["nms_thresh"];
    PTS_SELECT_THRESH = fsSettings["pts_select_thresh"];
    NMS_DIST = fsSettings["nms_dist"];
    TOPK = fsSettings["topk"];

    Height = fsSettings["H"];
    Width = fsSettings["W"];
    Desc_Dim = 128;
    Hc = Height/GRID_SIZE;
    Wc = Width/GRID_SIZE;
    Junction_Channel = GRID_SIZE*GRID_SIZE+1;

    fsSettings.release();


}

