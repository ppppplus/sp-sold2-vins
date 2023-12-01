#include <ros/ros.h>
#include <cv_bridge/cv_bridge.h>
#include <message_filters/subscriber.h>


#include <std_msgs/Float32MultiArray.h>
#include <std_msgs/Int32MultiArray.h>
#include <feature_tracker/Featuremap.h>
#include "postprocess_parameters.h"
#include "postprocess.h"


int heatmap_c, heatmap_h, heatmap_w;
int junction_c, junction_h, junction_w;
int coarse_desc_c, coarse_desc_h, coarse_desc_w;

// cv::Mat heatmap_data;
// cv::Mat junction_data;
// cv::Mat coarse_desc_data;
float* heatmap_data_array;
std::vector<float>* junction_data_array;
std::vector<float>* coarse_desc_data_array;

std_msgs::Float32MultiArray heatmap;
std_msgs::Float32MultiArray junction;
std_msgs::Float32MultiArray coarse_desc;
std_msgs::Int32MultiArray pts;

ros::Publisher pub_pts;

void publish_pts(std::vector<int32_t> pts)
{
    int l = pts.size();
    std_msgs::Int32MultiArray msg;
    std_msgs::MultiArrayDimension d1, d2, d3;
    
    d1.size = l;
    d2.size = 1;
    d3.size = 1;
    std_msgs::MultiArrayLayout layout;
    layout.data_offset = 0;
    layout.dim = std::vector<std_msgs::MultiArrayDimension>({d1, d2, d3});
    msg.data = pts;
    msg.layout = layout;
    pub_pts.publish(msg);
}

void featuremap_callback(const feature_tracker::Featuremap::ConstPtr &map_msg)
{
    heatmap = map_msg -> heatmap;
    junction = map_msg -> junction;
    coarse_desc = map_msg -> coarse_desc;
    heatmap_h = heatmap.layout.dim[0].size;
    heatmap_w = heatmap.layout.dim[1].size;
    heatmap_c = heatmap.layout.dim[2].size;
    junction_h = junction.layout.dim[0].size;
    junction_w = junction.layout.dim[1].size;
    junction_c = junction.layout.dim[2].size;
    coarse_desc_h = coarse_desc.layout.dim[0].size;
    coarse_desc_w = coarse_desc.layout.dim[1].size;
    coarse_desc_c = coarse_desc.layout.dim[2].size;
    // heatmap_data = cv::Mat(heatmap.data).reshape(heatmap_c, heatmap_h);
    // junction_data = cv::Mat(junction.data).reshape(junction_c, junction_h);
    // coarse_desc_data = cv::Mat(junction.data).reshape(junction_c, junction_h);
    heatmap_data_array = heatmap.data.data();
    // Eigen::Map<Eigen::MatrixXf> heatmapMap(heatmap_data_array, heatmap_h, heatmap_w);
    // Eigen::MatrixXf heatmapMatrix = Eigen::MatrixXf(heatmapMap);
    // Eigen::MatrixXd heatmap_data = heatmapMatrix.resize(heatmap_h, heatmap_w);
    // Eigen::Ref<Eigen::MatrixXf> heatmap_data_array = heatmapMatrix;
    junction_data_array = &junction.data;
    coarse_desc_data_array = &coarse_desc.data;

    // std::cout << "get heatmap with size: " << heatmap.data.size() << std::endl;
    // std::cout << "get junction with size: " << junction.data.size() << std::endl;
    // std::cout << "get coarse_desc with size: " << coarse_desc.data.size() << std::endl;
    
    // vector<FeaturePts> pts;
    std::vector<int32_t> pts;
    std::vector<float> scores;
    // std::vector<FeaturePts> juncs;
    Eigen::MatrixXi juncs(CANDK, 2);
    postprocess_pts(junction_data_array, coarse_desc_data_array, pts, scores, juncs);
    publish_pts(pts);
    std::vector<FeatureLines> feature_lines;
    std::vector<int32_t> lines;
    // std::vector<FeaturePts>* juncs_array = &juncs;
    // Eigen::MatrixXi* juncs_array = juncs;
    postprocess_lines(heatmap_data_array, juncs, coarse_desc_data_array, lines, feature_lines);
    std::cout<<"lines_size: "<<lines.size()<<std::endl;
}

int main(int argc, char **argv)
{
    ros::init(argc, argv, "postprocess_node");
    ros::NodeHandle n("~"); 
    ros::console::set_logger_level(ROSCONSOLE_DEFAULT_NAME, ros::console::levels::Info);
    readPostprocessParameters(n);
    sampler_init();
    std::cout << "postprocess node initialized, waiting for topic " << MAP_TOPIC << std::endl;
    ros::Subscriber sub_featuremap = n.subscribe(MAP_TOPIC, 100, featuremap_callback);
    pub_pts = n.advertise<std_msgs::Int32MultiArray>(PTS_TOPIC, 100);
    // pub_pts = n.advertise<feature_tracker::Points>("pts", 1000);
    // pub_lines = n.advertise<feature_tracker::Lines>("lines", 1000);
    /*
    if (SHOW_TRACK)
        cv::namedWindow("vis", cv::WINDOW_NORMAL);
    */
    ros::spin();
    return 0;
}
