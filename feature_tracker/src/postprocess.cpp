#include "postprocess.h"
using namespace std;
void postprocess_pts(vector<double>* input_junction,
                        vector<double>* input_desc,
                        vector<FeaturePts> & out_points
                        // ,vector<float>& out_desc
                        )
{

    vector<double>* result_junction = input_junction; //65*Hc*Wc
    vector<double>* result_desc = input_desc;
    int num_semi = Junction_Channel*Hc*Wc; //65*Hc*Wc
    int num_desc = Desc_Dim*Hc*Wc;

    // static TimerKeeper cnn_post_exp_cost("CNN后处理--exp操作 平均用时: ");
    // cnn_post_exp_cost.mark();
    // cnn_post_exp_cost.average_time_cost();
    
    
    static TimerKeeper post_softmax_cost("CNN后处理--softmax（仅计算8x8区域置信度最大值） 平均用时: ");
    post_softmax_cost.mark();
    // float semi[Height][Width];
    FeaturePts coarse_semi[Hc][Wc]; // Height/Cell*Width/Cell个数据，每个里面存有原图x，y坐标和分数
    vector<FeaturePts> tmp_point;
    // float coarse_desc[Height/Cell][Width/Cell][D];
    
    // int16_t max_point_select_thresh=floor(log(CONF_thresh)/SEMI_QUANTIZE);//最大值相对于65channel的score小于后续的CONF-THRESH
    // int16_t other_point_sum_thresh=floor(log(0.001)/SEMI_QUANTIZE);// 其他值 相对于最大权值可以忽略
    // float flag_value_for_fast_exp=1.0/SEMI_QUANTIZE;//快速exp的适用范围
    for(int i=0; i<Hc; i++) {
        for(int j=0; j<Wc; j++) {
            //selec max point 遍历Hc*Wc的特征图
            int cell_index=j*Junction_Channel+i*Junction_Channel*Wc; // featuremap中的一个特征点位置
            double semi_max=-100;
            int max_h=0, max_w=0;
            for(int kh=0; kh<GRID_SIZE; kh++) 
            {
                for(int kw=0; kw<GRID_SIZE; kw++) 
                {
                    double cur_semi= result_junction->at(kw+kh*GRID_SIZE+cell_index);
                    if(cur_semi > semi_max) {
                        semi_max = cur_semi;
                        max_h = kh;
                        max_w = kw;
                    }
                }
            } // 找到当前cell中最大值的位置（相对于整个一维数组 max_h,max_w）和值semi_max
            
            // get normalized score
            double semi_background=result_junction->at(Junction_Channel-1+cell_index); //当前cell垃圾桶
            if ( (semi_max - semi_background) > PTS_SELECT_THRESH){
                // 说明当前cell中的最大值有效，纳入exp计算
                double cell_sum = 0;
                double max_semi_total=max(semi_max,semi_background); // 65通道内最大值
                for(int k=0; k<Junction_Channel; k++) {
                    double cur_semi =  result_junction->at(k+cell_index);
                    if((cur_semi - max_semi_total) > PTS_SELECT_THRESH)
                        // 说明当前的分数有效，纳入exp计算
                        cell_sum = cell_sum + exp(cur_semi);//fastexp::exp<float, IEEE, 4UL>(float(cur_semi*SEMI_QUANTIZE));
                }
                // cell_sum为64维度中不太小的值的e指数之和
                coarse_semi[i][j].score = exp(semi_max)/cell_sum;//fastexp::exp<float, IEEE, 4UL>(float(semi_max*SEMI_QUANTIZE))/cell_sum;
                coarse_semi[i][j].H = max_h+i*GRID_SIZE;
                coarse_semi[i][j].W = max_w+j*GRID_SIZE;
            }
            // if (true){//(int16_t(semi_max) - semi_background) >max_point_select_thresh ){
            //     unsigned long long int cell_sum = 0;
            //     int8_t max_semi_total=max(semi_max,semi_background);
            //     for(int k=0; k<Feature_Length; k++) {
            //         int8_t cur_semi =  result_semi[k+cell_index];
            //         // if((int16_t(cur_semi) - max_semi_total) > other_point_sum_thresh )
            //             unsigned int cell_sum_max = 0x80000000;
            //             unsigned int cell_sum_element = cell_sum_max >> (semi_max - cur_semi);
            //             cell_sum = cell_sum + cell_sum_element;//fastexp::exp<float, IEEE, 0UL>(float(cur_semi*SEMI_QUANTIZE));//
            //     }
            //     coarse_semi[i][j].semi = -float(cell_sum);//fastexp::exp<float, IEEE, 0UL>(float(semi_max*SEMI_QUANTIZE))/cell_sum;//
            // }
        }
    }
    post_softmax_cost.average_time_cost();

    static TimerKeeper post_nms_cost("CNN后处理--nms（仅抑制8x8块的邻居） 平均用时: ");
    post_nms_cost.mark();
    // coarse_semi 为Hc*Wc的softmax结果
    
    //coarse NMS
    for(int i=1; i<Hc-1; i++) {
        for(int j=1; j<Wc-1; j++) {
            if(coarse_semi[i][j].score != 0) {
                // 记录下此cell的分数，在其九宫格内做nms
                double tmp_score = coarse_semi[i][j].score;
                for(int kh=max(1,i-1); kh<min(Hc-1,i+1+1); kh++)
                    for(int kw=max(1,j-1); kw<min(Wc-1,j+1+1); kw++)
                        if(i!=kh||j!=kw) {
                            if(abs(coarse_semi[i][j].H-coarse_semi[kh][kw].H)<=NMS_DIST && abs(coarse_semi[i][j].W-coarse_semi[kh][kw].W)<=NMS_DIST) {
                                if(tmp_score>=coarse_semi[kh][kw].score)
                                    coarse_semi[kh][kw].score = 0;
                                else
                                    coarse_semi[i][j].score = 0;
                            }
                        }
                if(coarse_semi[i][j].score!=0)
                    tmp_point.push_back(coarse_semi[i][j]);
            }
        }
    } 
    
    post_nms_cost.average_time_cost();
    


    static TimerKeeper post_rank_cost("CNN后处理--thresh_check + rank_topK 平均用时: ");
    post_rank_cost.mark();
    //CONF_thresh
    vector<FeaturePts> thresh_points;
    for(int i=0; i<tmp_point.size(); i++) {
        if(tmp_point[i].score >= NMS_THRESH) {
            thresh_points.push_back(tmp_point[i]);
        }
    }

    // thresh_points是经过softmax+nms+阈值筛选留下的点集及其分数

    // topk 并且这k个数是排序好的
    vector<FeaturePts> top_points; 
    int thresh_size=thresh_points.size();
    int keep_k_points=min(TOPK, thresh_size);
    make_heap(thresh_points.begin(), thresh_points.end(), LessFunc());//大根堆
    for(int i=0;i<keep_k_points;i++)   
        pop_heap(thresh_points.begin(),thresh_points.end()-i, LessFunc());
    //从大到小k个数
    for(int i=0;i<keep_k_points;i++) 
        top_points.push_back(thresh_points[thresh_size-1-i]);
    out_points = top_points;
    post_rank_cost.average_time_cost();
    cout<<" finally selected points size:"<<out_points.size()<<endl;

    // static TimerKeeper cnn_post_desc_cost("CNN后处理--desc norm+grid_sample 平均用时: ");
    // cnn_post_desc_cost.mark();
    // out_desc.resize(top_points.size()*D);
    // Bilinera(top_points,result_desc,out_desc.data());
    // cnn_post_desc_cost.average_time_cost();

}