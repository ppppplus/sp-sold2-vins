#include "postprocess.h"
using namespace std;

void sampler_init()
{
    for (int i=0; i<NUM_SAMPLES; i++)
    {
        sampler(0, i) = static_cast<double>(i)/(NUM_SAMPLES-1);
        sampler1(0, i) = 1.0 - sampler(0, i);
    }
};

FeatureLines::FeatureLines(int x1, int y1, int x2, int y2)
{
    startPointx = x1;
    startPointy = y1;
    endPointx = x2;
    endPointy = y2;
};

void postprocess_pts(vector<float>* input_junction,
                        vector<float>* input_desc,
                        vector<int32_t> &out_pts,
                        vector<float> &out_scores,
                        Eigen::Ref<Eigen::MatrixXi> out_junc
                        // vector<FeaturePts> &out_junc
                        // ,vector<float>& out_desc
                        )
{

    vector<float>* result_junction = input_junction; //65*Hc*Wc
    vector<float>* result_desc = input_desc;
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
            float semi_max=-100;
            int max_h=0, max_w=0;
            for(int kh=0; kh<GRID_SIZE; kh++) 
            {
                for(int kw=0; kw<GRID_SIZE; kw++) 
                {
                    float cur_semi= result_junction->at(kw+kh*GRID_SIZE+cell_index);
                    if(cur_semi > semi_max) {
                        semi_max = cur_semi;
                        max_h = kh;
                        max_w = kw;
                    }
                }
            } // 找到当前cell中最大值的位置（相对于整个一维数组 max_h,max_w）和值semi_max
            
            // get normalized score
            float semi_background=result_junction->at(Junction_Channel-1+cell_index); //当前cell垃圾桶
            if ( (semi_max - semi_background) > PTS_SELECT_THRESH){
                // 说明当前cell中的最大值有效，纳入exp计算
                float cell_sum = 0;
                float max_semi_total=max(semi_max,semi_background); // 65通道内最大值
                for(int k=0; k<Junction_Channel; k++) {
                    float cur_semi =  result_junction->at(k+cell_index);
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
                float tmp_score = coarse_semi[i][j].score;
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
    // vector<FeaturePts> top_junc; 
    Eigen::MatrixXi top_junc(CANDK, 2);
    vector<int32_t> top_pts;
    vector<float> top_scores;
    int thresh_size = thresh_points.size();
    int keep_k_points=min(TOPK, thresh_size);
    int keep_candidate_points = min(CANDK, keep_k_points);
    make_heap(thresh_points.begin(), thresh_points.end(), LessFunc());//大根堆
    for(int i=0;i<keep_k_points;i++)   
        pop_heap(thresh_points.begin(),thresh_points.end()-i, LessFunc());
    //从大到小k个数
    for(int i=0;i<keep_candidate_points;i++) 
    {
        top_pts.push_back(thresh_points[thresh_size-1-i].H);
        top_pts.push_back(thresh_points[thresh_size-1-i].W);
        top_scores.push_back(thresh_points[thresh_size-1-i].score);
        top_junc(i, 0) = thresh_points[thresh_size-1-i].H;
        top_junc(i, 1) = thresh_points[thresh_size-1-i].W;
        // top_junc.push_back(thresh_points[thresh_size-1-i]);
    }
    for(int i=keep_candidate_points;i<keep_k_points;i++)
    {
        top_pts.push_back(thresh_points[thresh_size-1-i].H);
        top_pts.push_back(thresh_points[thresh_size-1-i].W);
        top_scores.push_back(thresh_points[thresh_size-1-i].score);
    }

    out_junc = top_junc;
    out_pts = top_pts;
    out_scores = top_scores;
    post_rank_cost.average_time_cost();
    cout<<" finally selected points size:"<<out_scores.size()<<endl;

    // static TimerKeeper cnn_post_desc_cost("CNN后处理--desc norm+grid_sample 平均用时: ");
    // cnn_post_desc_cost.mark();
    // out_desc.resize(top_points.size()*D);
    // Bilinera(top_points,result_desc,out_desc.data());
    // cnn_post_desc_cost.average_time_cost();

};

float getSampleScore(
    float x, float y, const vector<float> *heatmap
)
{
    int row_floor = floor(x)<Wc*GRID_SIZE ? floor(x) : Wc*GRID_SIZE-1;
    int row_ceil = ceil(x)<Wc*GRID_SIZE ? ceil(x) : Wc*GRID_SIZE-1;
    int col_floor = floor(y)<Hc*GRID_SIZE ? floor(y) : Hc*GRID_SIZE-1;
    int col_ceil = ceil(y)<Hc*GRID_SIZE ? ceil(y) : Hc*GRID_SIZE-1;
    int index_lu = col_floor + Hc*GRID_SIZE*row_floor;
    int index_ru = col_ceil + Hc*GRID_SIZE*row_floor;
    int index_ld = col_floor + Hc*GRID_SIZE*row_ceil;
    int index_rd = col_ceil + Hc*GRID_SIZE*row_ceil;
    float score;
    score = heatmap->at(index_lu)*(row_ceil-x)*(col_ceil-y) +
            heatmap->at(index_ru)*(row_ceil-x)*(y-col_floor) +
            heatmap->at(index_ld)*(x-row_floor)*(col_ceil-y) +
            heatmap->at(index_rd)*(x-row_floor)*(y-col_floor);
    return score;
};

bool filterLines(
    // const std::vector<FeatureLines>* intput_lines,
    const int startx, const int starty, const int endx, const int endy,
    const std::vector<float>* heatmap,
    int numSamples
    // std::vector<FeatureLines>& output_lines
    )
    {
        // 在线段上做采样
        // 输入：线段类，包含一个线段的两个端点坐标；以及概率图heatmap
        // 输出：线段对应的numSamples个采样点以及对应的分数
        // FeatureLines line = lines[i];
        // const int startx = input_lines[i].startPointx;
        // const int starty = input_lines[i].startPointy;
        // const int endx = input_lines[i].endPointx;
        // const int endy = input_lines[i].endPointy;
        float line_score = 0.0;
        int valid_samples = 0;
        
        for (int j=0; j<numSamples; j++)
        {
            float t = float(j)/(numSamples - 1);
            float x = startx+t*(endx-startx);
            float y = starty+t*(endy-starty);
            float sample_score = getSampleScore(x, y, heatmap);
            line_score += sample_score;
            if (sample_score > DETECT_THRESH)
            {
                valid_samples += 1;
            }
        }
        line_score /= numSamples;
        // std::cout<< "line score: " << line_score <<std::endl; 
        float inlier_ratio = float(valid_samples)/numSamples;
        if (line_score > DETECT_THRESH && inlier_ratio > INLIER_THRESH)
            // output_lines.push_back(input_lines[i]);
            return true;
        else
            return false;
    };


// void postprocess_lines(vector<float>* input_heatmap,
//                         vector<FeaturePts>* input_pts,
//                         vector<float>* input_desc,
//                         vector<int32_t> &out_lines,
//                         vector<FeatureLines> &res_lines
//                         // vector<float> &out_scores
//                         // vector<FeaturePts> & out_points
//                         // ,vector<float>& out_desc
//                         )
// {
//     int numSamples = NUM_SAMPLES;
//     // int num_candidate_junc = input_pts->size();

//     int num_candidate_junc = input_pts->size();
//     static TimerKeeper post_lines_cost("CNN后处理--线特征后处理（利用候选端点和线概率图）平均用时: ");
//     post_lines_cost.mark();
// 	for (int i = 0; i < num_candidate_junc; i++)
// 	{
// 		int startx = input_pts[0][i].H;
//         int starty = input_pts[0][i].W;

//         for (int j=i+1; j<num_candidate_junc; j++)
//         {
//             int endx = input_pts[0][j].H;
//             int endy = input_pts[0][j].W;
//             // static TimerKeeper tmp_cost("CNN后处理--线特征筛选平均用时: ");
//             // tmp_cost.mark();
//             if (filterLines(startx, starty, endx, endy, input_heatmap, numSamples)){
//                 FeatureLines tmp_line(startx, starty, endx, endy);
//                 res_lines.push_back(tmp_line);
//                 out_lines.push_back(startx);
//                 out_lines.push_back(starty);
//                 out_lines.push_back(endx);
//                 out_lines.push_back(endy);
//             }
//             // tmp_cost.average_time_cost();
//         }
//     }
//     post_lines_cost.average_time_cost();
// };


void postprocess_lines(
                        // vector<float>* input_heatmap,
                        // vector<FeaturePts>* input_pts,
                        float* input_heatmap,
                        Eigen::MatrixXi input_pts,
                        vector<float>* input_desc,
                        vector<int32_t> &out_lines,
                        vector<FeatureLines> &res_lines
                        // vector<float> &out_scores
                        // vector<FeaturePts> & out_points
                        // ,vector<float>& out_desc
                        )
{
    int numSamples = NUM_SAMPLES;
    // int num_candidate_junc = input_pts->size();

    // int num_candidate_junc = input_pts->size();
    int num_candidate_junc = input_pts.rows();
    const int candidate_num = num_candidate_junc*(num_candidate_junc-1)/2;
    
    Eigen::MatrixXi candidate_map(num_candidate_junc, num_candidate_junc);
    Eigen::MatrixXi candidate_index(candidate_num, 2);
    Eigen::MatrixXi candidate_junc_start(candidate_num, 2);
    Eigen::MatrixXi candidate_junc_end(candidate_num, 2);
    int num = 0;
    
    static TimerKeeper post_lines_cost("CNN后处理--线特征后处理（利用候选端点和线概率图）平均用时: ");
    post_lines_cost.mark();
    for (int i = 0; i < num_candidate_junc; ++i) {
        for (int j = i+1; j < num_candidate_junc; ++j) {
            candidate_map(i, j) = 1;
            candidate_index(num, 0) = i;
            candidate_index(num, 1) = j;
            candidate_junc_start(num, 0) = input_pts(i, 0);
            candidate_junc_start(num, 1) = input_pts(i, 1);
            candidate_junc_end(num, 0) = input_pts(j, 0);
            candidate_junc_end(num, 1) = input_pts(j, 1);
            num++;
        }
    }
    post_lines_cost.average_time_cost();
    Eigen::MatrixXd cand_x(candidate_num, NUM_SAMPLES);
    Eigen::MatrixXd cand_y(candidate_num, NUM_SAMPLES);
    Eigen::MatrixXi startx = candidate_junc_start.col(0);
    cand_x = candidate_junc_start.col(0).cast<double>()*sampler + candidate_junc_end.col(0).cast<double>()*sampler1;
    cand_y = candidate_junc_start.col(1).cast<double>()*sampler + candidate_junc_end.col(1).cast<double>()*sampler1;
    // cand_x和cand_y为待筛选的线段采样点，维度均为candidate_num*NUM_SAMPLES，代表candidate_num条线段上NUM_SAMPLES个点的横纵坐标

    Eigen::MatrixXi cand_x_floor = (cand_x.array().floor()).cast<int>();
    Eigen::MatrixXi cand_x_ceil = (cand_x.array().ceil()).cast<int>();
    Eigen::MatrixXi cand_y_floor = (cand_y.array().floor()).cast<int>();
    Eigen::MatrixXi cand_y_ceil = (cand_y.array().ceil()).cast<int>();
    
    // Eigen::Matrix<Eigen::Index, Eigen::Dynamic, Eigen::Dynamic> cand_x_floor_index = cand_x_floor.cast<Eigen::Index>();
    // Eigen::Matrix<Eigen::Index, Eigen::Dynamic, Eigen::Dynamic> cand_x_ceil_index = cand_x_ceil.cast<Eigen::Index>();
    // Eigen::Matrix<Eigen::Index, Eigen::Dynamic, Eigen::Dynamic> cand_y_floor_index = cand_y_floor.cast<Eigen::Index>();
    // Eigen::Matrix<Eigen::Index, Eigen::Dynamic, Eigen::Dynamic> cand_y_ceil_index = cand_y_ceil.cast<Eigen::Index>();

    // Eigen::Index cand_x_ceil_index = cand_x_ceil.cast<Eigen::Index>();
    // Eigen::Index cand_y_floor_index = cand_y_floor.cast<Eigen::Index>();
    // Eigen::Index cand_y_ceil_index = cand_y_ceil.cast<Eigen::Index>();


    // 思路：将cand_x/y_floor/ceil转换为下标矩阵cand_index_lu和cand_indexru等
    // 第一种方法：重复利用score vector，多操作放入循环
    // Eigen::VectorXd score(NUM_SAMPLES);
    // for (int i=0; i<candidate_num; i++)
    // {
    //     for (int j=0; j<NUM_SAMPLES; j++)
    //     {
    //         scores(j) = input_heatmap(cand_x_floor(i,j), cand_y_floor(i,j))*(cand_x_ceil(i,j) - cand_x(i,j))*(cand_y_ceil(i,j) - cand_y(i,j)) +
    //         input_heatmap(cand_x_floor(i,j), cand_y_ceil(i,j))*(cand_x_ceil(i,j) - cand_x(i,j))*(cand_w(i,j) - cand_w_floor(i,j)) +
    //         input_heatmap(cand_x_ceil(i,j), cand_y_floor(i,j))*(cand_x(i,j) - cand_x_floor(i,j))*(cand_y_ceil(i,j) - cand_y(i,j)) +
    //         input_heatmap(cand_x_ceil(i,j), cand_y_ceil(i,j))*(cand_x(i,j) - cand_x_floor(i,j))*(cand_y(i,j) - cand_y_floor(i,j))
    //     }
    //     Eigen::VectorXd
    // } 
    // Eigen::VectorXd linescores = scores.rowwise().mean();
    // 第二种方法：用Matrix整体计算，减少循环中的计算量
    vector<FeatureLines> cand_lines;
    Eigen::VectorXf scores(NUM_SAMPLES);
    float line_score;
    int count;
    float inlier_rate;
    // 指针下：heatmap(i, j)对应于 *(input_heatmap+i*Wc+j)
    // 从cand_x_*中获取i，从cand_y_*中获取j
    for (int i=0; i<candidate_num; i++)
    {
        for (int j=0; j<NUM_SAMPLES; j++)
        {
            scores(j) = *(input_heatmap+Wc*cand_x_floor(i,j) + cand_y_floor(i,j))*(cand_x_ceil(i,j) - cand_x(i,j))*(cand_y_ceil(i,j) - cand_y(i,j)) +
            *(input_heatmap+Wc*cand_x_floor(i,j) + cand_y_ceil(i,j))*(cand_x_ceil(i,j) - cand_x(i,j))*(cand_y(i,j) - cand_y_floor(i,j)) +
            *(input_heatmap+Wc*cand_x_ceil(i,j) + cand_y_floor(i,j))*(cand_x(i,j) - cand_x_floor(i,j))*(cand_y_ceil(i,j) - cand_y(i,j)) +
            *(input_heatmap+Wc*cand_x_ceil(i,j) + cand_y_ceil(i,j))*(cand_x(i,j) - cand_x_floor(i,j))*(cand_y(i,j) - cand_y_floor(i,j));
        }
        line_score = scores.array().mean();
        count = (scores.array() > DETECT_THRESH).count();
        inlier_rate = float(count)/NUM_SAMPLES;
        if (line_score > DETECT_THRESH && inlier_rate > INLIER_THRESH)
        {
            FeatureLines tmp_line(candidate_junc_start(i,0), candidate_junc_start(i,1), candidate_junc_end(i,0), candidate_junc_end(i,1));
            cand_lines.push_back(tmp_line);
        }
    } 
    res_lines = cand_lines;
    // Eigen::VectorXd linescores = scores.rowwise().mean();
    // Eigen::VectorXd filteredVector = (linescores.array() > DETECT_THRESH).select(linescores, 0.0);

    // Eigen::MatrixXf feat1 = (cand_x_ceil - cand_x).cwiseProduct(cand_y_ceil - cand_y).cwiseProduct(heatmap.block(cand_x_floor, cand_y_floor));
    // Eigen::MatrixXf feat1 = input_heatmap.block(cand_x_floor, cand_y_floor);
    // scores = input_heatmap(cand_x_floor_index, cand_y_floor_index);
    // Eigen::VectorXd line_scores = scores.rowwise().mean();
    // Eigen::Array<bool> detect_res()
    // cand_y = candidate_junc_start.col(1)*sampler + candidate_junc_end.col(1)*sampler1;

    // 构建上三角矩阵candidate_map，以及index矩阵candidate_index
    // candidate_junc_start为这candidate_num条候选线段的起始点集，candidate_junc_end为这candidate_num条线段的末端点集
	// Eigen::Index candidate_indices = candi date_map.cast<int>().nonZeros();
    // Eigen::ArrayXXi candidate_coords(2, candidate_indices.cols());

    // // Convert candidate indices to coordinates
    // candidate_coords.row(0) = candidate_indices.cast<float>().cast<float>().row(0);
    // candidate_coords.row(1) = candidate_indices.cast<float>().cast<float>().row(1);  

    // Simulate the sampler (self.torch_sampler.to(device)[None, ...])
    // Eigen::ArrayXf sampler(64);
    // sampler.setRandom(); // Simulated sampler

    // // Compute candidate samples using broadcasting
    // Eigen::ArrayXXf cand_samples_h = candidate_coords.row(0) * sampler + (1.0 - sampler) * candidate_coords.row(0);
    // Eigen::ArrayXXf cand_samples_w = candidate_coords.row(1) * sampler + (1.0 - sampler) * candidate_coords.row(1);

    // // Clip to image boundary using Eigen's min and max functions
    // cand_samples_h = cand_samples_h.min(Height - 1).max(0);
    // cand_samples_w = cand_samples_w.min(Width - 1).max(0);
    
};


// { 
//     int num_candidate_endpts = input_pts.size()
//     vector<vector<uint16_t>> line_map_pred = initial_matrix()
//     std::vector<Point> samplingPoints;

//     for (size_t i = 0; i < points.size() - 1; ++i) {
//         const Point& startPoint = points[i];
//         const Point& endPoint = points[i + 1];

//         for (int j = 0; j < numSamples; ++j) {
//             double t = static_cast<double>(j) / (numSamples - 1);
//             double x = startPoint.x + t * (endPoint.x - startPoint.x);
//             double y = startPoint.y + t * (endPoint.y - startPoint.y);
//             samplingPoints.push_back({x, y});
//         }
//     }

//     return samplingPoints;
// }
