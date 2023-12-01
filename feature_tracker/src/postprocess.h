#ifndef POSTPROCESS_H
#define POSTPROCESS_H

#include <cstdio>
#include <iostream>
#include <queue>
#include <execinfo.h>
#include <csignal>

#include <iostream>
#include <string>
#include <vector>
#include <sys/time.h>
#include <vector>

#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include<Eigen/Core>
#include<Eigen/Dense>
#include "NumCpp.hpp"


#include "postprocess_parameters.h"

// using fastexp::IEEE;
// using fastexp::Product;
using namespace std;

void sampler_init();

class FeaturePts
{
    public:
        int H;
        int W;
        float score;
};

class FeatureLines
{
    public:
        int startPointx;
        int startPointy;
        int endPointx;
        int endPointy;
        float score;
        FeatureLines(int x1, int x2, int x3, int x4);
};

class LessFunc {
    public:
        bool operator() (const FeaturePts &l, const FeaturePts &r) const {
            return l.score < r.score;
        }
};

class LineLessFunc{
        public:
        bool operator() (const FeatureLines &l, const FeatureLines &r) const {
            return l.score < r.score;
        }
};

class TimerKeeper{
    public:
        int sum_step;
        float sum_time;

        struct  timeval  start;
        struct  timeval  end;
        string remarks;

        TimerKeeper(string r):remarks(r){sum_step=0;sum_time=0;}

        void mark(){    
            gettimeofday(&start,NULL); //程序段开始前取得系统运行时间(ms)
        }

        void average_time_cost( bool print=true){
            gettimeofday(&end,NULL);//程序段结束后取得系统运行时间(ms)
            float new_cost=1000.0 * (end.tv_sec-start.tv_sec)+ (end.tv_usec-start.tv_usec)/1000.0;
            sum_time+=new_cost;
            sum_step++;
            if(print)
                cout << remarks.c_str() << sum_time/sum_step << " ms" << endl;

        }
};

// vector<vector<double>> initial_matrix(int m, int n) {
//     //初始化矩阵
//     vector<vector<double>> array(m);
//     for (int i = 0; i < m; i++)
//     {
//         array[i].resize(n);
//     }
//     return array;
// }

void postprocess_pts(vector<float>* input_junction,
                        vector<float>* input_descriptor,
                        vector<int32_t> &out_pts,
                        vector<float> &out_scores,
                        Eigen::Ref<Eigen::MatrixXi> out_junc
                        ); 
                        // vector<FeaturePts> &out_junc);
                        // vector<int8_t>& out_desc)
void postprocess_lines( // vector<float>* input_heatmap,
                        // vector<FeaturePts>* input_pts,shuang
                        nc::NdArray<float> input_heatmap,
                        nc::NdArray<int> input_pts,
                        vector<float>* input_desc,
                        vector<int32_t> &out_lines,
                        // vector<float> &out_scores
                        vector<FeatureLines> &res_lines
                        );
template<typename dtype>
nc::NdArray<dtype> getSlicefromIndexArray(nc::NdArray<dtype> matrix, nc::NdArray<dtype> index1, nc::NdArray<dtype> index2);


nc::NdArray<int> createCandIndex(int num_candidate_junc);

void createSampler(int num_samples, nc::NdArray<float> &sampler, nc::NdArray<float>&tsampler);

template<typename dtype>
nc::NdArray<dtype> getCandScore(nc::NdArray<dtype>cand_x, nc::NdArray<dtype>cand_y, nc::NdArray<dtype> heatmap, int num_samples);
#endif