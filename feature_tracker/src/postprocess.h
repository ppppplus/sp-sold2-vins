#pragma once

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

#include "postprocess_parameters.h"

// using fastexp::IEEE;
// using fastexp::Product;
using namespace std;

class FeaturePts
{
    public:
        int H;
        int W;
        float score;
};

class LessFunc {
    public:
        bool operator() (const FeaturePts &l, const FeaturePts &r) const {
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

void postprocess_pts(vector<double>* input_junction,
                        vector<double>* input_descriptor,
                        vector<FeaturePts> & out_points);
                        // vector<int8_t>& out_desc)

