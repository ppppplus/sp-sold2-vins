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


#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>

#include "postprocess_parameters.h"

// using fastexp::IEEE;
// using fastexp::Product;

void postprocess_pts(float* input_junction,
                                float* input_descriptor,
                                vector<TrackPoint> & out_points,
                                vector<int8_t>& out_desc)

class FeaturePts
{
    public:
        int H;
        int W;
        float score;
}