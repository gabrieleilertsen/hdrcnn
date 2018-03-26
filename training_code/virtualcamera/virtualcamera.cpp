/*
  License:
  -----------------------------------------------------------------------------
  Copyright (c) 2017, Gabriel Eilertsen.
  All rights reserved.
  
  Redistribution and use in source and binary forms, with or without 
  modification, are permitted provided that the following conditions are met:
  
  1. Redistributions of source code must retain the above copyright notice, 
     this list of conditions and the following disclaimer.
  
  2. Redistributions in binary form must reproduce the above copyright notice,
     this list of conditions and the following disclaimer in the documentation
     and/or other materials provided with the distribution.
  
  3. Neither the name of the copyright holder nor the names of its contributors
     may be used to endorse or promote products derived from this software 
     without specific prior written permission.
  
  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" 
  AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE 
  IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE 
  ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE 
  LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR 
  CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF 
  SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS 
  INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN 
  CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) 
  ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE 
  POSSIBILITY OF SUCH DAMAGE.
  -----------------------------------------------------------------------------
 
  Description: Data augmentation by means of a virtual camera. The application
               takes an input directory containing images, and prepares the
               data for training. For each image a set of images are cropped
               and scaled, followed by simulation of a camera capture. This
               includes parameters such as camera curve, noise, and white 
               balancing. The images are stored to specified output directory
               with one ground truth HDR image and one JPEG LDR image.
  Author: Gabriel Eilertsen, gabriel.eilertsen@liu.se
  Date: March 2018
*/


#include <fstream>
#include <vector>
#include <cassert>
#include <time.h>
#include <stdlib.h> 
#include <string>
#include <math.h>
#include <random>

#include "opencv2/opencv.hpp"
#include "util.h"

using namespace cv;
using namespace std;

// Data generation/augmentation parameters
struct Params
{
    Params()
    {
        linearize = 0;                           // If the input images should be linearized
        K = 10;                                  // Number of images to generate from a 1 MP input image
        ss[0] = 320; ss[1] = 320; ss[2] = 3;     // Size of output images
        imsc[0] = 0.2; imsc[1] = 0.6;            // How large fraction of input image to select
        clip[0] = 0.85; clip[1] = 0.95;          // Min and max percentage of intensities to maintain (the rest will be clipped in the LDR)
        noise[0] = 0; noise[1] = 0.01;           // Noise std is picked between these
        hue[0] = 0.0; hue[1] = 7.0;              // Hue mean and std
        sat[0] = 0.0; sat[1] = 0.1;              // Color saturation mean and std
        sigmoid_n[0] = 0.9; sigmoid_n[1] = 0.1;  // Sigmoidal parameter 'n' mean and std
        sigmoid_a[0] = 0.6; sigmoid_a[1] = 0.1;  // Sigmoidal parameter 'a' mean and std
        jpgQ = 30;                               // Minimum JPEG quality
        ipath = "input_data";                    // Input data directory
        opath = "training_data";                 // Output training data directory

        // Normal distributions for randomizing camera parameters
        distribution_hue = normal_distribution<float>(hue[0],hue[1]);
        distribution_sat = normal_distribution<float>(sat[0],sat[1]);
        distribution_sign = normal_distribution<float>(sigmoid_n[0],sigmoid_n[1]);
        distribution_siga = normal_distribution<float>(sigmoid_a[0],sigmoid_a[1]);
    }
    
    bool linearize;
    int K, ss[3], jpgQ;
    string ipath, opath;
    float imsc[2], clip[2], noise[2], hue[2], sat[2], sigmoid_n[2], sigmoid_a[2];

    default_random_engine generator;
    normal_distribution<float> distribution_hue;
    normal_distribution<float> distribution_sat;
    normal_distribution<float> distribution_sign;
    normal_distribution<float> distribution_siga;
};


// The virtual camera that performs processing and augmentation
struct VirtualCamera
{
    VirtualCamera(string fname_i, int k, Params *P_i)
    {
        N_bins = 256;
        P = P_i;
        K = k;
        fname = fname_i;
        N = 10;
        sz[0] = sz[1] = sz[2] = 0;
        range = new float[N_bins+1];
    }


    ~VirtualCamera()
    {
        if (range != NULL) delete[] range;
    }


    void init()
    {
        N = max(1.0, P->K*(sz[0]*sz[1])/(1e6));

        host_roi = Mat(P->ss[0], P->ss[1], CV_32FC3);
        host_roi8 = Mat(P->ss[0], P->ss[1], CV_8UC3);
    }


    // Read an input image
    bool read()
    {
        host_im = cv::imread(fname.c_str(), CV_LOAD_IMAGE_COLOR|CV_LOAD_IMAGE_ANYDEPTH);
        sz[0] = host_im.rows; sz[1] = host_im.cols; sz[2] = host_im.channels();

        // If needed, convert to float and normalize
        if (host_im.type() != CV_32FC3 && host_im.type() != CV_32FC1)
        {
            if (sz[2] == 1)
                host_im.convertTo(host_im, CV_32FC1);
            else
                host_im.convertTo(host_im, CV_32FC3);

            double mi, ma;
            minMaxLoc(host_im, &mi, &ma);
            multiply(host_im, (1.0f/ma) * Scalar(1,1,1), host_im);
        }

        cv::cvtColor(host_im, host_im, CV_BGR2RGB, 0);

        init();

        return 1;
    }


    // Print HDR image as binary RAW data
    void printHDR(int k=1)
    {
        char str[500];
        sprintf(str, "%s/bin/im_%06d_%06d.bin", P->opath.c_str(), K, k);
        ofstream os(str);

        // Write metadata
        float sz_f[5];
        sz_f[0] = P->ss[0]; sz_f[1] = P->ss[1]; sz_f[2] = P->ss[2];
        sz_f[3] = sigmoid_n; sz_f[4] = sigmoid_a;
        os.write((char*) &sz_f[0], 5*sizeof(float));

        // Write HDR image
        os.write((char*) &host_roi.data[0], P->ss[0]*P->ss[1]*P->ss[2]*sizeof(float));
        os.close();
    }

    // Print JPEG with random compression/quality
    void printLDR(int k=1)
    {
        char str[500];
        sprintf(str, "%s/jpg/im_%06d_%06d.jpg", P->opath.c_str(), K, k);

        // Write JPG compressed LDR image
        host_roi.convertTo(host_roi8, CV_8UC3, 255.0f);
        cvtColor(host_roi8, host_roi8, CV_BGR2RGB);
        std::vector<int> compression_params;
        compression_params.push_back(CV_IMWRITE_JPEG_QUALITY);
        compression_params.push_back(min(P->jpgQ + rand() % (105-P->jpgQ), 100));
        imwrite(str, host_roi8, compression_params);
    }


    // Perform the data augmentation
    void run()
    {
        //printf("%s\n", fname.c_str());

        if (!read())
            return;

        if (sz[0] < P->ss[0] || sz[1] < P->ss[1])
        {
            printf("ERROR: Image too small (%s)", fname.c_str());
            printf("    Im size = [%d,%d]\n", sz[1], sz[0]);
            return;
        }

        Scalar m_sc;
        float sc;
        int sy, sx, yo, xo;
        float r = (float(sz[1])/sz[0]) / (float(P->ss[1])/P->ss[0]);

        // Create and process N output images from on input image
        for (int k=0; k<N; k++)
        {
            // Image cropping and random flipping
            sc = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
            sx = max(float(P->ss[1]), min(float(sz[1])*min(1.0f,1.0f/r), sc*(P->imsc[1]*sz[1]-P->imsc[0]*sz[1]) + P->imsc[0]*sz[1]));
            sy = max(float(P->ss[0]), min(float(sz[0])*min(1.0f,r), sx*(float(P->ss[0])/P->ss[1])));
            yo = sy < sz[0] ? rand() % (sz[0] - sy) : 0;
            xo = sx < sz[1] ? rand() % (sz[1] - sx) : 0;
            Mat roi(host_im, Rect(xo, yo, sx, sy));
            resize(roi, host_roi, Size(P->ss[1], P->ss[0]), 0, 0, INTER_LINEAR);
            if ((static_cast <float> (rand()) / static_cast <float> (RAND_MAX)) < 0.5f)
                flip(host_roi, host_roi, 1);
            max(host_roi, 1e-5, host_roi);
            min(host_roi, 1e5, host_roi);

            // If input is display-reffered, we need to linearize assuming a certain camera curve
            if (P->linearize)
            {
                host_roi = host_roi.mul(1.0 / (1.6*Scalar(1,1,1) - host_roi));
                multiply(host_roi, 0.6*Scalar(1,1,1), host_roi);
                pow(host_roi, 1.0/0.9, host_roi);
            }

            try 
            {
                // Random saturation treshold
                sc = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
                float cl = P->clip[0] + sc*(P->clip[1]-P->clip[0]);

                // Histogram in gamma corrected domain
                int i;
                float gamma = 0.5;
                Mat bw = host_roi.reshape(1);
                double mi, ma;
                minMaxLoc(bw, &mi, &ma);
                ma += 1e-5;
                mi = pow(mi, gamma);
                ma = pow(ma, gamma);
                for (i=0; i<=N_bins; i++)
                    range[i] = pow(mi + (ma-mi)*float(i)/N_bins, 1.0f/gamma);
                const float* ranges[] = {range};
                int ch = 0;
                MatND hist;
                calcHist(&bw, 1, &ch, Mat(), hist, 1, &N_bins, ranges, false, false);

                // Using the histogram, we can find a scaling that will anchor
                // the percentile cl to 1, where cl is a random selection between
                // P->clip[0] and P->clip[1]. This means that upon clipping the scaled
                // image at 1, there will be 100(1-cl) pixels that are saturated in the
                // final LDR images.
                float c = 0, c0;
                for (i=0; i<N_bins; i++)
                {
                    c0 = c;
                    c += hist.at<float>(i)/(P->ss[0]*P->ss[1]*P->ss[2]);
                    if (c >= cl && i>0)
                        break;
                }
                
                // Apply the scaling
                float a = max(0.0f, min(1.0f, (cl-c0)/(c-c0)));
                sc = a*range[min(i+1,N_bins)]+(1-a)*range[i];
                multiply(host_roi, (1.0f/sc) * Scalar(1,1,1), host_roi);

                // Add gaussian noise, with random variance between P->noise[0] and P->noise[1]
                sc = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
                Mat noise = Mat(host_roi.size(),CV_32FC3);
                cv::randn(noise, Scalar(0,0,0), (P->noise[0] + sc*(P->noise[1]-P->noise[0])) * Scalar(1,1,1));
                host_roi = max(0, host_roi + noise);

                // Random changes in hue and saturation
                if (P->hue[1] > 0 || P->sat[1] > 0)
                {
                    pow(host_roi, 0.5, host_roi);

                    float hue = P->distribution_hue(P->generator);
                    float sat = P->distribution_sat(P->generator);
                    cvtColor(host_roi,host_roi,CV_RGB2HSV);

                    std::vector<cv::Mat> planes(3);
                    cv::split(host_roi, planes);
                    planes[0] += hue;
                    planes[1] += sat;
                    merge(planes,host_roi);
                    cvtColor(host_roi,host_roi,CV_HSV2RGB);

                    pow(host_roi, 2.0, host_roi);
                }

                // Remove extreme values
                minMaxLoc(host_roi, &mi, &ma);
                assert(ma > mi && ma > 0);
                max(host_roi, 1e-5, host_roi);
                min(host_roi, 1e10, host_roi);

                // Print output HDR image
                printHDR(k+1);

                // Apply camera curve
                sigmoid_n = min(2.5f, max(0.2f, P->distribution_sign(P->generator)));
                sigmoid_a = min(5.0f, max(0.0f, P->distribution_siga(P->generator)));
                pow(host_roi, sigmoid_n, host_roi);
                host_roi = host_roi.mul(1.0 / (host_roi + sigmoid_a*Scalar(1,1,1)));
                multiply(host_roi, (1.0 + sigmoid_a)*Scalar(1,1,1), host_roi);

                // Print output LDR image
                printLDR(k+1);
            }
            catch (exception& e)
            {
                cout << e.what() << endl;
                printf("%s\n", fname.c_str());
                printf("%d: Im size = [%d,%d], Crop = [%d,%d,%d,%d]\n", k, sz[1], sz[0], xo, yo, sx, sy);
            }
        }
    }
    
    Mat host_im, host_roi, host_roi8;
    Params *P;
    string fname;
    int N, K, sz[3], N_bins;
    float *range, sigmoid_n, sigmoid_a;
};


int main(int argc, char *argv[])
{
    Params P;
    vector<string> fnames;
    bool verbose = false, rand_sampl = true;

    // Read input parameters from command line
    VCUtil::readArgument(argc, argv, "-randomseed", &rand_sampl);
    VCUtil::readArgument(argc, argv, "-imsize", P.ss, 3);
    VCUtil::readArgument(argc, argv, "-input_path", &P.ipath);
    VCUtil::readArgument(argc, argv, "-output_path", &P.opath);
    VCUtil::readArgument(argc, argv, "-subimages", &P.K);
    VCUtil::readArgument(argc, argv, "-cropscale", P.imsc, 2);
    VCUtil::readArgument(argc, argv, "-clip", P.clip, 2);
    VCUtil::readArgument(argc, argv, "-noise", P.noise, 2);
    VCUtil::readArgument(argc, argv, "-hue", P.hue, 2);
    VCUtil::readArgument(argc, argv, "-sat", P.sat, 2);
    VCUtil::readArgument(argc, argv, "-sigmoid_n", P.sigmoid_n, 2);
    VCUtil::readArgument(argc, argv, "-sigmoid_a", P.sigmoid_a, 2);
    VCUtil::readArgument(argc, argv, "-jpeg_quality", &P.jpgQ);
    VCUtil::readArgument(argc, argv, "-verbose", &verbose);
    VCUtil::readArgument(argc, argv, "-linearize", &P.linearize);

    // Setup normal distributions
    P.distribution_hue = normal_distribution<float>(P.hue[0],P.hue[1]);
    P.distribution_sat = normal_distribution<float>(P.sat[0],P.sat[1]);
    P.distribution_sign = normal_distribution<float>(P.sigmoid_n[0],P.sigmoid_n[1]);
    P.distribution_siga = normal_distribution<float>(P.sigmoid_a[0],P.sigmoid_a[1]);


    printf("\n\nSettings of data augmentation by means of virtual camera:\n");
    printf("\tInput path:         %s\n", P.ipath.c_str());
    printf("\tOutput path:        %s\n", P.opath.c_str());
    printf("\tOutput size:        %dx%dx%d\n", P.ss[0], P.ss[1], P.ss[2]);
    printf("\tCrop:               %0.2f-%0.2f%%\n", P.imsc[0], P.imsc[1]);
    printf("\tClipping range:     %0.2f-%0.2f%%\n", 100*(1-P.clip[1]), 100*(1-P.clip[0]));
    printf("\tSigmoidal camera curve:\n");
    printf("\t\tSigmoid n mean/std:    %0.4f/%0.4f\n", P.sigmoid_n[0], P.sigmoid_n[1]);
    printf("\t\tSigmoid a mean/std:    %0.4f/%0.4f\n", P.sigmoid_a[0], P.sigmoid_a[1]);
    printf("\tNoise std:          %0.4f-%0.4f\n", P.noise[0], P.noise[1]);
    printf("\tHue mean/std:       %0.4f/%0.4f\n", P.hue[0], P.hue[1]);
    printf("\tSat mean/std:       %0.4f/%0.4f\n", P.sat[0], P.sat[1]);
    printf("\tMin JPEG quality:   %d\n", P.jpgQ);
    printf("\tSub images:         %d (in a 1 mega-pixel image)\n", P.K);
    printf("\tLinearize input:    %d\n", P.linearize);
    printf("\tSeed random gen.:   %d\n\n", rand_sampl);

    // Seed random generator?
    if (rand_sampl)
        srand (time(NULL));

    // For timings
    struct timespec start, finish;
    double elapsed;

    // Locate all the input images
    VCUtil::findFiles(P.ipath.c_str(), &fnames, verbose);
    std::sort(fnames.begin(), fnames.end());

    printf("Total number of files to process: %d\n", int(fnames.size()));

    
    // Create output directories if do not exist
    if (!VCUtil::createDir(P.opath.c_str())) return -1;
    char str[500];
    sprintf(str, "%s/bin", P.opath.c_str());
    if (!VCUtil::createDir(str)) return -1;
    sprintf(str, "%s/jpg", P.opath.c_str());
    if (!VCUtil::createDir(str)) return -1;


    clock_gettime(CLOCK_MONOTONIC, &start);
    fprintf(stderr, "Processing ");

    // Data processing loop
    for (unsigned int k=0; k<fnames.size(); k++)
    {
        if ((k % max(1, int(fnames.size()/20))) == 0 && k>0)
            fprintf(stderr, ".");
        
        // Augment the image by means of the virtual camera
        VirtualCamera bp(fnames.at(k), k+1, &P);
        bp.run();
    }
    printf(" done\n");

    clock_gettime(CLOCK_MONOTONIC, &finish);
    elapsed = (finish.tv_sec - start.tv_sec);
    elapsed += (finish.tv_nsec - start.tv_nsec) / 1000000000.0;

    printf("Total time = %f seconds\n", elapsed);
    

    return 1;
}
