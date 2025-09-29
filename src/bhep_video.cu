#include "opencv2/opencv.hpp"
#include <iostream>
#include <opencv2/cudev.hpp>
#include <device_launch_parameters.h>
#include <math.h>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/core/cuda/common.hpp>
#include "opencv2/imgcodecs.hpp"
#include "opencv2/cudaarithm.hpp"
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <cuda_runtime.h>
#include <chrono>
#include <vector>
#include <stdlib.h>

using namespace std;
using namespace cv;

void BHEP(string path)
{
    Mat input, out;
    int frames = 0;
    VideoCapture cap(path);
    cap >> input;
    Mat gray(input.rows, input.cols, CV_8UC1);
    Mat output(input.rows, input.cols, CV_8UC1);
    int size = input.rows * input.cols;

    int b, g, r, hist[256], mean = 0, tl, tu, sl = 0, su = 0, f[256];
    float pdf[256], cdf[256];

    int frame_width = cap.get(CAP_PROP_FRAME_WIDTH);
    int frame_height = cap.get(CAP_PROP_FRAME_HEIGHT);
    VideoWriter outvid("BHEP.avi", VideoWriter::fourcc('M', 'J', 'P', 'G'), 30, Size(frame_width, frame_height), false);
    auto start = chrono::steady_clock::now();
    while (waitKey(27) != 27)
    {
        ++frames;
        cap >> input;
        if (input.empty())
            break;
        for (int i = 0; i < 256; i++)
            hist[i] = 0;
        for (int i = 0; i < input.rows; i++)
        {
            for (int j = 0; j < input.cols; j++)
            {
                b = input.at<Vec3b>(i, j)[0];
                g = input.at<Vec3b>(i, j)[1];
                r = input.at<Vec3b>(i, j)[2];
                gray.at<uchar>(i, j) = (b + g + r) / 3;
                hist[gray.at<uchar>(i, j)]++;
            }
        }
        cvtColor(input, input, COLOR_RGB2HLS);
        for (int i = 0; i < 256; i++)
            mean += i;
        mean = mean / 256;
        int sum = 0;
        for (int i = 0; i < 256; i++)
        {
            if (i <= mean)
                sl += hist[i];
            else
                su += hist[i];
            sum += hist[i];
        }
        tl = sl / (mean + 1);
        tu = su / (255 - mean);
        for (int i = 0; i < 256; i++)
        {
            if (i <= mean)
            {
                if (hist[i] > tl)
                    hist[i] = tl;
            }
            else
            {
                if (hist[i] > tu)
                    hist[i] = tu;
            }
        }
        float s1 = 0, s2 = 0;
        // waitKey(20);
        for (int i = 0; i < 256; i++)
        {
            if (i <= mean)
                s1 += hist[i];
            else
                s2 += hist[i];
        }
        for (int i = 0; i < 256; i++)
        {
            if ((i <= mean) & (s1 != 0))
                pdf[i] = hist[i] / s1;
            else if ((i > mean) & (s2 != 0))
                pdf[i] = hist[i] / s2;
            else
                pdf[i] = 0;
        }
        cdf[0] = pdf[0];
        cdf[mean + 1] = pdf[mean + 1];
        for (int i = 1; i < 256; i++)
        {
            if (i <= mean)
                cdf[i] = cdf[i - 1] + pdf[i];
            if (i > mean + 1)
                cdf[i] = cdf[i - 1] + pdf[i];
            //cout<<cdf[i]<<endl;
        }
        for (int i = 0; i < 256; i++)
        {
            if (i <= mean)
                f[i] = mean * (cdf[i] - 0.5 * pdf[i]);
            else
                f[i] = mean + 1 + (255 - (mean + 1)) * (cdf[i] - 0.5 * pdf[i]);
            //cout<<f[i]<<endl;
        }
        input.copyTo(out);
        cvtColor(out, out, COLOR_HLS2RGB);
        for (int i = 0; i < input.rows; i++)
        {
            for (int j = 0; j < input.cols; j++)
            {
                input.at<Vec3b>(i, j)[1] = f[gray.at<uchar>(i, j)];
            }
        }
        cvtColor(input, input, COLOR_HLS2RGB);
        // imshow("BHEP Output", input);
        // imshow("Input", out);
        outvid.write(output);
    }
    auto end = chrono::steady_clock::now();

    cout << "Elapsed time in nanoseconds: "
         << chrono::duration_cast<chrono::nanoseconds>(end - start).count()
         << " ns" << endl;

    cout << "Elapsed time in microseconds: "
         << chrono::duration_cast<chrono::microseconds>(end - start).count()
         << " µs" << endl;

    cout << "Elapsed time in milliseconds: "
         << chrono::duration_cast<chrono::milliseconds>(end - start).count()
         << " ms" << endl;

    cout << "Elapsed time in seconds: "
         << chrono::duration_cast<chrono::seconds>(end - start).count()
         << " sec";
    cout << frames << endl;
}

__global__ void rgbtogray(const cv::cudev::PtrStepSz<uchar3> src, cv::cudev::PtrStepSz<uchar> dst, cv::cudev::PtrStepSz<int> histo)
{
    const int x = blockDim.x * blockIdx.x + threadIdx.x;
    const int y = blockDim.y * blockIdx.y + threadIdx.y;

    if (x < src.cols && y < src.rows)
    {
        dst(y, x) = (src(y, x).x + src(y, x).y + src(y, x).z) / 3;
        atomicAdd(&histo(dst(y, x), 0), 1);
    }
}

__global__ void calcLimits(cv::cudev::PtrStepSz<int> histo, float *sl, float *su)
{
    const int x = blockDim.x * blockIdx.x + threadIdx.x;

    if (x < 256)
    {
        if (x < 128)
            atomicAdd(sl, histo(0, x));
        else
            atomicAdd(su, histo(0, x));
    }
}

__global__ void thresh(cv::cudev::PtrStepSz<int> hist, float *tl, float *tu)
{
    const int x = blockDim.x * blockIdx.x + threadIdx.x;
    if (x < 128)
    {
        if (hist(0, x) > int(*tl))
            hist(0, x) = *tl;
    }
    else
    {
        if (hist(0, x) > int(*tu))
            hist(0, x) = *tu;
    }
}

__global__ void calcPDF(cv::cudev::PtrStepSz<float> pdf, cv::cudev::PtrStepSz<int> hist, float *sl, float *su)
{
    const int x = blockDim.x * blockIdx.x + threadIdx.x;

    if (x < 128 & *sl != 0.0)
        pdf(0, x) = hist(0, x) / (*sl);
    else if (x > 127 & *su != 0.0)
        pdf(0, x) = hist(0, x) / (*su);
    else
        pdf(0, x) = 0;
}

__global__ void transformFunction(cv::cudev::PtrStepSz<float> pdf, cv::cudev::PtrStepSz<float> cdf)
{
    const int x = blockDim.x * blockIdx.x + threadIdx.x;

    if (x < 256)
    {
        if (x < 128)
            cdf(0, x) = 127 * (cdf(0, x) - 0.5 * pdf(0, x));
        else
            cdf(0, x) = 128 + 127 * (cdf(0, x) - 0.5 * pdf(0, x));
    }
}

__global__ void transformImage(cv::cudev::PtrStepSz<uchar> dst, cv::cudev::PtrStepSz<float> f)
{
    const int x = blockDim.x * blockIdx.x + threadIdx.x;
    const int y = blockDim.y * blockIdx.y + threadIdx.y;

    if (x < dst.cols && y < dst.rows)
    {
        dst(y, x) = f(dst(y, x), 0);
    }
}

int main()
{
    BHEP("input2n.mp4");
    cv::VideoCapture cap("input1n.mp4");
    cv::Mat img;
    cap.read(img);
    cv::Mat output(img.rows, img.cols, CV_8UC1);
    cv::Mat hist(256, 1, CV_32SC1), pdf(256, 1, CV_32FC1), cdf(256, 1, CV_32FC1);
    float *sl, *su, *tl, *tu;

    cv::cuda::GpuMat src;
    cv::cuda::GpuMat dst(img.rows, img.cols, CV_8UC1), gpu_hist(256, 1, CV_32SC1), gpu_pdf(256, 1, CV_32FC1), gpu_cdf(256, 1, CV_32FC1);
    float *gpu_f1, *gpu_f2;

    const dim3 block(16, 16);
    const dim3 grid(cv::cudev::divUp(img.cols, block.x), cv::cudev::divUp(img.rows, block.y));

    sl = (float *)malloc(sizeof(int));
    su = (float *)malloc(sizeof(int));
    tl = (float *)malloc(sizeof(int));
    tu = (float *)malloc(sizeof(int));

    cudaMalloc((void **)&gpu_f1, sizeof(float));
    cudaMalloc((void **)&gpu_f2, sizeof(float));
    auto start = chrono::steady_clock::now();
    int f = 0;

    while (cv::waitKey(40) != 27)
    {
        f++;
        cap >> img;

        if (img.cols == 0 || img.rows == 0)
        {
            break;
        }

        for (int i = 0; i < 256; i++)
        {
            // std::cout << pdf.at<float>(i, 0) << std::endl;
            cdf.at<float>(i, 0) = 0;
            pdf.at<float>(i, 0) = 0;
            hist.at<float>(i, 0) = 0;
        }

        *sl = 0;
        *su = 0;
        *tl = 0;
        *tu = 0;

        cudaMemcpy(gpu_f1, sl, sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(gpu_f2, su, sizeof(float), cudaMemcpyHostToDevice);

        src.upload(img);

        rgbtogray<<<grid, block>>>(src, dst, gpu_hist);

        dst.download(output);

        // cv::imshow("input", output);

        gpu_hist.download(hist);

        calcLimits<<<1, 256>>>(gpu_hist, gpu_f1, gpu_f2);

        cudaMemcpy(sl, gpu_f1, sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(su, gpu_f2, sizeof(float), cudaMemcpyDeviceToHost);

        *tl = *sl / 128;
        *tu = *su / 127;

        cudaMemcpy(gpu_f1, tl, sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(gpu_f2, tu, sizeof(float), cudaMemcpyHostToDevice);

        thresh<<<1, 256>>>(gpu_hist, gpu_f1, gpu_f2);

        *sl = 0.0;
        *su = 0.0;
        cudaMemcpy(gpu_f1, sl, sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(gpu_f2, su, sizeof(float), cudaMemcpyHostToDevice);

        calcLimits<<<1, 256>>>(gpu_hist, gpu_f1, gpu_f2);

        cudaMemcpy(sl, gpu_f1, sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(su, gpu_f2, sizeof(float), cudaMemcpyDeviceToHost);

        calcPDF<<<1, 256>>>(gpu_pdf, gpu_hist, gpu_f1, gpu_f2);

        gpu_pdf.download(pdf);

        cdf.at<float>(0, 0) = pdf.at<float>(0, 0);
        cdf.at<float>(127, 0) = pdf.at<float>(127, 0);

        for (int i = 1; i < 256; i++)
        {
            if (i < 127)
                cdf.at<float>(i, 0) = cdf.at<float>(i - 1, 0) + pdf.at<float>(i, 0);
            if (i > 127)
                cdf.at<float>(i, 0) = cdf.at<float>(i - 1, 0) + pdf.at<float>(i, 0);
            // std::cout << cdf.at<float>(i - 1, 0) << std::endl;
        }

        gpu_cdf.upload(cdf);

        transformFunction<<<1, 256>>>(gpu_pdf, gpu_cdf);

        transformImage<<<grid, block>>>(dst, gpu_cdf);

        dst.download(output);

        // cv::imshow("Output", output);
    }
    auto end = chrono::steady_clock::now();

    cout << f << " Elapsed time in nanoseconds: "
         << chrono::duration_cast<chrono::nanoseconds>(end - start).count()
         << " ns" << endl;

    cout << "Elapsed time in microseconds: "
         << chrono::duration_cast<chrono::microseconds>(end - start).count()
         << " µs" << endl;

    cout << "Elapsed time in milliseconds: "
         << chrono::duration_cast<chrono::milliseconds>(end - start).count()
         << " ms" << endl;

    cout << "Elapsed time in seconds: "
         << chrono::duration_cast<chrono::seconds>(end - start).count()
         << " sec";

    // cv::waitKey(0);

    return 0;
}