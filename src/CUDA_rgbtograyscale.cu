#include <iostream>
#include <opencv2/cudev.hpp>
#include <device_launch_parameters.h>
#include "opencv2/opencv.hpp"

__global__ void rgbtogray(const cv::cudev::PtrStepSz<uchar3> src, cv::cudev::PtrStepSz<uchar> dst)
{
    const int x = blockDim.x * blockIdx.x + threadIdx.x;
	const int y = blockDim.y * blockIdx.y + threadIdx.y;

    if(x < src.cols && y < src.rows)
    {
        dst(y, x) = (0.114*src(y, x).x + 0.587*src(y, x).y + 0.299*src(y, x).z);
    }
}

int main (int argc, char* argv[])
{
    cv::Mat img = cv::imread("/home/srinath/FYP/tig.jpg", 1);
    cv::cuda::GpuMat src;
    cv::cuda::GpuMat dst(img.rows, img.cols, CV_8UC1);
    src.upload(img);

    // const dim3 block(64, 2);
	// const dim3 grid(cv::cudev::divUp(img.cols, block.x), cv::cudev::divUp(img.rows, block.y));
    const dim3 block(16,16);

	// Calculate grid size to cover the whole image
	const dim3 grid(cv::cudev::divUp(img.cols, block.x), cv::cudev::divUp(img.rows, block.y));
    std::cout << std::endl << grid.x << "  " << grid.y << std::endl;
    rgbtogray<<<grid, block>>>(src, dst);

    CV_CUDEV_SAFE_CALL(cudaGetLastError());
	CV_CUDEV_SAFE_CALL(cudaDeviceSynchronize());

    cv::Mat result;
    dst.download(result);

    cv::imshow("result", result);
    cv::waitKey();
    
}