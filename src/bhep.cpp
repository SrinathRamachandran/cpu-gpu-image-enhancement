#include"opencv2/opencv.hpp"

using namespace std;
using namespace cv;

Mat input=imread("input/lowcont.jpeg");
Mat gray(input.rows,input.cols,CV_8UC1);
Mat output(input.rows,input.cols,CV_8UC1);
int size=input.rows*input.cols;

int main()
{
    int b,g,r,hist[256],mean=0,tl,tu,sl=0,su=0,f[256];
    float pdf[256],cdf[256];
    for(int i=0;i<256;i++)hist[i]=0;
    for(int i=0;i<input.rows;i++)
    {
        for(int j=0;j<input.cols;j++)
        {
            b=input.at<Vec3b>(i,j)[0];
            g=input.at<Vec3b>(i,j)[1];
            r=input.at<Vec3b>(i,j)[2];
            gray.at<uchar>(i,j)=(b+g+r)/3;
            hist[gray.at<uchar>(i,j)]++;
        }
    }
    for(int i=0;i<256;i++)mean+=i;
    mean=mean/256;int sum=0;
    for(int i=0;i<256;i++)
    {
        if(i<=mean)sl+=hist[i];
        else su+=hist[i];
        sum+=hist[i];
    }
    tl=sl/(mean+1);
    tu=su/(255-mean);
    for(int i=0;i<256;i++)
    {
        if(i<=mean)
        {
            if(hist[i]>tl)hist[i]=tl;
        }
        else
        {
            if(hist[i]>tu)hist[i]=tu;
        }
    }
    float s1=0,s2=0;
    for(int i=0;i<256;i++)
    {
        if(i<=mean)s1+=hist[i];
        else s2+=hist[i];
    }
    for(int i=0;i<256;i++)
    {
        if((i<=mean)&(s1!=0))pdf[i]=hist[i]/s1;
        else if((i>mean)&(s2!=0))pdf[i]=hist[i]/s2;
        else pdf[i]=0;      
    }
    cdf[0]=pdf[0];cdf[mean+1]=pdf[mean+1];
    for(int i=1;i<256;i++)
    {
        if(i<=mean)cdf[i]=cdf[i-1]+pdf[i];
        if(i>mean+1)cdf[i]=cdf[i-1]+pdf[i];
        //cout<<cdf[i]<<endl;
    }
    for(int i=0;i<256;i++)
    {
        if(i<=mean)f[i]=mean*(cdf[i]-0.5*pdf[i]);
        else f[i]=mean+1+(255-(mean+1))*(cdf[i]-0.5*pdf[i]);
        //cout<<f[i]<<endl;
    }
    for(int i=0;i<input.rows;i++)
    {
        for(int j=0;j<input.cols;j++)
        {
            output.at<uchar>(i,j)=f[gray.at<uchar>(i,j)];
        }
    }
    imwrite("bheplow.jpeg",output);
    //imwrite("gray.jpg",gray);
    return 0;
}
