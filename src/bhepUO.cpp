#include"opencv2/opencv.hpp"
#include<chrono>

using namespace std;
using namespace cv;

VideoCapture cap("input1n.mp4");

int main()
{
    auto start = std::chrono::high_resolution_clock::now();

    int hist[256],mean,fcount=0,tl,tu,f[256];
    float cdf[256],pdf[256],sl=0,su=0,s1,s2;
    Mat input;
    //cap.read(input);
    input=imread("input/lowcont.jpg");
    int limit=cap.get(CV_CAP_PROP_FRAME_COUNT);
    limit=1;
    Mat output(input.rows,input.cols,CV_8UC1);
    Mat gray(input.rows,input.cols,CV_8UC1);
    
    
    while(waitKey(40)!=27)
    {

    if(fcount==limit)
    {
        break;
    }

    //cap>>input;
    
    for(int i=0;i<256;i++)
    {
        hist[i]=0;
        pdf[i]=0;
        cdf[i]=0;
        f[i]=0;
    }
    sl=0;su=0;mean=0;s1=0;s2=0;
    
    for(int i=0;i<input.rows;i++)
    {
        for(int j=0;j<input.cols;j++)
        {
            int b=input.at<Vec3b>(i,j)[0];
            int g=input.at<Vec3b>(i,j)[1];
            int r=input.at<Vec3b>(i,j)[2];
            gray.at<uchar>(i,j)=(b+g+r)/3;
        }
    }

    for(int i=0;i<input.rows;i++)
    {
        for(int j=0;j<input.cols;j++)
        {
            hist[gray.at<uchar>(i,j)]++;
        }
    }

    //for(int i=0;i<256;i++)
    //{
    //    mean+=i;
    //}

    //mean=(input.rows*input.cols)/mean;
    mean=127;
    
    for(int i=0;i<256;i++)
    {
        if(i<=mean)
        {
            sl=sl+hist[i];
        }
        else
        {
            su=su+hist[i];
        }
    }

    tl=sl/(mean+1);
    tu=su/(255-mean);

    for(int i=0;i<256;i++)
    {
        if(i<=mean)
        {
            if(hist[i]>tl)
            {
                hist[i]=tl;
            }
        }
        else
        {
            if(hist[i]>tu)
            {
                hist[i]=tu;
            }
        }
    }

    for(int i=0;i<256;i++)
    {
        if(i<=mean)
        {
            s1=s1+hist[i];
        }
        else
        {
            s2=s2+hist[i];
        }
    }

    for(int i=0;i<256;i++)
    {
        if((i<=mean)&(s1!=0))
        {
            pdf[i]=hist[i]/sl;
        }
        else if((i>mean)&(s2!=0))
        {
            pdf[i]=hist[i]/su;
        }
        else
        {
            pdf[i]=0;
        }
    }

    cdf[0]=pdf[0];
    for(int i=1;i<256;i++)
    {
        cdf[i]=cdf[i-1]+pdf[i];
        if(i==(mean+1))
        {
            cdf[i]=pdf[i];
        }
    }

    for(int i=0;i<256;++i)
            {
                if(i<=mean)
                {
                    f[i]=mean*(cdf[i]-0.5*pdf[i]);
                }
                else
                {
                    f[i]=(mean+1)+(256-mean)*(cdf[i]-0.5*pdf[i]);
                }
                
            }

    for(int i=0;i<input.rows;i++)
    {
        for(int j=0;j<input.cols;j++)
        {
            /*if(i<=mean)
            {
                output.at<uchar>(i,j)=mean*(cdf[gray.at<uchar>(i,j)]-0.5*pdf[gray.at<uchar>(i,j)]);
            }            
            else
            {
                output.at<uchar>(i,j)=(mean+1)+(255-mean+1)*(cdf[gray.at<uchar>(i,j)]-0.5*pdf[gray.at<uchar>(i,j)]);
            }*/
            output.at<uchar>(i,j)==f[gray.at<uchar>(i,j)];
        }
    }
    imshow("ut",output);
    
    imwrite("bheplow.jpg",output);

    fcount=fcount+1;

    }

    auto stop = std::chrono::high_resolution_clock::now();
    int time=std::chrono::duration_cast<chrono::milliseconds>(stop - start).count();
    cout<<"Execute "<<fcount<<" frames in "<<time<<"ms"<<endl;

    return 0;
}
