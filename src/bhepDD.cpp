#include"opencv2/opencv.hpp"
#include<chrono>
#include<thread>

using namespace std;
using namespace cv;

VideoCapture cap("input/sample1.mp4");
int r=cap.get(CV_CAP_PROP_FRAME_HEIGHT);
int c=cap.get(CV_CAP_PROP_FRAME_WIDTH);
Mat input(r,c,CV_8UC3);
int limit=cap.get(CV_CAP_PROP_FRAME_COUNT);
Mat output(r,c,CV_8UC1);
Mat gray(r,c,CV_8UC1);
bool p1=false,p2=false,p3=false,done=true;

void part1()
{
    int b,g,r,hist[256],mean=127,p1count=0,f[256],tl,tu,a=(input.rows/3);
    float cdf[256],pdf[256],sl=0,su=0,s1=0,s2=0;

    while(done==true)
    {
        if(p1==true)
        {

        for(int i=0;i<256;++i)
        {
            hist[i]=0;
            pdf[i]=0;
            cdf[i]=0;
        }
        sl=0;su=0;s1=0;s2=0;

        for(int i=0;i<a;++i)
        {
            for(int j=0;j<input.cols;++j)
            {
                hist[gray.at<uchar>(i,j)]++;
            }
        }

        for(int i=0;i<256;++i)
        {
            if(i<=mean)
            {
                sl+=hist[i];
            }
            else
            {
                su+=hist[i];
            }
        }

        tl=sl/(mean+1);
        tu=su/(255-mean);

        for(int i=0;i<256;++i)
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

        for(int i=0;i<256;++i)
        {
            if(i<=mean)
            {
                s1+=hist[i];
            }
            else
            {
                s2+=hist[i];
            }
        }

        for(int i=0;i<256;++i)
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
        for(int i=1;i<256;++i)
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

        for(int i=0;i<(a);++i)
        {
            for(int j=0;j<input.cols;++j)
            {
                output.at<uchar>(i,j)=f[gray.at<uchar>(i,j)];
            }
        }
        for(int j=0;j<input.cols;++j)
        {
            output.at<uchar>(a-1,j)=0.5*(f[gray.at<uchar>(a-1,j)]+f[gray.at<uchar>(a,j)]);
        }
        p1count++;
        p1=false;
        }
    }
    cout<<"Part1 done"<<endl;
}

void part2()
{
    int b,g,r,hist[256],mean=127,tl,tu,f[256],p2count=0,u=(input.rows/3)-1,l=(2*input.rows/3);
    float cdf[256],pdf[256],mbe[256],sl=0,su=0,s1,s2;

    while(done==true)
    {
        if(p2==true)
        {

        for(int i=0;i<256;++i)
        {
            hist[i]=0;
            pdf[i]=0;
            cdf[i]=0;
        }
        sl=0;su=0;s1=0;s2=0;

        for(int i=u;i<(l);++i)
        {
            for(int j=0;j<input.cols;++j)
            {
                hist[gray.at<uchar>(i,j)]++;
            }
        }

        for(int i=0;i<256;++i)
        {
            if(i<=mean)
            {
                sl+=hist[i];
            }
            else
            {
                su+=hist[i];
            }
        }

        tl=sl/(mean+1);
        tu=su/(255-mean);

        for(int i=0;i<256;++i)
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

        for(int i=0;i<256;++i)
        {
            if(i<=mean)
            {
                s1+=hist[i];
            }
            else
            {
                s2+=hist[i];
            }
        }

        for(int i=0;i<256;++i)
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
        for(int i=1;i<256;++i)
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

        for(int i=u;i<(l);++i)
        {
            for(int j=0;j<input.cols;++j)
            {
                output.at<uchar>(i,j)=f[gray.at<uchar>(i,j)];
            }
        }
        for(int j=0;j<input.cols;++j)
        {
            output.at<uchar>(u,j)=0.5*(f[gray.at<uchar>(u,j)]+f[gray.at<uchar>(u-1,j)]);
            output.at<uchar>(l-1,j)=0.5*(f[gray.at<uchar>(l-1,j)]+f[gray.at<uchar>(l,j)]);
        }
        p2count++;
        p2=false;
        }
    }
    cout<<"Part2 done"<<endl;
}

void part3()
{
    int b,g,r,hist[256],mean=127,tl,tu,f[256],p3count=0,u=(2*input.rows/3)-1;
    float cdf[256],pdf[256],mbe[256],s1,s2,sl=0,su=0;

    while(done==true)
    {
        if(p3==true)
        {

        for(int i=0;i<256;++i)
        {
            hist[i]=0;
            pdf[i]=0;
            cdf[i]=0;
        }
        sl=0;su=0;s1=0;s2=0;

        for(int i=(u);i<input.rows;++i)
        {
            for(int j=0;j<input.cols;++j)
            {
                hist[gray.at<uchar>(i,j)]++;
            }
        }

        for(int i=0;i<256;++i)
        {
            if(i<=mean)
            {
                sl+=hist[i];
            }
            else
            {
                su+=hist[i];
            }
        }

        tl=sl/(mean+1);
        tu=su/(255-mean);

        for(int i=0;i<256;++i)
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

        for(int i=0;i<256;++i)
        {
            if(i<=mean)
            {
                s1+=hist[i];
            }
            else
            {
                s2+=hist[i];
            }
        }

        for(int i=0;i<256;++i)
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
        for(int i=1;i<256;++i)
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

        for(int i=(u);i<input.rows;++i)
        {
            for(int j=0;j<input.cols;++j)
            {
                output.at<uchar>(i,j)=f[gray.at<uchar>(i,j)];
            }
        }
        for(int j=0;j<input.cols;++j)
        {
            output.at<uchar>(u,j)=0.5*(f[gray.at<uchar>(u,j)]+f[gray.at<uchar>(u-1,j)]);
        }
        p3count++;
        p3=false;
        }
    }
    cout<<"Part3 done"<<endl;
}

int main()
{
    auto start = std::chrono::high_resolution_clock::now();
    int fcount=0,a=(1/3)*input.rows,b=(2/3)*input.rows;
    thread t0(part1);
    thread t1(part2);
    thread t2(part3);

    while(waitKey(40)!=27)
    {

    if(fcount==limit)
    {
        done=false;
        break;
    }

    cap>>input;
    
    if(input.rows!=0 & input.cols!=0)
    {
        for(int i=0;i<input.rows;++i)
        {
            for(int j=0;j<input.cols;++j)
            {
                int b=input.at<Vec3b>(i,j)[0];
                int g=input.at<Vec3b>(i,j)[1];
                int r=input.at<Vec3b>(i,j)[2];
                gray.at<uchar>(i,j)=(b+g+r)/3;
            }
        }

        p1=p2=p3=true;

        while(p1==true || p2==true || p3==true)
        {}
        
        imshow("out",output);
    }

    fcount++;

    }
    t0.join();
    t1.join();
    t2.join();


    auto stop = std::chrono::high_resolution_clock::now();
    int time=std::chrono::duration_cast<chrono::milliseconds>(stop - start).count();
    cout<<"Execute "<<fcount<<" frames in "<<time<<"ms";

    return 0;
}
