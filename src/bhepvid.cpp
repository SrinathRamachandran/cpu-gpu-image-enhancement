#include"opencv2/opencv.hpp"
#include<thread>
#include<math.h>
#include<queue>
#include<mutex>

using namespace std;
using namespace cv;

VideoCapture cap("input1n.mp4");
int fw = cap.get(CV_CAP_PROP_FRAME_WIDTH); 
int fh = cap.get(CV_CAP_PROP_FRAME_HEIGHT);
mutex m1,m2,m3,m4;
queue <Mat> iq,gq,oq;
queue <int> tlq,tuq;
bool x,a,b;
int s2=0;
VideoWriter video("output/bhepinput1n.avi", CV_FOURCC('M', 'J', 'P', 'G'), 15, Size(fw,fh ), 0);

void part1()
{
    Mat input(fh,fw,CV_8UC3);
    Mat gray(fh,fw,CV_8UC1);
    int b,g,r;int tl,tu,hist[256],sl,su,mean=127;
    while(true)
    {
        if(iq.size()>0)
        {
            m2.lock();
            input=iq.front();
            iq.pop();
            m2.unlock();
            for(int i=0;i<256;i++)hist[i]=0;
            sl=0;su=0;
            for(int i=0;i<input.rows;i++)
            {
                for(int j=0;j<input.cols;j++)
                {
                    b=input.at<Vec3b>(i,j)[0];
                    g=input.at<Vec3b>(i,j)[1];
                    r=input.at<Vec3b>(i,j)[2];
                    gray.at<uchar>(i,input.cols-j)=(b+g+r)/3;
                    hist[gray.at<uchar>(i,input.cols-j)]++;
                }
            }
            for(int i=0;i<256;i++)
            {
                if(i<=mean)sl+=hist[i];
                else su+=hist[i];
            }
            if((sl+su)>0)
            {
                
                tl=sl/(mean+1);
                tu=su/(255-mean);
                m2.lock();
                s2++;
                gq.push(gray);
                m2.unlock();
                m2.lock();
                tlq.push(tl);
                m2.unlock();
                m2.lock();
                tuq.push(tu);
                m2.unlock();
            }
        }
        if((iq.size()==0)&(x==false))break;
    }
    a=false;
    cout<<"Thread1"<<endl;
}

void part2()
{   
    Mat gray(fh,fw,CV_8UC1);
    Mat output(fh,fw,CV_8UC1);
    int hist[256],mean=127,tl,tu,f[256];
    float sl,su,pdf[256],cdf[256];
    while(true)
    {
        if(gq.size()>0)
        {
            m3.lock();
            gray=gq.front();
            tl=tlq.front();
            tu=tuq.front();
            gq.pop();tlq.pop();tuq.pop();
            m3.unlock();
            sl=0;su=0;
            for(int i=0;i<256;i++)hist[i]=0;
            for(int i=0;i<gray.rows;i++)
            {
                for(int j=0;j<gray.cols;j++)
                {
                    hist[gray.at<uchar>(i,j)]++;
                }
            }
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
            for(int i=0;i<256;i++)
            {
                if(i<=mean)sl+=hist[i];
                else su+=hist[i];
            }
            for(int i=0;i<256;i++)
            {
                if((i<=mean)&(sl!=0))pdf[i]=hist[i]/sl;
                else if((i>mean)&(su!=0))pdf[i]=hist[i]/su;
                else pdf[i]=0;
            }
            cdf[0]=pdf[0];
            for(int i=1;i<256;i++)
            {
                cdf[i]=cdf[i-1]+pdf[i];
                if(i==(mean+1))cdf[mean+1]=pdf[mean+1];
            }
            for(int i=0;i<256;i++)
            {
                if(i<=mean)f[i]=mean*(cdf[i]-0.5*pdf[i]);
                else f[i]=mean+1+(255-(mean+1))*(cdf[i]-0.5*pdf[i]);
            }
            for(int i=0;i<gray.rows;i++)
            {
                for(int j=0;j<gray.cols;j++)
                {
                    output.at<uchar>(i,j)=f[gray.at<uchar>(i,j)];
                }
            }
            
            m3.lock();
            oq.push(output);
            m3.unlock();
        }
        if((gq.size()==0)&(a==false))break;
    }
    b=false;
    cout<<"Thread2"<<endl;
}

int main()
{
    
    auto start=std::chrono::high_resolution_clock::now();
    VideoWriter video("output/bhepinput1n.avi", CV_FOURCC('M', 'J', 'P', 'G'), 15, Size(fw,fh ), 0);
    Mat input(fh,fw,CV_8UC3);
    Mat gray(fh,fw,CV_8UC1);
    x=true;a=b=true;int size;int s1=0;
    thread t1(part1);
    thread t2(part2);
    while(waitKey(40)!=27)
    {
        cap>>input;
        size = input.rows*input.cols;
        if(size>0){m1.lock();iq.push(input);m1.unlock();}
        else x=false;
        if(oq.size()>0)
        {
            //s1++;
            m1.lock();
            gray=oq.front();
            oq.pop();
            imshow("gray",gray);
            //video.write(gray);
            m1.unlock();
            
        }
        if((a==false)&(b==false)){cout<<"Main"<<endl;break;}
    }
    x=false;
    t1.join();
    t2.join();
    auto stop=std::chrono::high_resolution_clock::now();
    int time=std::chrono::duration_cast<chrono::milliseconds>(stop-start).count();
    cout<<endl<<"time:"<<time;
}
