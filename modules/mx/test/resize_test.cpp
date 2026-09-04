#include "opencv2/mx/imgproc.hpp"
#include <opencv2/core.hpp>
#include <iostream>
#include <vector>
#include <cmath>

int main()
{
    struct Mode { int cn, w, h; };
    const Mode modes[] = {
        {1,512,768},{3,512,768},{1,1024,384},{4,1024,384},
        {1,512,384},{2,512,384},{3,512,384},{4,512,384},
        {1,256,192},{2,256,192},{3,256,192},{4,256,192},
        {1,4,3},{2,4,3},{3,4,3},{4,4,3},{1,342,384},{1,342,256},
        {2,342,256},{3,342,256},{4,342,256},{1,512,256},{1,146,110},
        {3,146,110},{4,146,110},{1,931,698},{2,931,698},{3,931,698},
        {4,931,698},{1,853,640},{3,853,640},{4,853,640},
        {1,1004,753},{2,1004,753},{3,1004,753},{4,1004,753},
        {1,2048,1536},{2,2048,1536},{4,2048,1536},
        {1,3072,2304},{3,3072,2304},{1,7168,5376}};
    const int rows=768, cols=1024;
    cv::Mat base(rows, cols, CV_8UC1); cv::RNG rng(0x123456789abcdefULL);
    for (int j=0;j<rows;j++) for(int i=0;i<cols;i++) {
        double v;
        if(j<rows/2) v=i<cols/2 ? (std::sin((i+1)*CV_PI/256.)*std::sin((j+1)*CV_PI/256.)*std::sin(7*CV_PI/8)+1)*128 : ((i/128+j/128)%2)*250+(j/128)%2;
        else if(i<cols/2) v=(i/128)*(85-j/256*40)*(j/128%2)+(7-i/128)*(85-j/256*40)*((j/128+1)%2);
        else v=(uchar)rng;
        base.at<uchar>(j,i)=(uchar)v;
    }
    for(size_t k=0;k<sizeof(modes)/sizeof(modes[0]);k++) {
        Mode m=modes[k]; cv::Mat src;
        if(m.cn==1) src=base; else { std::vector<cv::Mat> ch(m.cn,base); cv::merge(ch,src); }
        cv::Mat ref; cv::resize(src,ref,cv::Size(m.w,m.h),0,0,cv::INTER_LINEAR_EXACT);
        cv::mx::GpuMat in,out; in.upload(src); cv::mx::resize(in,out,ref.size(),0,0,cv::INTER_LINEAR);
        cv::Mat got; out.download(got); double e=cv::norm(ref,got,cv::NORM_INF);
        if(e>1) { std::cerr<<"FAIL case "<<k<<" CV_8UC"<<m.cn<<" -> "<<m.w<<"x"<<m.h<<" max_error="<<e<<"\n"; return 1; }
        std::cout<<"case "<<k<<" CV_8UC"<<m.cn<<" -> "<<m.w<<"x"<<m.h<<" max_error="<<e<<"\n";
    }
    std::cout<<"All OpenCV Resize_Bitexact Linear8U cases passed (max error <= 1)\n";
    return 0;
}
