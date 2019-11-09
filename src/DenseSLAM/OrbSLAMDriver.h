#ifndef ORBSLAMDRIVER_H
#define ORBSLAMDRIVER_H

#include <string>
#include "thread"
#include <opencv2/core/core.hpp>
#include "../ORB-SLAM2-API-M/include/System.h"
#include "../ORB-SLAM2-API-M/include/MapDrawer.h"
#include "../ORB-SLAM2-API-M/include/FrameDrawer.h"
#include <Eigen/Core>

namespace ORB_SLAM2{
namespace drivers{

///@brief Mat->Eigen
Eigen::Matrix4f MatToEigen(const cv::Mat& mat_matrix);

///@brief Eigen->Mat
cv::Mat EigenToMat(const Eigen::Matrix4f &eigen_matrix);

// \brief DynSLAM和ORBSLAM2的接口
class OrbSLAMDriver: public ORB_SLAM2::System{
public:
    
    ///OrbSLAMDriver的构造函数
    OrbSLAMDriver(const string &strVocFile, const string &strSettingsFile, 
		  const eSensor sensor, const bool bUseViewer = true):
		  ORB_SLAM2::System(strVocFile, strSettingsFile, sensor, bUseViewer){
		   last_egomotion_.setOnes();
		  };
		 
    virtual ~OrbSLAMDriver() {
    }
    
    ///@brief 返回世界坐标系到当前帧的变换Tcw
    Eigen::Matrix4f GetPose() const{
      cv::Mat OrbSLAMWorldToCurrFramePose = this->GetWorldToCurrFramePose();
      return MatToEigen(OrbSLAMWorldToCurrFramePose);
    }
    
    ///@brief 获取俩帧之前的相对位置
    void Track(){
      cv::Mat CurrentFrame = this->GetWorldToCurrFramePose();
      cv::Mat LastFrameInv = this->GetWorldTolastFramePose();
      (this->last_egomotion_) = MatToEigen(CurrentFrame*LastFrameInv);
    }
    
    Eigen::Matrix4f GetlastEgomotion() const{
      return last_egomotion_;
    }
    
    ORB_SLAM2::MapDrawer* GetOrbSlamMapDrawer() const{
        return this->GetMapDrawer();
    }
    
    ORB_SLAM2::FrameDrawer* GetOrbSlamFrameDrawer() const{
        return this->GetFrameDrawer();
    }
    
    ORB_SLAM2::Tracking* GetOrbSlamTracker() const{
        return this->GetTracker();
    }
    
    cv::Mat orbTrackStereo(const cv::Mat& imleft, const cv::Mat& imright, const double& timestamp) {
         return this->TrackStereo(imleft, imright, timestamp);
    }
    
    cv::Mat orbTrackRGBDSLAM(const cv::Mat &im, const cv::Mat &depthmap, const double &timestamp){
         return this->TrackRGBD(im,depthmap,timestamp);
    }
    
    void orbShutdown(){
         this->Shutdown();
    }
    
    void orbSaveTrajectoryKITTI(string saveName){
       this->SaveTrajectoryKITTI(saveName);
    }
    
private:
    Eigen::Matrix4f last_egomotion_;    
};
}//driver
}// ORB_SLAM2
#endif