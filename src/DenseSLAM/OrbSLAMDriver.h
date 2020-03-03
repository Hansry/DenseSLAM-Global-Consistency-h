#ifndef ORBSLAMDRIVER_H
#define ORBSLAMDRIVER_H

#include <string>
#include "thread"
#include <opencv2/core/core.hpp>
#include "../ORB-SLAM2-API-M/include/System.h"
#include "../ORB-SLAM2-API-M/include/MapDrawer.h"
#include "../ORB-SLAM2-API-M/include/FrameDrawer.h"
#include "../ORB-SLAM2-API-M/include/Tracking.h"
#include "../ORB-SLAM2-API-M/include/LocalMapping.h"
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
		  ORB_SLAM2::System(strVocFile, strSettingsFile, sensor, bUseViewer),
		  strSettingsFile_(strSettingsFile){
		   last_egomotion_.setOnes();
		  };
		 
    virtual ~OrbSLAMDriver() {
    }
    
    ///@brief 返回世界坐标系到当前帧的变换Twc
    cv::Mat GetPose() const{
      cv::Mat OrbSLAMWorldToCurrFramePose = this->GetWorldToCurrFramePose();
      return OrbSLAMWorldToCurrFramePose;
    }
    
    ///@brief 得到OrbSLAM此时的跟踪状态,跟踪状态为2时为跟踪成功
    int GetOrbSlamTrackingState() const {
       return this->GetTrackingState();
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
    
    cv::Mat orbTrackMonocular(const cv::Mat &im, const double &timestamp){
         return this->TrackMonocular(im, timestamp);
    }
    
    void orbShutdown(){
         this->Shutdown();
    }
    
    void orbSaveTrajectoryKITTI(string saveName){
       this->SaveTrajectoryKITTI(saveName);
    }
    
    string getOrbParamFile() const{
       return strSettingsFile_;
    }
    
    list<ORB_SLAM2::KeyFrame*>* GetOrbSlamLocalBAKeyframe() {
      return this->GetLocalMapper()->getProcessKeyFrames();
    }
    
    std::condition_variable* GetTrackingCondVar() {
      return this->GetTracker()->getCond();
    }
    
    std::condition_variable* GetTrackingCondVar_n() {
       return this->GetTracker()->getCond_n();
    }
    
    bool* GetIsDenseMapCreate() {
      return this->GetTracker()->getIsDenseMapCreate();
    }
    
    bool* GetTrackingGL() {
      return this->GetTracker()->getGlobalLable();
    }
    
    bool* GetTrackingGL_n() {
      return this->GetTracker()->getGlobalLable_n();
    }
    
    void SetTrackingPose(cv::Mat Tcw){
      this->GetTracker()->mCurrentFrame.SetPose(Tcw);
    }
    
    map<double, cv::Mat> SaveTUMTrajectory(const string& filename){
      return this->SaveKeyFrameTrajectoryTUM(filename);
    }

private:
    Eigen::Matrix4f last_egomotion_;
    string strSettingsFile_;    
    
 };
}//driver
}// ORB_SLAM2
#endif