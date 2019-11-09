#include "OrbSLAMDriver.h"

namespace ORB_SLAM2{
namespace drivers{
  
Eigen::Matrix4f MatToEigen(const cv::Mat &mat_matrix){
  Eigen::Matrix4f matrix;
  matrix<<mat_matrix.at<float>(0,0),mat_matrix.at<float>(0,1),mat_matrix.at<float>(0,2),mat_matrix.at<float>(0,3),
          mat_matrix.at<float>(1,0),mat_matrix.at<float>(1,1),mat_matrix.at<float>(1,2),mat_matrix.at<float>(1,3),
          mat_matrix.at<float>(2,0),mat_matrix.at<float>(2,1),mat_matrix.at<float>(2,2),mat_matrix.at<float>(2,3),
          mat_matrix.at<float>(3,0),mat_matrix.at<float>(3,1),mat_matrix.at<float>(3,2),mat_matrix.at<float>(3,3);
  return matrix;
}
  
cv::Mat EigenToMat(const Eigen::Matrix4f &Eig_matrix){
  cv::Mat Matrix = cv::Mat::eye(4,4,CV_32F);
  for(int row=0;row<Matrix.rows;row++){
    for(int col=0;col<Matrix.cols;col++){
       Matrix.at<float>(row,col) = Eig_matrix(row,col);
    }
  }
  return Matrix;
}
  
} //
}