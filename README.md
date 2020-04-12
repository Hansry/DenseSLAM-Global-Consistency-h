# DenseSLAM: Simultaneous Localization and Mapping for sparse and dense maps generation.

This is a dense SLAM system written in C++. It builds on volumetric fusion similar to [InfiniTAM](https://github.com/victorprad/InfiniTAM), which use voxel hashing for map storage). This system uses the odometry from ORBSLAM2 and generates the dense map simultaneously. 

The source code is [hosted on GitHub](https://github.com/Hansry/DenseSLAM-Global-Consistency-h).

## Building and Running DenseSLAM

### Building 

This project is built using CMake, and it depends on several submodules. 
As such, make sure you don't forget the `--recursive` flag when cloning the 
repository. If you did
forget it, just run `git submodule update --init --recursive`.

 1. Clone the repository if you haven't already:
    ```bash
    git clone --recursive https://github.com/Hansry/DenseSLAM-Global-Consistency-h
    ```
 2. Install OpenCV 2.4.9, CUDA 9.0, [Pangolin](https://github.com/stevenlovegrove/Pangolin.git).
 3. Install the prerequisites (Ubuntu example):
    ```bash
    sudo apt-get install libxmu-dev libxi-dev freeglut3 freeglut3-dev glew-utils libglew-dev libglew-dbg libpthread-stubs0-dev binutils-dev libgflags-dev libpng++-dev libeigen3-dev
    ```
 4. Build Pangolin to make sure it gets put into the CMake registry:
    ```bash
    cd src/Pangolin && mkdir build/ && cd $_ && cmake ../ && make -j$(nproc)
    ```
 5. Build g2o to make sure it gets put into CMake registry:
    ```bash
    cd src/ORB-SLAM2-API-M/Thirdparty/g2o_with_orbslam2 && mkdir build && cd build && cmake .. && make -j$(nproc) && sudo make install
    ```
 6. Build the project in the standard CMake fashion:
    ```bash
    mkdir build && cd build && cmake .. && make -j$(nproc)
    ```

### Demo Sequence
  After building the project, try processing the demo sequence: [here is a short sample from KITTI 2011_09_30_drive_0033_sync](https://pan.baidu.com/s/1Ufy_I_Uc2dTlzvnhqiKdeQ) and the password is `fwax`.

  1. Extract that to a directory, and run DenseSLAM on it (the mkdir circumvents a silly bug):
        ```bash
        mkdir -p csv && cd build && ./DenseSLAMGUI --dataset_root=path/to/extracted/archive --dataset_root=../data/KITTI/2011_09_30_drive_0033_sync  --sensor_type=1 --dataset_type=0
        ```
  2. Note that:
     
     sesor_type:  =0 means MONOCULAR, =1 means STEREO, =2 means RGBD;

     dataset_type: =0 meas KITTI dataset, =1 means TUM dataset, =2 means ICLNUIM dataset

