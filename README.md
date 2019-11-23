# DenseSLAM: Simultaneous Localization and Mapping for sparse and dense maps generation.

This is a dense SLAM system written in C++. It builds on [InfiniTAM](https://github.com/victorprad/InfiniTAM). Using the odometry from ORBSLAM2, this system generates the dense map form InfiniTAM. 

The source code is [hosted on GitHub](https://github.com/Hansry/DenseSLAM-Global-Consistency-h).

## Related Repositories

 * [My InfiniTAM fork](https://github.com/Hansry/InfiniTAM-Global-Consistency-h), which is used by this system for dense map reconstruction (via volumetric fusion, using voxel hashing for map storage).

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
 2. Install OpenCV 2.4.9 and CUDA 9.0.
 3. Install the prerequisites (Ubuntu example):
    ```bash
    sudo apt-get install libxmu-dev libxi-dev freeglut3 freeglut3-dev glew-utils libglew-dev libglew-dbg libpthread-stubs0-dev binutils-dev libgflags-dev libpng++-dev libeigen3-dev
    ```
 4. Build Pangolin to make sure it gets put into the CMake registry:
    ```bash
    cd src/Pangolin && mkdir build/ && cd $_ && cmake ../ && make -j$(nproc)
    ```
 5. Build the project in the standard CMake fashion:
    ```bash
    mkdir build && cd build && cmake .. && make -j$(nproc)
    ```

### Demo Sequence
 1. After building the project, try processing the demo sequence: 
    [here is a short sample from KITTI Odometry Sequence 06](https://drive.google.com/uc?export=download&confirm=Nnbd&id=1V-I4Tle7MNbmnf2qRe6aTpjxOld2M2i8).
      1. Extract that to a directory, and run DynSLAM on it (the mkdir circumvents a silly bug):
        ```bash
        mkdir -p csv/ && build/DynSLAM --use_dispnet --dataset_root=path/to/extracted/archive --dataset_type=kitti-odometry
        ```
