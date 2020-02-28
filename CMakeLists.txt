cmake_minimum_required(VERSION 3.1)
project(DenseSLAM-Global-Consistency-huang)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake")

macro(CANONIFY_BOOL var)
  if (${var})
    set(${var} TRUE)
  else()
    set(${var} FALSE)
  endif()
endmacro()

# We use Eigen for generic matrix support and linear algebra routines.
# set(EIGEN3_ROOT ${PROJECT_SOURCE_DIR}/src/eigen)
# set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)

find_package(Eigen3 3.2 REQUIRED)
find_package(GLUT REQUIRED)
find_package(OpenGL REQUIRED)

find_package(CUDA  REQUIRED)
CANONIFY_BOOL(CUDA_FOUND)
message(STATUS "CUDA found: " ${CUDA_FOUND})
OPTION(WITH_CUDA "Build with CUDA support?" ${CUDA_FOUND})

IF(MSVC_IDE)
  add_definitions(-D_CRT_SECURE_NO_WARNINGS)
  add_definitions(-DUSING_CMAKE=1)
ELSE(MSVC_IDE)
  set(CFLAGS_WARN "-Wall -Wextra -Wno-unused-parameter -Wno-strict-aliasing")
  set(CMAKE_CXX_FLAGS "-fPIC -O3 -march=native -g ${CFLAGS_WARN} ${CMAKE_CXX_FLAGS}")
ENDIF(MSVC_IDE)

if(APPLE)
  set(CUDA_HOST_COMPILER /usr/bin/clang)
endif(APPLE)

IF("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -stdlib=libstdc++ -Wno-deprecated-declarations -Wno-unused-function")
  set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -stdlib=libstdc++")
ENDIF()

# Set some additional common compilation flags, like C++11 feature support.
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

# We use Pangolin for the main GUI. Pangolin is a lightweight GUI library focused on visual
# applications with minimal support for multiple panels, widgets, and live plotting.
# Note that this part *must* be after the CUDA setup, lest you get errors about hidden symbols
# referenced by some DSO.
#add_subdirectory(src/Pangolin)   
set(BUILD_PANGOLIN_EIGEN ON CACHE BOOL "")
find_package(Pangolin REQUIRED)

# Use the Google flag library for sane handling of command-line flags.
add_subdirectory(src/gflags)
# Library for stereo visual odometry and sparse scene flow estimation.
add_subdirectory(src/libviso2)
# find_package(Sophus REQUIRED)

IF(WITH_CUDA)
  include_directories(${CUDA_INCLUDE_DIRS})
ELSE()
  add_definitions(-DCOMPILE_WITHOUT_CUDA)
ENDIF()

# We rely on OpenCV for image IO and manipulation. Moreover, our (future) Caffe
# dependencies will require it anyway...
find_package(OpenCV REQUIRED)

include_directories(${EIGEN3_INCLUDE_DIR})
include_directories(${Viso2_INCLUDE_DIRS})
include_directories(${Pangolin_INCLUDE_DIRS})
include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(${Sophus_INCLUDE_DIRS})

message(${PROJECT_SOURCE_DIR})


add_subdirectory(src/ORB-SLAM2-API-M/Thirdparty/DBoW2)
add_subdirectory(src/ORB-SLAM2-API-M)
add_subdirectory(src/InfiniTAM-Global-Consistency-h/InfiniTAM)
add_subdirectory(src/DenseSLAM/InstRecLib)


#add_subdirectory(src/ORB-SLAM2/Thirdparty/g2o)
# Typically you don't have to add headers as sources, since they get included and compiled
# anyway, but CLion's static analysis works better on them if they're listed as sources.
set(DENSESLAM_SOURCES
    # A tiny library for working with portable float map (PFM) files.
    src/pfmLib/ImageIOpfm.cpp
    src/DenseSLAM/InfiniTamDriver.h
    src/DenseSLAM/InfiniTamDriver.cpp
    src/DenseSLAM/DepthProvider.h
    src/DenseSLAM/DSHandler3D.cpp
    src/DenseSLAM/DenseSlam.cpp
    src/DenseSLAM/DenseSLAMGUI.cpp
    src/DenseSLAM/DenseSLAMGUI.h
    src/DenseSLAM/Input.cpp
    src/DenseSLAM/OrbSLAMDriver.h
    src/DenseSLAM/OrbSLAMDriver.cpp
    src/DenseSLAM/PrecomputedDepthProvider.cpp
    src/DenseSLAM/PrecomputedDepthProvider.h
    src/DenseSLAM/Utils.cpp 
    )

set(DENSESLAM_GUI_SOURCES
    ${DENSESLAM_SOURCES}
    src/DenseSLAM/SystemEntry.cpp)

add_library(DenseSLAM ${DENSESLAM_SOURCES} ${EXTRA_EXECUTABLE_FLAGS})
#target_link_libraries(DenseSLAMGUI Engine)
# target_link_libraries(DenseSLAMGUI Utils)
target_link_libraries(DenseSLAM ITMLib)
target_link_libraries(DenseSLAM ORUtils)
# target_link_libraries(DenseSLAM FernRelocLib)
target_link_libraries(DenseSLAM MiniSlamGraphLib)
target_link_libraries(DenseSLAM InstRecLib)
target_link_libraries(DenseSLAM ${OpenCV_LIBS})
target_link_libraries(DenseSLAM gflags)
target_link_libraries(DenseSLAM ${Viso2_LIBS})
target_link_libraries(DenseSLAM ORB_SLAM2)
message(${PROJECT_SOURCE_DIR})

add_executable(DenseSLAMGUI ${DENSESLAM_GUI_SOURCES} ${EXTRA_EXECUTABLE_FLAGS})
target_link_libraries(DenseSLAMGUI ${Pangolin_LIBRARIES})
#target_link_libraries(DenseSLAMGUI Engine)
# target_link_libraries(DenseSLAMGUI Utils)
target_link_libraries(DenseSLAMGUI ITMLib)
# target_link_libraries(DenseSLAMGUI FernRelocLib)
target_link_libraries(DenseSLAMGUI MiniSlamGraphLib)
target_link_libraries(DenseSLAMGUI ORUtils)
target_link_libraries(DenseSLAMGUI InstRecLib)
target_link_libraries(DenseSLAMGUI ${OpenCV_LIBS})
target_link_libraries(DenseSLAMGUI gflags)
target_link_libraries(DenseSLAMGUI ${Viso2_LIBS})
target_link_libraries(DenseSLAMGUI ORB_SLAM2)


