# Camera 3D Object Tracking

In this project it is implemented a time-to-collision system. To do so, it is implemented a YOLO object detection system which runs on a series of images to identify vehicles. YOLO stands for “you only look once,” referring to the way the object detection is implemented, where the network is restricted to determine all the objects along with their confidences and bounding boxes, in one forward pass of the network for maximum speed. It is based on the paper and associated library Darknet.

![yolo](/images/yoloResults.PNG)

Then, it is developed a match of 3D objects over time by using keypoint correspondences, a computation of the TTC based on Lidar and camera measurements. Lastly it is stated a list of detector/descriptor combinations to identify the most suitable pair.

![ttc](/images/ttcResults.PNG)


## Dependencies for Running Locally
* cmake >= 2.8
  * All OSes: [click here for installation instructions](https://cmake.org/install/)
* make >= 4.1 (Linux, Mac), 3.81 (Windows)
  * Linux: make is installed by default on most Linux distros
  * Mac: [install Xcode command line tools to get make](https://developer.apple.com/xcode/features/)
  * Windows: [Click here for installation instructions](http://gnuwin32.sourceforge.net/packages/make.htm)
* Git LFS
  * Weight files are handled using [LFS](https://git-lfs.github.com/)
* OpenCV >= 4.1
  * This must be compiled from source using the `-D OPENCV_ENABLE_NONFREE=ON` cmake flag for testing the SIFT and SURF detectors.
  * The OpenCV 4.1.0 source code can be found [here](https://github.com/opencv/opencv/tree/4.1.0)
* gcc/g++ >= 5.4
  * Linux: gcc / g++ is installed by default on most Linux distros
  * Mac: same deal as make - [install Xcode command line tools](https://developer.apple.com/xcode/features/)
  * Windows: recommend using [MinGW](http://www.mingw.org/)

## Basic Build Instructions

1. Clone this repo.
2. Make a build directory in the top level project directory: `mkdir build && cd build`
3. Compile: `cmake .. && make`
4. Run it: `./3D_object_tracking`.
