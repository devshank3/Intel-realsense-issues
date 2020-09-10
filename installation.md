Getting Started with Python - pyrealsense2

## Installation 
- Installation without building from source

1. Unplug any connected Intel RealSense camera
2. Install the core packages required to build librealsense binaries and the affected kernel modules:
   `sudo apt-get install git libssl-dev libusb-1.0-0-dev pkg-config libgtk-3-dev`

3. Ubuntu 18:
   `sudo apt-get install libglfw3-dev libgl1-mesa-dev libglu1-mesa-dev`

4. Run the udev rules update patch from repo root
   `sudo ./scripts/setup_udev_rules.sh`

5. Python Package
   `pip install pyrealsense2`

PyPi ---> [link](https://pypi.python.org/pypi/pyrealsense2)


5. Other Python dependencies
- Numpy
- OpenCV-python
- Pyglet
- Matplotlib
- Open3D


## Resources:
---

1. https://github.com/IntelRealSense/librealsense/issues/5049
2. https://github.com/IntelRealSense/librealsense/blob/master/doc/installation.md
