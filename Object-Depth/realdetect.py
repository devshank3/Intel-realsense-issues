#!/usr/bin/python

import jetson.inference
import jetson.utils

import argparse
import sys
import pyrealsense2 as rs
import numpy as np
import cv2
from enum import IntEnum


class Preset(IntEnum):
    Custom = 0
    Default = 1
    Hand = 2
    HighAccuracy = 3
    HighDensity = 4
    MediumDensity = 5


# parse the command line
parser = argparse.ArgumentParser(description="Locate objects in a live camera stream using an object detection DNN.",
                                 formatter_class=argparse.RawTextHelpFormatter, epilog=jetson.inference.detectNet.Usage())

parser.add_argument("--network", type=str, default="ssd-mobilenet-v2",
                    help="pre-trained model to load (see below for options)")
parser.add_argument("--overlay", type=str, default="box,labels,conf",
                    help="detection overlay flags (e.g. --overlay=box,labels,conf)\nvalid combinations are:  'box', 'labels', 'conf', 'none'")
parser.add_argument("--threshold", type=float, default=0.4,
                    help="minimum detection threshold to use")
parser.add_argument("--camera", type=str, default="0",
                    help="index of the MIPI CSI camera to use (e.g. CSI camera 0)\nor for VL42 cameras, the /dev/video device to use.\nby default, MIPI CSI camera 0 will be used.")
parser.add_argument("--width", type=int, default=1280,
                    help="desired width of camera stream (default is 1280 pixels)")
parser.add_argument("--height", type=int, default=720,
                    help="desired height of camera stream (default is 720 pixels)")

try:
    opt = parser.parse_known_args()[0]
except:
    print("")
    parser.print_help()
    sys.exit(0)


pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
profile = pipeline.start(config)
colorizer = rs.colorizer()
depth_sensor = profile.get_device().first_depth_sensor()
depth_sensor.set_option(rs.option.visual_preset, 4)
# Getting the depth sensor's depth scale (see rs-align example for explanation)
depth_scale = depth_sensor.get_depth_scale()


# We will not display the background of objects more than
#  clipping_distance_in_meters meters away
clipping_distance_in_meters = 3  # 3 meter
clipping_distance = clipping_distance_in_meters / depth_scale


align_to = rs.stream.color
align = rs.align(align_to)

# load the object detection network
net = jetson.inference.detectNet(opt.network, sys.argv, opt.threshold)

# create the camera and display
# camera = jetson.utils.gstCamera(opt.width, opt.height, opt.camera)
display = jetson.utils.glDisplay()

# process frames until user exits
try:
   while True:
    # capture the image
    # img, width, height = camera.CaptureRGBA()

    # Get frameset of color and depth
    frames = pipeline.wait_for_frames()

    # Align the depth frame to color frame
    aligned_frames = align.process(frames)

    spatial = rs.spatial_filter()
    # spatial.set_option(rs.option.holes_fill, 5)

    # spatial = rs.spatial_filter()
    # spatial.set_option(rs.option.filter_magnitude, 5)
    spatial.set_option(rs.option.filter_smooth_alpha, .25)
    spatial.set_option(rs.option.filter_smooth_delta, 45)
    spatial.set_option(rs.option.holes_fill, 2)
    # Get aligned frames
    aligned_depth_frame = aligned_frames.get_depth_frame()

    aligned_depth_frame = spatial.process(aligned_depth_frame)
    color_frame = aligned_frames.get_color_frame()
    
    depth_image = np.asanyarray(aligned_depth_frame.get_data())
    color_image = np.asanyarray(color_frame.get_data())
    colorized_depth = np.asanyarray(colorizer.colorize(aligned_depth_frame).get_data())
    rgba = cv2.cvtColor(color_image,cv2.COLOR_BGR2RGBA)
    width = color_image.shape[1]
    height = color_image.shape[0]
    
    img = jetson.utils.cudaFromNumpy(rgba)
    # detect objects in the image (with overlay)
    detections = net.Detect(img, width, height, opt.overlay)

    # print the detections
    print("detected {:d} objects in image".format(len(detections)))

    for detection in detections:
        print(detection)
        (ox,oy)=detection.Center
        ox = int(ox)
        oy = int(oy)
        print("#####################################################")
        print("Mean center of the object detected")
        print(ox,oy)
        depthy = depth_image[oy,ox]
        
        print("#####################################################")
        print(depthy)
        
        print("#####################################################") 
        print("depth of the object in meter scale ")
        print(depthy*depth_scale)
        print("#####################################################")

    # render the image
    display.RenderOnce(img, width, height)

    
    cv2.imshow('jetcolor depth', colorized_depth)

    # update the title bar
    display.SetTitle("{:s} | Network {:.0f} FPS".format(
        opt.network, net.GetNetworkFPS()))

    # print out performance info
    net.PrintProfilerTimes()
    
    '''
    grey_color = 10
    #depth image is 1 channel, color is 3 channels
    depth_image_3d = np.dstack((depth_image, depth_image, depth_image))
    bg_removed = np.where((depth_image_3d > clipping_distance) | \
    (depth_image_3d <= 0), grey_color, color_image)
    cv2.namedWindow('Recorder Realsense', cv2.WINDOW_AUTOSIZE)
    cv2.imshow('Recorder Realsense', bg_removed)
    '''
    key = cv2.waitKey(1)

    # if 'esc' button pressed, escape loop and exit program
    if key & 0xFF == ord('q') or key == 27:
        cv2.destroyAllWindows()
        break
            
finally:
    pipeline.stop()
