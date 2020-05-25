import pyrealsense2 as rs
import numpy as np
import cv2 

alpha_slider_max = 100

def nothing(x):
    pass

def a_b(val):
    alpha = val / alpha_slider_max
    beta = ( 1.0 - alpha )
    return alpha,beta


pipeline = rs.pipeline()

config = rs.config()
config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)

profile = pipeline.start(config)

depth_sensor = profile.get_device().first_depth_sensor()
depth_scale = depth_sensor.get_depth_scale()
print("Depth Scale is: " , depth_scale)

clipping_distance_in_meters = 4 #1 meter
clipping_distance = clipping_distance_in_meters / depth_scale

align_to = rs.stream.color
align = rs.align(align_to)

cv2.namedWindow('Aligned_color_depth')

cv2.createTrackbar('Alpha','Aligned_color_depth',0,alpha_slider_max,nothing)

try:
    while True:
       
        frames = pipeline.wait_for_frames()
        aligned_frames = align.process(frames)

        aligned_depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()

        if not aligned_depth_frame or not color_frame:
            continue

        depth_image = np.asanyarray(aligned_depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())

        grey_color = 0
        depth_image_3d = np.dstack((depth_image,depth_image,depth_image)) 
        bg_removed = np.where((depth_image_3d > clipping_distance) | (depth_image_3d <= 0), grey_color, color_image)

        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.08), cv2.COLORMAP_JET)
        
        alpha,beta = a_b(cv2.getTrackbarPos('Alpha','Aligned_color_depth'))

        dst = cv2.addWeighted(color_image, alpha, depth_colormap, beta, 0.0)

        cv2.namedWindow('Aligned_color_depth', cv2.WINDOW_AUTOSIZE)
        cv2.imshow('Aligned_color_depth', dst)

        key = cv2.waitKey(1)
        if key & 0xFF == ord('q') or key == 27:
            cv2.destroyAllWindows()
            break
finally:
    pipeline.stop()