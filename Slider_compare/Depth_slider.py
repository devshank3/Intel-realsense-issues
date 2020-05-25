
import pyrealsense2 as rs
import numpy as np
import cv2

pipeline = rs.pipeline()


def nothing(x):
    pass


config = rs.config()
config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)


profile = pipeline.start(config)

depth_sensor = profile.get_device().first_depth_sensor()
depth_scale = depth_sensor.get_depth_scale()
print("Depth Scale is: " , depth_scale)


clipping_distance_in_cmeters = 1 
clipping_distance = clipping_distance_in_cmeters / depth_scale


align_to = rs.stream.color
align = rs.align(align_to)

cv2.namedWindow('Depth TrackBar')
cv2.createTrackbar('Dist in centimeter','Depth TrackBar',0,400,nothing)

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

        Black_bg = 0

        depth_image_3d = np.dstack((depth_image,depth_image,depth_image)) 
        bg_removed = np.where((depth_image_3d > clipping_distance) | (depth_image_3d <= 0), Black_bg, color_image)

        cv2.namedWindow('Depth TrackBar', cv2.WINDOW_AUTOSIZE)
        cv2.imshow('Depth TrackBar', bg_removed)
        
        clipping_distance_in_cmeters = cv2.getTrackbarPos('Dist in centimeter','Depth TrackBar')/100
        clipping_distance = clipping_distance_in_cmeters / depth_scale

        key = cv2.waitKey(1)
        if key & 0xFF == ord('q') or key == 27:
            cv2.destroyAllWindows()
            break
finally:
    pipeline.stop()