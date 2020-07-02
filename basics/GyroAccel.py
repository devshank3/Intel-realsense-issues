import pyrealsense2 as rs
import numpy as np
import cv2
import time


pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 360, rs.format.z16, 60)
config.enable_stream(rs.stream.color, 640, 360, rs.format.bgr8, 60)
config.enable_stream(rs.stream.accel, rs.format.motion_xyz32f, 250)
config.enable_stream(rs.stream.gyro, rs.format.motion_xyz32f, 200)

pipeline.start(config)

try:
    while True:
        time1 = time.time()

        frames = pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()
        accel_frame = frames.first_or_default(rs.stream.accel).as_motion_frame()
        gyros_frame = frames.first_or_default(rs.stream.gyro).as_motion_frame()

        if not depth_frame or not color_frame or not accel_frame or not gyros_frame:
            continue

        if accel_frame:
            accel_sample = accel_frame.get_motion_data()
            print("Accel: ", accel_sample)
        if gyros_frame:
            gyros_sample = gyros_frame.get_motion_data()
            print("Gyros: ", gyros_sample)

        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())
        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.08), cv2.COLORMAP_JET)

        cv2.imshow("Color", color_image)
        cv2.imshow("Depth Colormap", depth_colormap)
        key = cv2.waitKey(1)
        if key == 27 or key & 0xFF == ord('q'):
            break
        print('Done 1 Frame! Time: {0:0.3f}ms & FPS: {1:0.2f}'.format(
            (time.time()-time1)*1000,
            1/(time.time()-time1)
        ))

finally:
    pipeline.stop()
    cv2.destroyAllWindows()
