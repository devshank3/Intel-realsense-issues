import numpy as np  
import cv2
import pyrealsense2 as rs
import argparse

#defaults
width = 1280
height  = 720
fps = 30
thresh_in_mtr = 1
#Argparse
parser = argparse.ArgumentParser()
parser.add_argument("path", help = "Enter the path for Background Image", default= "image.jpg")
args = parser.parse_args()
bg_img = args.path


def main():

    config  = rs.config()
    config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
    config.enable_stream(rs.stream.depth, width, height, rs.format.z16,fps)

    pipeline = rs.pipeline()
    profile = pipeline.start(config)
    align = rs.align(rs.stream.color)

    depth_scale = profile.get_device().first_depth_sensor().get_depth_scale()
    max_dist = thresh_in_mtr/depth_scale
    print(depth_scale)
    print(max_dist)

    bg_image = cv2.imread(bg_img)
    bg_image = cv2.resize(bg_image,(width,height))
    bg_image = cv2.medianBlur(bg_image,5)

    try:
        while True:
            frames = pipeline.wait_for_frames()
            aligned_frames = align.process(frames)

            color_frame = aligned_frames.get_color_frame()
            depth_frame = aligned_frames.get_depth_frame()

            if not depth_frame or not color_frame:
                continue
            
            #depth processing starts
            depth_image = np.asanyarray(depth_frame.get_data())
            depth_image_show = cv2.resize(depth_image,(640,360)) 
            #filtering
            depth_filtered = (depth_image < max_dist)* depth_image
            depth_filtered_show = cv2.resize(depth_filtered, (640,360))
            stat1 = np.hstack((depth_image_show,depth_filtered_show))
            depth_gray_filt = (depth_filtered * 255. / max_dist).reshape((height,width)).astype(np.uint8)
            depth_gray_filt_show = cv2.resize(depth_gray_filt,(640,360))
            ret, depth_mask = cv2.threshold(depth_gray_filt, 1,255,cv2.THRESH_BINARY)
            depth_mask = cv2.GaussianBlur(depth_mask, (9, 9), 0)
            depth_mask = cv2.medianBlur(depth_mask, 9)
            depth_mask_show = cv2.resize(depth_mask,(640,360))
            depth_mask = cv2.cvtColor(depth_mask, cv2.COLOR_GRAY2BGR)
            stat2 = np.hstack((depth_gray_filt_show,depth_mask_show))
            final_stat = np.vstack((stat1,stat2))
            #show status
            cv2.namedWindow("Stat1")
            cv2.imshow("Stat1",stat1)
            cv2.namedWindow("Stat2")
            cv2.imshow("Stat2",stat2)
            #merge
            color_image = np.asanyarray(color_frame.get_data())
            depth_mask_norm = (depth_mask / 255.0)
            compo_image = bg_image
            compo_image = (compo_image.astype(np.float32) * (1-depth_mask_norm)).astype(np.uint8)
            compo_image[0:height, 0:width] += (color_image*depth_mask_norm).astype(np.uint8)
            #show mwerge
            cv2.namedWindow("Merged")
            cv2.imshow("Merged",compo_image)

            if cv2.waitKey(1) & 0xff == 27:
                break

    finally:
        pipeline.stop()
    
if __name__ == "__main__":
    main()

