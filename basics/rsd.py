import time
import logging 

import numpy as np 
import pyrealsense2 as rs
import cv2

MODE_1 = [640,480,60]
MODE_2 = [1280,720,30]


PRESET = {"Custom":0,"Default":1,"Hand":2,"HighAccuracy":3,"HighDensity":4,"MediumDensity":5}

class rsd435i(object):
    """
    Data from rsd435i 
    mode = 1 sets w=640,h=480,fps=60 ||
    mode = 2 sets w=1280,h=720,fps=30 ||
    align = T/F ||
    preset = {"Custom":0,"Default":1,"Hand":2,"HighAccuracy":3,"HighDensity":4,"MediumDensity":5}
    """
    def __init__(self,mode = 1, align = False, preset = "Default"):

        if mode == 1:
            self.width = MODE_1[0]
            self.height = MODE_1[1]
            self.fps = MODE_1[2]

        elif mode == 2:
            self.width = MODE_2[0]
            self.height = MODE_2[1]
            self.fps = MODE_2[2]

        self.align = align
        self.preset = preset

        #config
        self.pipeline = rs.pipeline()
        config = rs.config()

        config.enable_stream(rs.stream.depth,self.width,self.height, rs.format.z16,self.fps)
        config.enable_stream(rs.stream.color,self.width, self.height, rs.format.bgr8,self.fps)

        profile = self.pipeline.start(config)

        depth_sensor = profile.get_device().first_depth_sensor()
        depth_sensor.set_option(rs.option.visual_preset, PRESET[self.preset])
        depth_scale = depth_sensor.get_depth_scale()

        print("Depth Scale is: ", depth_scale)
        print("Preset used is:", self.preset)
        print("Streaming  {}x{} at {}".format(str(self.width),str(self.height),str(self.fps)))

        if align:
            self.align_p = rs.align(rs.stream.color)
        
        for i in range(0,5):
            self.pipeline.wait_for_frames()
        
        self.color_image = None
        self.depth_image = None

        self.frame_count = 0
        self.start_time = time.time()
        self.frame_time = self.start_time

        self.running = True
    
    def stop_pipeline(self):
        self.pipeline.stop()
        self.pipeline = None

    def poll(self):
        last_time = self.frame_time

        try:
            frames = self.pipeline.wait_for_frames()
        except Exception as e:
            print(e)
            logging.error(e)
            return
        
        aligned_frames = self.align_p.process(frames) if self.align else None
        depth_frame = aligned_frames.get_depth_frame() if self.align else frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame() if self.align else frames.get_color_frame()

        self.depth_image = np.asanyarray(depth_frame.get_data(), dtype=np.uint16) 
        self.color_image = np.asanyarray(color_frame.get_data(), dtype=np.uint8)
    
    #def update(self):
    #   while self.running:
    #       self.poll()

    def run_threaded(self):
        return self.color_image,self.depth_image

    def run(self):
        self.poll()
        return self.run_threaded()

    def shutdown(self):
        self.running = False
        time.sleep(2)

        self.stop_pipeline()


if __name__ == "__main__":
    show_opencv_window = True
    try:
        camera = rsd435i(mode=2)
        frame_count = 0
        while True:
            color_img,depth_img = camera.run()
            frame_count += 1
    
            if show_opencv_window :
                cv2.namedWindow("Realsense",cv2.WINDOW_AUTOSIZE)
                depth_colormap =  cv2.applyColorMap(cv2.convertScaleAbs(depth_img, alpha=0.13), cv2.COLORMAP_TWILIGHT_SHIFTED)
                images = np.hstack((color_img,depth_colormap))
                cv2.imshow("Realsense",images)

            key = cv2.waitKey(1)
            if key & 0xFF == ord("q") or key ==27:
                cv2.destroyAllWindows()
                break               
        else:
            time.sleep(0.05)
    finally:
        camera.shutdown()



        
