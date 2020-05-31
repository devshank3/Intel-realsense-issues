import pyrealsense2 as rs
import numpy as np
from enum import IntEnum
from datetime import datetime
import open3d as o3d

class Preset(IntEnum):
    Custom = 0
    Default = 1
    Hand = 2
    HighAccuracy = 3
    HighDensity = 4
    MediumDensity = 5

def get_intrinsic_matrix(frame):
    intrinsics = frame.profile.as_video_stream_profile().intrinsics
    out = o3d.camera.PinholeCameraIntrinsic(1280, 720, intrinsics.fx,intrinsics.fy, intrinsics.ppx,intrinsics.ppy)                                 
    return out

def nothing():
    pass
key_frame = 0

def save(vis, action, mods):
    global key_frame
    if action == 1:  # key down
        key_frame = key_frame + 1
        print("saving PCD no {}".format(str(key_frame)) ) 
        o3d.io.write_point_cloud("{}.pcd".format(str(key_frame)), pcd)


if __name__ == "__main__":
    print("Press s to save the PCD pointcloud")
    pipeline = rs.pipeline()
    
    config = rs.config()

    config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 1280, 720, rs.format.rgb8, 30)

    profile = pipeline.start(config)
    depth_sensor = profile.get_device().first_depth_sensor()

    depth_sensor.set_option(rs.option.visual_preset, Preset.Hand)

    depth_scale = depth_sensor.get_depth_scale()

    clipping_distance_in_meters = 1  # 3 meter
    clipping_distance = clipping_distance_in_meters / depth_scale


    align_to = rs.stream.color
    align = rs.align(align_to)

    vis  = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window(window_name='Real sense PCD viewer')
    opt = vis.get_render_option()
    opt.background_color = np.asarray([0, 0, 0])


    pcd = o3d.geometry.PointCloud()
    flip_transform = [[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]]

    frame_count = 0
    try:
        while True:
 
            frames = pipeline.wait_for_frames()
            aligned_frames = align.process(frames)
            aligned_depth_frame = aligned_frames.get_depth_frame()
            color_frame = aligned_frames.get_color_frame()
            intrinsic = o3d.camera.PinholeCameraIntrinsic(get_intrinsic_matrix(color_frame))

            if not aligned_depth_frame or not color_frame:
                continue

            depth_image = o3d.geometry.Image(np.array(aligned_depth_frame.get_data()))
            color_image = o3d.geometry.Image(np.asarray(color_frame.get_data()))

            rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
                color_image,
                depth_image,
                depth_scale=1.0 / depth_scale,
                depth_trunc=clipping_distance_in_meters,
                convert_rgb_to_intensity=False)

            temp = o3d.geometry.PointCloud.create_from_rgbd_image(
                rgbd_image, intrinsic)
            temp.transform(flip_transform)
            pcd.points = temp.points
            pcd.colors = temp.colors

            if frame_count == 0:
                vis.add_geometry(pcd)

            vis.register_key_action_callback(ord("S"),save)

            vis.update_geometry(pcd)
            
            
            vis.poll_events()
            vis.update_renderer()
           
            frame_count += 1

    finally:
        pipeline.stop()
    vis.destroy_window()   

