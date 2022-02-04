from sklearn.feature_extraction import image
import pyrealsense2 as rs
import numpy as np
import cv2
import shutil

NUM_CAPTURES = 2
WIDTH = 640
HEIGHT = 480


if __name__ == "__main__":
    # Create a pipeline
    pipeline = rs.pipeline()

    #Create a config and configure the pipeline to stream
    #  different resolutions of color and depth streams
    config = rs.config()

    # note: using 640 x 480 depth resolution produces smooth depth boundaries
    #       using rs.format.bgr8 for color image format for OpenCV based image visualization
    config.enable_stream(rs.stream.color, WIDTH, HEIGHT, rs.format.bgr8, 30)
    config.enable_stream(rs.stream.depth, WIDTH, HEIGHT, rs.format.z16, 30)
    config.enable_stream(rs.stream.infrared, 1, WIDTH, HEIGHT, rs.format.y8, 30)
    config.enable_stream(rs.stream.infrared, 2, WIDTH, HEIGHT, rs.format.y8, 30)
 
    # Start streaming
    profile = pipeline.start(config)
    device = profile.get_device().query_sensors()[0]
    device.set_option(rs.option.emitter_enabled, 0)
    
    depth_stereo_sensor = rs.depth_stereo_sensor(device)
    print("baseline : ", depth_stereo_sensor.get_stereo_baseline())
    
    lf_profile = profile.get_stream(rs.stream.infrared, 1)
    rf_profile = profile.get_stream(rs.stream.infrared, 2)
    color_profile = profile.get_stream(rs.stream.color)
    
    T_lf_rgb = lf_profile.get_extrinsics_to(color_profile)
    T_rf_rgb = rf_profile.get_extrinsics_to(color_profile)

    print("T_lf_rgb translation : ", T_lf_rgb.translation)
    print("T_lf_rgb rotation    : ", T_lf_rgb.rotation)
    print("T_rf_rgb translation : ", T_rf_rgb.translation)
    print("T_rf_rgb rotation    : ", T_rf_rgb.rotation)

    # Create an align object
    # rs.align allows us to perform alignment of depth frames to others frames
    # The "align_to" is the stream type to which we plan to align depth frames.
    align_to = rs.stream.color
    align = rs.align(align_to)

    

    # Streaming loop
    try:
        while True:

            frames = pipeline.wait_for_frames()
            aligned_frames = align.process(frames)
            # intrinsics = frames.profile.as_video_stream_profile().intrinsics
            # print(intrinsics)
            aligned_color_frame = aligned_frames.get_color_frame()
            aligned_depth_frame = aligned_frames.get_depth_frame()
            aligned_lf_frame = aligned_frames.get_infrared_frame(1)
            aligned_rf_frame = aligned_frames.get_infrared_frame(2)

            # Validate that both frames are valid
            if (not aligned_color_frame or
                not aligned_frames or
                not aligned_lf_frame or
                not aligned_rf_frame):  
                continue
            
            color_image = np.asanyarray(aligned_color_frame.get_data())
            depth_image = np.asanyarray(aligned_depth_frame.get_data())
            lf_image = np.asanyarray(aligned_lf_frame.get_data())
            rf_image = np.asanyarray(aligned_rf_frame.get_data())
            
            lf_image_3d = np.dstack((lf_image, lf_image, lf_image))
            rf_image_3d = np.dstack((rf_image, rf_image, rf_image))
            
            # Render images
            depth_image = cv2.applyColorMap(
                cv2.convertScaleAbs(depth_image, alpha=0.09), cv2.COLORMAP_JET)
            
            # images = np.hstack((rf_image_3d, lf_image_3d, color_image))         
            ir_images = np.hstack((lf_image_3d, rf_image_3d))
            rgb_depth_images = np.hstack((color_image, depth_image))
            images = np.vstack((ir_images, rgb_depth_images))
            # images = np.hstack((depth_image_3d, lf_image_3d))
            cv2.namedWindow('Recorder Realsense', cv2.WINDOW_AUTOSIZE)
            cv2.imshow('Recorder Realsense', images)
            key = cv2.waitKey(1)

            # if 'esc' button pressed, escape loop and exit program
            if key == 27:
                cv2.destroyAllWindows()
                break
    finally:
        pipeline.stop()

