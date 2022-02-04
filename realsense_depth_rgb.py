import pyrealsense2 as rs
import numpy as np
import cv2
import shutil

NUM_CAPTURES = 8
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
    config.enable_stream(rs.stream.depth, WIDTH, HEIGHT, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, WIDTH,
                             HEIGHT, rs.format.bgr8, 30)

    # Start streaming
    profile = pipeline.start(config)
    depth_sensor = profile.get_device().first_depth_sensor()

    # Getting the depth sensor's depth scale (see rs-align example for explanation)
    depth_scale = depth_sensor.get_depth_scale()

    # We will not display the background of objects more than
    #  clipping_distance_in_meters meters away
    clipping_distance_in_meters = 3.0  # 3 meter
    clipping_distance = clipping_distance_in_meters / depth_scale

    # Create an align object
    # rs.align allows us to perform alignment of depth frames to others frames
    # The "align_to" is the stream type to which we plan to align depth frames.
    align_to = rs.stream.color
    align = rs.align(align_to)

    decimation = rs.decimation_filter(magnitude=4)
    spatial = rs.spatial_filter(smooth_alpha=0.5,
                                smooth_delta=20,
                                magnitude=2,
                                hole_fill=0)
    temporal = rs.temporal_filter(smooth_alpha=0.4,
                                  smooth_delta=20,
                                  persistence_control=3)
    hole_filling = rs.hole_filling_filter(mode=1)

    depth_to_disparity = rs.disparity_transform(True)
    disparity_to_depth = rs.disparity_transform(False)

    # Streaming loop
    frame_count = 0
    try:
        while True:

            aligned_depth_frame = []
            for i in range(NUM_CAPTURES):
                frames = pipeline.wait_for_frames()
                decimated_frames = decimation.process(frames).as_frameset()
                aligned_frames = align.process(decimated_frames)
                aligned_depth_frame.append(aligned_frames.get_depth_frame())

            # Validate that both frames are valid
            if not aligned_depth_frame:  # or not color_frame:
                continue

            ################## APPLY FILTERS ##################
            # treat the depth image after applying the decimation filter.
            for i in range(NUM_CAPTURES):
                filtered_frame = depth_to_disparity.process(
                    aligned_depth_frame[i])
                filtered_frame = spatial.process(filtered_frame)
                filtered_frame = temporal.process(filtered_frame)
                filtered_frame = disparity_to_depth.process(filtered_frame)
                filtered_frame = hole_filling.process(filtered_frame)

            filtered_depth_frame = filtered_frame
            ###################################################

            depth_image = np.asanyarray(filtered_depth_frame.get_data())

            color_frame = aligned_frames.get_color_frame()
            color_image = np.asanyarray(color_frame.get_data())

            # Remove background - Set pixels further than clipping_distance to grey
            grey_color = 153
            #depth image is 1 channel, color is 3 channels
            depth_image_3d = np.dstack((depth_image, depth_image, depth_image))
            bg_removed = np.where((depth_image_3d > clipping_distance) |
                                  (depth_image_3d <= 0), grey_color, color_image)

            # Render images
            depth_colormap = cv2.applyColorMap(
                cv2.convertScaleAbs(depth_image, alpha=0.09), cv2.COLORMAP_JET)
            images = np.hstack((bg_removed, depth_colormap))
            cv2.namedWindow('Recorder Realsense', cv2.WINDOW_AUTOSIZE)
            cv2.imshow('Recorder Realsense', images)
            key = cv2.waitKey(1)

            # if 'esc' button pressed, escape loop and exit program
            if key == 27:
                cv2.destroyAllWindows()
                break
    finally:
        pipeline.stop()

