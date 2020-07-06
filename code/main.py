import cv2

import numpy as np
from tools import Tools
from image_processing_utilities import ImageProcessingUtilities
from video_stream_processing import VideoStreamProcessing
from line import Line
from vehicle_detector import VehicleDetector

tools = Tools()
imageProcessing = ImageProcessingUtilities()
videoStreamProcessing = VideoStreamProcessing()
vehicleDetector = VehicleDetector()
left_right_lines = [Line(), Line()]

# camera parameters
global mtx, dist, transform_mtx, inverse_transform_mtx, width, height

# SVM classifier parameters
global svc, X_scaler, orient, pix_per_cell, cell_per_block, spatial_size, hist_bins


def process_frame(frame, im_name="No name"):
    """
    :param frame: a colored image to be processed. width, height should correspond those from calibration.
    :param im_name: optional parameter for plotting with matplotlib.pyplot
    :return: an output image (np.array)
    This function is the image processing pipeline.
    Pre-requisites are the outputs from the camera calibration and image 3D-2D transformation.
    """

    global mtx, dist, transform_mtx, inverse_transform_mtx, width, height, left_right_lines

    # Step 1: distortion correction of the image
    undistorted = cv2.undistort(frame, mtx, dist, None, mtx)

    # Step 2: Color/gradient threshold
    combined_binary_image = imageProcessing.apply_color_gradient_thresholds(undistorted, im_name)

    # Step 3. Perspective transform
    warped_image = cv2.warpPerspective(combined_binary_image, transform_mtx, (width, height))
    # tools.plot_images(test_image, combined_binary_image, warped_image, [im_name,
    #                                                               'combined_binary_image', 'warped_image'])

    # optional histogram visualization step.
    # histogram = tools.get_histogram(warped_image)

    # Step 4. Detect lane lines
    # find a lane line using sliding window only on the first frame.
    # Reuse left and right Polynomials from the first frame for all the consequent frames
    # Find lanes using sliding window method only if:
    # - this is a first frame, hence no lanes have been detected
    # - at least one of the lanes haven't been detected in the previous frame.
    if left_right_lines[0].detected is False and left_right_lines[1].detected is False:
        # Apply Sliding Window and Fit a Polynomial
        # No need to pass left_right_lines, because the line haven't been detected in the previous frame anyway
        out_img, left_right_lines = imageProcessing.fit_sliding_polynomial(warped_image, im_name)
    else:
        # Processing of the 2nd and all consequent frames in the video stream
        out_img, left_right_lines = videoStreamProcessing.search_around_poly(warped_image, left_right_lines)

    # Step 5. Determine the lane curvature in pixels for both lane lines
    ploty, left_right_lines[0].radius_of_curvature, left_right_lines[1].radius_of_curvature = \
        imageProcessing.measure_curvature_pixels(left_right_lines[0].best_fit, left_right_lines[1].best_fit)

    # Auxiliary step: add curvature and offset information to the output image
    text_lines = tools.get_text_overlay(left_right_lines[0].radius_of_curvature,
                                        left_right_lines[1].radius_of_curvature,
                                        left_right_lines[0].line_base_pos)
    output_image = imageProcessing.draw_plane_over_image(ploty, frame, undistorted, warped_image,
                                                         left_right_lines[0].bestx, left_right_lines[1].bestx,
                                                         inverse_transform_mtx, text_lines, im_name)

    # ystart, ystop, scale
    y_ranges = [[400, 464, 1.0],  [416, 480, 1.0], [400, 496, 1.5], [400, 656, 1.5], [400, 528, 2],
                [400, 580, 2], [400, 560, 2.5]]
    output_image = vehicleDetector.detect_vehicles(frame, output_image, y_ranges)
    return output_image


# Execution flow of the advanced lane finding pipeline
# Definition of the variables
def main():
    # Load camera parameters
    camera_params_count = 6
    global mtx, dist, transform_mtx, inverse_transform_mtx, width, height
    mtx, dist, transform_mtx, inverse_transform_mtx, width, height = tools.load_params(camera_params_count,
                                                                                       'camera_parameters.pkl')

    # Step 2: process test images
    # imageProcessing.process_test_images(process_frame)

    # # Step 3: video stream processing pipeline
    videoStreamProcessing.process_video(process_frame)


# Program start
if __name__ == "__main__":
    main()
