import numpy as np
import cv2
import matplotlib.pyplot as plt
from tools import Tools
from line import Line


class ImageProcessingUtilities:

    def __init__(self):
        self.tools = Tools()

    @staticmethod
    def get_trapezoidal_transform_matrix(test_image):
        height, width, channels = test_image.shape  # no color channels

        src = np.float32(
            [[586, 461],  # top left
             [705, 461],  # top right
             [1041, 676],  # bottom right
             [276, 676]])  # bottom left

        dst = np.float32(
            [[300, 200],  # top left
             [900, 200],  # top right
             [900, 710],  # bottom right
             [300, 710]])  # bottom left

        transform_mtx = cv2.getPerspectiveTransform(src, dst)
        inverse_transform_mtx = cv2.getPerspectiveTransform(dst, src)

        # plot both images for visual validity
        # warped = cv2.warpPerspective(test_image, transform_mtx, (width, height))
        # self.tools.plot_dots(test_image, warped, src, dst)

        return transform_mtx, inverse_transform_mtx, width, height

    def get_transformation_values(self, images, im_names):
        for pos, test_image in enumerate(images):
            if im_names[pos].split('/')[-1] == 'straight_lines2.jpg' or \
                    im_names[pos].split('/')[-1] == 'straight_lines1.jpg':
                transform_mtx, inverse_transform_mtx, width, height = self.get_trapezoidal_transform_matrix(test_image)
                # break
        if transform_mtx is None or width is None or height is None:
            print("No transformation values for {} have been calculated. Exiting with ERROR"
                  .format(im_names[pos].split('/')[-1]))
            exit(1)
        return transform_mtx, inverse_transform_mtx, width, height

    @staticmethod
    def threshold_gradient(gray):
        """
        :param gray: a grayscale image
        :return: binary image with white pixels as result of the Sobel operator
        Applying Sobel operator to the grayscale image. Calculates the first, second, third, or mixed image derivatives.
        """
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0)  # Take the derivative in x
        abs_sobelx = np.absolute(sobelx)  # Absolute x derivative to accentuate lines away from horizontal
        scaled_sobel = np.uint8(255 * abs_sobelx / np.max(abs_sobelx))

        # Threshold x gradient
        thresh_min = 20
        thresh_max = 100
        sxbinary = np.zeros_like(scaled_sobel)
        sxbinary[(scaled_sobel >= thresh_min) & (scaled_sobel <= thresh_max)] = 1

        return sxbinary

    @staticmethod
    def threshold_color(hls):
        """
        :param hls: an image in [HLS](https://de.wikipedia.org/wiki/HSV-Farbraum) Color space
        :return: binary image with white pixels as result of threshold applied to the Saturation channel
        Extract saturation channel from the HLS-colorspace image and apply filter to it.
        """
        s_channel = hls[:, :, 2]
        # Threshold color channel
        s_thresh_min = 170
        s_thresh_max = 255
        s_binary = np.zeros_like(s_channel)
        s_binary[(s_channel >= s_thresh_min) & (s_channel <= s_thresh_max)] = 1

        return s_binary

    def apply_color_gradient_thresholds(self, image, im_name="No name"):
        # Convert to HLS color space and separate the S channel
        hls = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
        # Grayscale image
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

        sxbinary = self.threshold_gradient(gray)
        s_binary = self.threshold_color(hls)

        # Stack each channel to view their individual contributions in green and blue respectively
        # This returns a stack of the two binary images, whose components you can see as different colors
        color_binary = np.dstack((np.zeros_like(sxbinary), sxbinary, s_binary)) * 255

        # Combine the two binary thresholds
        combined_binary = np.zeros_like(sxbinary)
        combined_binary[(s_binary == 1) | (sxbinary == 1)] = 1

        # Plotting thresholded images
        # self.tools.plot_images(image, color_binary, combined_binary, [im_name,
        #                                                               'Stacked thresholds',
        #                                                               'Combined S channel and gradient thresholds'])
        return combined_binary

    @staticmethod
    def find_lane_pixels(binary_warped):
        # Take a histogram of the bottom half of the image
        histogram = np.sum(binary_warped[binary_warped.shape[0] // 2:, :], axis=0)
        # Create an output image to draw on and visualize the result
        out_img = np.dstack((binary_warped, binary_warped, binary_warped))
        # Find the peak of the left and right halves of the histogram
        # These will be the starting point for the left and right lines
        midpoint = np.int(histogram.shape[0] // 2)
        leftx_base = np.argmax(histogram[:midpoint])
        rightx_base = np.argmax(histogram[midpoint:]) + midpoint

        # HYPERPARAMETERS
        # Choose the number of sliding windows
        nwindows = 9
        # Set the width of the windows +/- margin
        margin = 100
        # Set minimum number of pixels found to recenter window
        minpix = 50

        # Set height of windows - based on nwindows above and image shape
        window_height = np.int(binary_warped.shape[0] // nwindows)
        # Identify the x and y positions of all nonzero pixels in the image
        nonzero = binary_warped.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Current positions to be updated later for each window in nwindows
        leftx_current = leftx_base
        rightx_current = rightx_base

        # Create empty lists to receive left and right lane pixel indices
        left_lane_inds = []
        right_lane_inds = []

        # Step through the windows one by one
        for window in range(nwindows):
            # Identify window boundaries in x and y (and right and left)
            win_y_low = binary_warped.shape[0] - (window + 1) * window_height
            win_y_high = binary_warped.shape[0] - window * window_height
            win_xleft_low = leftx_current - margin
            win_xleft_high = leftx_current + margin
            win_xright_low = rightx_current - margin
            win_xright_high = rightx_current + margin

            # Draw the windows on the visualization image
            cv2.rectangle(out_img, (win_xleft_low, win_y_low),
                          (win_xleft_high, win_y_high), (0, 255, 0), 2)
            cv2.rectangle(out_img, (win_xright_low, win_y_low),
                          (win_xright_high, win_y_high), (0, 255, 0), 2)

            # Identify the nonzero pixels in x and y within the window #
            good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                              (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
            good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                               (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]

            # Append these indices to the lists
            left_lane_inds.append(good_left_inds)
            right_lane_inds.append(good_right_inds)

            # If you found > minpix pixels, recenter next window on their mean position
            if len(good_left_inds) > minpix:
                leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
            if len(good_right_inds) > minpix:
                rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

        # Concatenate the arrays of indices (previously was a list of lists of pixels)
        try:
            left_lane_inds = np.concatenate(left_lane_inds)
            right_lane_inds = np.concatenate(right_lane_inds)
        except ValueError:
            # Avoids an error if the above is not implemented fully
            pass

        # Extract left and right line pixel positions
        leftx = nonzerox[left_lane_inds]
        lefty = nonzeroy[left_lane_inds]
        rightx = nonzerox[right_lane_inds]
        righty = nonzeroy[right_lane_inds]

        return leftx, lefty, rightx, righty, out_img

    def calculate_distance_from_lane_center(self, left_pixel, right_pixel):
        x_width = 1280      # in pixel
        lane_width_m = 3.7    # in meter
        frame_middle = x_width / 2
        lane_width_pixel = right_pixel - left_pixel
        lane_middle_pixel = (right_pixel + left_pixel)/2
        offset_from_middle_pixel = lane_middle_pixel - frame_middle
        offset_from_middle_m = (offset_from_middle_pixel * lane_width_m)/lane_width_pixel

        return offset_from_middle_m

    def fit_sliding_polynomial(self, binary_warped, im_name="No name"):
        left_right_lines = [Line(), Line()]

        # Find our lane pixels first
        leftx, lefty, rightx, righty, out_img = self.find_lane_pixels(binary_warped)

        # Fit a second order polynomial to each using `np.polyfit`
        left_fit = np.polyfit(lefty, leftx, 2)
        right_fit = np.polyfit(righty, rightx, 2)

        # Generate x and y values for plotting
        ploty = np.linspace(0, binary_warped.shape[0] - 1, binary_warped.shape[0])
        try:
            left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
            right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]

            left_right_lines[0].detected = True
            left_right_lines[1].detected = True
            left_right_lines[0].add_polyfit(left_fit)
            left_right_lines[1].add_polyfit(right_fit)
            left_right_lines[0].add_xfitted(left_fitx)
            left_right_lines[1].add_xfitted(right_fitx)

            offset_from_middle_m = self.calculate_distance_from_lane_center(left_fitx[len(left_fitx)-1], right_fitx[len(right_fitx)-1])
            left_right_lines[0].line_base_pos = offset_from_middle_m
            left_right_lines[1].line_base_pos = offset_from_middle_m

            left_right_lines[0].allx = leftx
            left_right_lines[0].ally = lefty

            left_right_lines[1].allx = rightx
            left_right_lines[1].ally = righty

        except TypeError:
            # Avoids an error if `left` and `right_fit` are still none or incorrect
            print('fit_sliding_polynomial failed to fit a line!\nGive it a try in the next frame')
            left_fitx = 1 * ploty ** 2 + 1 * ploty
            right_fitx = 1 * ploty ** 2 + 1 * ploty
            left_right_lines[0].reset_line()
            left_right_lines[0].reset_line()

        ## Visualization ##
        # Colors in the left and right lane regions
        out_img[lefty, leftx] = [255, 0, 0]
        out_img[righty, rightx] = [0, 0, 255]

        # Plots the left and right polynomials on the lane lines
        # plt.plot(left_fitx, ploty, color='yellow')
        # plt.plot(right_fitx, ploty, color='yellow')
        # self.tools.plot_image(out_img, im_name)

        return out_img, left_right_lines

    @staticmethod
    def measure_curvature_pixels(left_fit, right_fit):
        """
        :param left_fit: 3 polynomial coefficients of the left lane border line
        :param right_fit: 3 polynomial coefficients of the right lane border line
        :return: ploty: y-range linespace, left_curverad: radius of the left line in RAD,
        right_curverad: radius of the left line in RAD

        Calculates the curvature of polynomial functions in pixels.
        """
        ploty = np.linspace(0, 719, num=720)  # to cover same y-range as image

        # Define y-value where we want radius of curvature
        # We'll choose the maximum y-value, corresponding to the bottom of the image
        y_eval = np.max(ploty)

        # Calculation of R_curve (radius of curvature)
        left_curverad = ((1 + (2 * left_fit[0] * y_eval + left_fit[1]) ** 2) ** 1.5) / np.absolute(2 * left_fit[0])
        right_curverad = ((1 + (2 * right_fit[0] * y_eval + right_fit[1]) ** 2) ** 1.5) / np.absolute(2 * right_fit[0])

        return ploty, left_curverad, right_curverad

    @staticmethod
    def add_text_to_image(result, text_lines):

        # font
        font = cv2.FONT_HERSHEY_SIMPLEX
        # coordinates of the bottom-left corner of the text string
        origin_x = 50
        origin_y = 50
        # fontScale
        fontScale = 1
        # While color in BGR
        color = (255, 255, 255)
        # Line thickness of 2 px
        thickness = 2
        for line in text_lines:
            cv2.putText(result, line, (origin_x, origin_y), font, fontScale, color, thickness, cv2.LINE_AA)
            origin_y = origin_y + 50
        return result

    def draw_plane_over_image(self, ploty, original_image, undist_image, warped_image, left_fitx, right_fitx,
                              inverse_transform_mtx, text_lines, im_name="No name"):

        # Create an image to draw the lines on
        warp_zero = np.zeros_like(warped_image).astype(np.uint8)
        color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

        # Recast the x and y points into usable format for cv2.fillPoly()
        pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
        pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
        pts = np.hstack((pts_left, pts_right))

        # Draw the lane onto the warped blank image
        cv2.fillPoly(color_warp, np.int_([pts]), (0, 255, 0))

        # Warp the blank back to original image space using inverse perspective matrix (Minv)
        newwarp = cv2.warpPerspective(color_warp, inverse_transform_mtx, (original_image.shape[1], original_image.shape[0]))
        # Combine the result with the original image
        result = cv2.addWeighted(undist_image, 1, newwarp, 0.3, 0)

        result = self.add_text_to_image(result, text_lines)

        # self.tools.plot_image(result, im_name)
        return result
