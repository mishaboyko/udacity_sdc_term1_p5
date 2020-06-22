import numpy as np
import cv2
import matplotlib.pyplot as plt
from image_processing_utilities import ImageProcessingUtilities
# Imports to edit/save/watch video clips
from moviepy.editor import VideoFileClip

class VideoStreamProcessing:

    def __init__(self):
        self.imageProcessing = ImageProcessingUtilities()

    @staticmethod
    def fit_poly(img_shape, leftx, lefty, rightx, righty):
        # Fit a second order polynomial to each with np.polyfit()
        left_fit = np.polyfit(lefty, leftx, 2)
        right_fit = np.polyfit(righty, rightx, 2)

        # Generate x and y values for plotting
        ploty = np.linspace(0, img_shape[0] - 1, img_shape[0])

        # Calculate both polynomials using ploty, left_fit and right_fit
        left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
        right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]

        return left_fit, right_fit, left_fitx, right_fitx, ploty

    def search_around_poly(self, binary_warped, left_right_lines):
        margin = 100

        # Grab activated pixels
        nonzero = binary_warped.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])

        left_fit = left_right_lines[0].best_fit
        right_fit = left_right_lines[1].best_fit

        # Set the area of search based on activated x-values
        # within the +/- margin of our polynomial function
        # Hint: consider the window areas for the similarly named variables
        # in the previous quiz, but change the windows to our new search area
        left_lane_inds = ((nonzerox > (left_fit[0] * (nonzeroy ** 2) + left_fit[1] * nonzeroy +
                                       left_fit[2] - margin)) & (nonzerox < (left_fit[0] * (nonzeroy ** 2) +
                                                                             left_fit[1] * nonzeroy + left_fit[
                                                                                 2] + margin)))
        right_lane_inds = ((nonzerox > (right_fit[0] * (nonzeroy ** 2) + right_fit[1] * nonzeroy +
                                        right_fit[2] - margin)) & (nonzerox < (right_fit[0] * (nonzeroy ** 2) +
                                                                               right_fit[1] * nonzeroy + right_fit[
                                                                                   2] + margin)))

        # Again, extract left and right line pixel positions
        leftx = nonzerox[left_lane_inds]
        lefty = nonzeroy[left_lane_inds]
        rightx = nonzerox[right_lane_inds]
        righty = nonzeroy[right_lane_inds]

        # Fit new polynomials
        left_fit, right_fit, left_fitx, right_fitx, ploty = self.fit_poly(binary_warped.shape, leftx, lefty, rightx, righty)
        """
        Idea for optimization: Add try/catch block and reset values if no left_fitx & right_fitx found.
        """

        ## Visualization ##
        # Create an image to draw on and an image to show the selection window
        out_img = np.dstack((binary_warped, binary_warped, binary_warped)) * 255
        window_img = np.zeros_like(out_img)
        # Color in left and right line pixels
        out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
        out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]

        # Generate a polygon to illustrate the search window area
        # And recast the x and y points into usable format for cv2.fillPoly()
        left_line_window1 = np.array([np.transpose(np.vstack([left_fitx - margin, ploty]))])
        left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([left_fitx + margin,
                                                                        ploty])))])
        left_line_pts = np.hstack((left_line_window1, left_line_window2))
        right_line_window1 = np.array([np.transpose(np.vstack([right_fitx - margin, ploty]))])
        right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([right_fitx + margin,
                                                                         ploty])))])
        right_line_pts = np.hstack((right_line_window1, right_line_window2))

        # Draw the lane onto the warped blank image
        cv2.fillPoly(window_img, np.int_([left_line_pts]), (0, 255, 0))
        cv2.fillPoly(window_img, np.int_([right_line_pts]), (0, 255, 0))
        out_img = cv2.addWeighted(out_img, 1, window_img, 0.3, 0)

        # Plot the polynomial lines onto the image
        # plt.plot(left_fitx, ploty, color='yellow')
        # plt.plot(right_fitx, ploty, color='yellow')
        #
        # # View your output
        # plt.imshow(out_img)
        # plt.show()

        ## End visualization steps ##

        left_right_lines[0].detected = True
        left_right_lines[1].detected = True
        left_right_lines[0].add_polyfit(left_fit)
        left_right_lines[1].add_polyfit(right_fit)
        left_right_lines[0].add_xfitted(left_fitx)
        left_right_lines[1].add_xfitted(right_fitx)

        offset_from_middle_m = self.imageProcessing.calculate_distance_from_lane_center(left_fitx[len(left_fitx)-1],
                                                                                        right_fitx[len(right_fitx)-1])
        left_right_lines[0].line_base_pos = offset_from_middle_m
        left_right_lines[1].line_base_pos = offset_from_middle_m
        left_right_lines[0].allx = leftx
        left_right_lines[0].ally = lefty
        left_right_lines[1].allx = rightx
        left_right_lines[1].ally = righty

        return out_img, left_right_lines

    @staticmethod
    def process_video(process_frame):
        project_path = "./"
        video_input = "project_video.mp4"
        video_output = 'output_video/project_video_out.mp4'

        clip = VideoFileClip(project_path+video_input)#.subclip(0, 5)
        white_clip = clip.fl_image(process_frame)  # NOTE: this function expects color images!!
        white_clip.write_videofile(project_path+video_output, audio=False)

