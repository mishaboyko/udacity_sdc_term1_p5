import glob
import cv2
import numpy as np
import math
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import pickle


class Tools:

    @staticmethod
    def check_image_scale(image):
        rgb_dimensions = (image[3][0])
        if type(rgb_dimensions[0]) != np.uint8:
            print("Color range [0, 1]: {} is not supported. Exiting with error".format(type(rgb_dimensions[0])))
            print("Consider apply [0, 255] scaling: image = image.astype(np.float32)*255")
            # Scale down to [0, 1] color range
            # image = image.astype(np.float32)/255
            # Scale up to [0, 255] color range
            # image = image.astype(np.float32)*255

            exit(0)

    def get_car_noncar_images(self):
        project_path = "../"
        train_images_path = "train_images/"
        car_images_subdir = "vehicles/**/"
        noncar_images_subdir = "non-vehicles/**/"
        name_pattern = '*.png'

        cars = glob.glob(project_path + train_images_path + car_images_subdir + name_pattern, recursive=True)
        noncars = glob.glob(project_path + train_images_path + noncar_images_subdir + name_pattern, recursive=True)

        print("found {} car images and {} non-car images".format(len(cars), len(noncars)))

        return cars, noncars

    @staticmethod
    def get_image_from_dir(path, name_pattern):
        # reading in images from directory
        images = []
        image_names = glob.glob(path + name_pattern)

        for im_name in image_names:
            images.append(cv2.cvtColor(cv2.imread(im_name), cv2.COLOR_BGR2RGB))

            # Plotting option 1
            #cv2.imshow("image", mpimg.imread(im_name))
            #cv2.waitKey()

            # Plotting option 2
            #plt.imshow(mpimg.imread(im_name))
            #plt.show()
        return images, image_names

    @staticmethod
    def plot_dots(origin_image, warped_image, dots_origin, dots_warped):
        f, arr = plt.subplots(1, 2, figsize=(24, 9))
        f.tight_layout()
        arr[0].imshow(origin_image)
        arr[0].plot(dots_origin[0][0], dots_origin[0][1], '.')     # top left
        arr[0].plot(dots_origin[1][0], dots_origin[1][1], '.')     # top right
        arr[0].plot(dots_origin[2][0], dots_origin[2][1], '.')    # bottom right
        arr[0].plot(dots_origin[3][0], dots_origin[3][1], '.')     # bottom left

        arr[1].imshow(warped_image)
        arr[1].plot(dots_warped[0][0], dots_warped[0][1], '.')     # top left
        arr[1].plot(dots_warped[1][0], dots_warped[1][1], '.')     # top right
        arr[1].plot(dots_warped[2][0], dots_warped[2][1], '.')    # bottom right
        arr[1].plot(dots_warped[3][0], dots_warped[3][1], '.')     # bottom left
        plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
        plt.show()

    @staticmethod
    def plot_image(image, name="No name", text_overlay=""):
        # font = {'family': 'serif',
        #         'color': 'white',
        #         'weight': 'normal',
        #         'size': 12,
        #         }
        plt.imshow(image)
        plt.title(name)
        # plt.text(10, 100, text_overlay, fontdict=font)  # bbox=dict(fill=False, edgecolor='red', linewidth=2))

        plt.show()

        # store_path = "./output_images/lane_plane_overlays/"
        # plt.savefig(store_path+name+"_overlay.jpg")
        # plt.clf()

    @staticmethod
    def plot_images(image, undistorted_img, warped_image, names):
        f, arr = plt.subplots(2, 2, figsize=(24, 9))
        f.tight_layout()
        arr[0, 0].imshow(image)
        arr[0, 0].set_title(names[0], fontsize=20)
        arr[0, 1].imshow(undistorted_img)
        arr[0, 1].set_title(names[1], fontsize=20)
        arr[1, 0].imshow(warped_image, cmap='gray')
        arr[1, 0].set_title(names[2], fontsize=20)

        plt.subplots_adjust(left=0., right=1, top=0.95, bottom=0.05)
        plt.show()

        # store_path = "./output_images/color_gradient_thresholded/"
        # plt.savefig(store_path+names[0]+"thresh.jpg")

    @staticmethod
    def get_histogram(warped_image):
        # Grab only the bottom half of the image
        # Lane lines are likely to be mostly vertical nearest to the car
        bottom_half = warped_image[warped_image.shape[0] // 2:, :]

        # Sum across image pixels vertically - make sure to set `axis`
        # i.e. the highest areas of vertical lines should be larger values
        histogram = np.sum(bottom_half, axis=0)

        # Create histogram of image binary activations
        # Visualize the resulting histogram
        # plt.plot(histogram)
        # plt.show()

        return histogram

    @staticmethod
    def get_text_overlay(left_curverad, right_curverad, offset_from_mid):
        text_offset = "Vehicle is "

        if offset_from_mid < 0:
            text_offset = text_offset + "{}m left of center".format(abs(round(offset_from_mid, 2)))
        else:
            text_offset = text_offset + "{}m right of center".format(abs(round(offset_from_mid, 2)))

        text_radius = "Curvature radius (m) of lanes: left={}, right={}".format(math.ceil(left_curverad),
                                                                                math.ceil(right_curverad))
        return [text_radius, text_offset]

    @staticmethod
    def dump_parameters(params_array, file_name):
        project_path = "../"
        f = open(project_path+file_name+'.pkl', 'wb')
        for param in params_array:
            pickle.dump(param, f)
        f.close()

    @staticmethod
    def load_params(params_count, file_name):
        fh = open('../'+file_name, 'rb')
        params = []

        for pos in range(0, params_count):
            params.append(pickle.load(fh))

        fh.close()
        return params
