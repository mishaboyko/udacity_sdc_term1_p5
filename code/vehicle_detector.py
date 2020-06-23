import matplotlib.image as mpimg
from sklearn.svm import LinearSVC

from tools import Tools
import numpy as np
import cv2
from skimage.feature import hog
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import time
from inspect import currentframe, getframeinfo


class VehicleDetector:

    def __init__(self):
        self.tools = Tools()
        self.image_scaling_checked = False

    @staticmethod
    def bin_spatial(img, size=(32, 32)):
        """
        Function to compute binned color features
        :param img: Image, from which a feature should be extracted
        :param size: Spatial binning dimensions
        :return: binned color features for the given image
        """
        # Use cv2.resize().ravel() to create the feature vector
        features = cv2.resize(img, size).ravel()
        # Return the feature vector
        return features

    @staticmethod
    def color_hist(img, nbins=32, bins_range=(0, 256)):
        """
        function to compute color histogram features
        :param img: Image, from which a feature should be extracted
        :param nbins: Number of histogram bins
        :param bins_range: range of the histogram bins.
            Case PNG images: matplotlib image reads them in on a scale [0, 1], but cv2.imread() scales them [0, 255].
            Case JPG images: matplotlib image reads them in on a scale [0, 255]
            Note: if you take an image that is scaled [0, 1] and change color spaces using cv2.cvtColor()
                    you'll get back an image scaled [0, 255]
        :return:
        """
        # Compute the histogram of the color channels separately
        channel1_hist = np.histogram(img[:, :, 0], bins=nbins, range=bins_range)
        channel2_hist = np.histogram(img[:, :, 1], bins=nbins, range=bins_range)
        channel3_hist = np.histogram(img[:, :, 2], bins=nbins, range=bins_range)
        # Concatenate the histograms into a single feature vector
        hist_features = np.concatenate((channel1_hist[0], channel2_hist[0], channel3_hist[0]))
        # Return the individual histograms, bin_centers and feature vector
        return hist_features

    def get_hog_features(self, img, orient, pix_per_cell, cell_per_block,
                         vis=False, feature_vec=True):
        """
        Function to return HOG features and visualization
        :param img: Image, from which a feature should be extracted
        :param orient: Histogram Orientation Gradient (HOG) orientations
        :param pix_per_cell: HOG pixels per cell
        :param cell_per_block: HOG cells per block
        :param vis: boolean, if you want an HOG image to be rendered on-the-fly
        :param feature_vec: boolean, whether you want to obtain features as a vector
        :return:
        """
        # Call with two outputs if vis==True
        if vis:

            features, hog_image = hog(img, orientations=orient,
                                      pixels_per_cell=(pix_per_cell, pix_per_cell),
                                      cells_per_block=(cell_per_block, cell_per_block),
                                      block_norm='L2-Hys', visualize=vis,
                                      transform_sqrt=True, feature_vector=feature_vec)
            self.tools.plot_image(hog_image, "HOG image")
            return features
        # Otherwise call with one output
        else:
            features = hog(img, orientations=orient,
                           pixels_per_cell=(pix_per_cell, pix_per_cell),
                           cells_per_block=(cell_per_block, cell_per_block),
                           block_norm='L2-Hys', visualize=vis,
                           transform_sqrt=True, feature_vector=feature_vec)
            return features

    def extract_features(self, imgs, color_space='RGB', spatial_size=(32, 32),
                         hist_bins=32, orient=9,
                         pix_per_cell=8, cell_per_block=2, hog_channel=0,
                         spatial_feat=True, hist_feat=True, hog_feat=True):
        """
        Function to extract features from a list of images

        :param imgs: array with image paths, what should be loaded and processed
        :param color_space: color space, in which each image should be converted and afterwards proessed
        :param spatial_size: Spatial binning dimensions
        :param hist_bins: Number of histogram bins
        :param orient: Histogram Orientation Gradient (HOG) orientations
        :param pix_per_cell: HOG pixels per cell
        :param cell_per_block: HOG cells per block
        :param hog_channel: type of a HOG Channel. Value range: [0, 1, 2, ALL]
        :param spatial_feat: Boolean for Spatial features [on, off]
        :param hist_feat: Boolean for Histogram features [on, off]
        :param hog_feat: Boolean for HOG features [on, off]
        :return: array of extracted image features.
                Depending on bool values, it can contain: Spatial, Histogram and HOG features.
        """
        # Create a list to append feature vectors to
        features = []
        # Iterate through the list of images
        for file in imgs:
            file_features = []

            # use cv2.imread to scale PNG images in the [0, 255] range
            #image = mpimg.imread(file)
            image = cv2.imread(file)

            if not self.image_scaling_checked:
                self.tools.check_image_scale(image)

            # apply color conversion if other than 'RGB'
            if color_space != 'RGB':
                if color_space == 'HSV':
                    feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
                elif color_space == 'LUV':
                    feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2LUV)
                elif color_space == 'HLS':
                    feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
                elif color_space == 'YUV':
                    feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
                elif color_space == 'YCrCb':
                    feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)
            else:
                feature_image = np.copy(image)

            if spatial_feat:
                spatial_features = self.bin_spatial(feature_image, size=spatial_size)
                file_features.append(spatial_features)
            if hist_feat:
                # Apply color_hist()
                hist_features = self.color_hist(feature_image, nbins=hist_bins)
                file_features.append(hist_features)
            if hog_feat:
                # Call get_hog_features() with vis=False, feature_vec=True
                if hog_channel == 'ALL':
                    hog_features = []
                    for channel in range(feature_image.shape[2]):
                        hog_features.append(self.get_hog_features(feature_image[:, :, channel],
                                                                  orient, pix_per_cell, cell_per_block,
                                                                  vis=False, feature_vec=True))
                    hog_features = np.ravel(hog_features)
                else:
                    hog_features = self.get_hog_features(feature_image[:, :, hog_channel], orient,
                                                    pix_per_cell, cell_per_block, vis=False, feature_vec=True)
                    #self.tools.plot_image(image, "original image")
                # Append the new feature vector to the features list
                file_features.append(hog_features)
            features.append(np.concatenate(file_features))
        # Return list of feature vectors
        return features

    def train_classifier(self):
        color_space = 'RGB'  # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
        orient = 9  # HOG orientations
        pix_per_cell = 8  # HOG pixels per cell
        cell_per_block = 2  # HOG cells per block
        hog_channel = 0  # Can be 0, 1, 2, or "ALL"
        spatial_size = (16, 16)  # Spatial binning dimensions
        hist_bins = 16  # Number of histogram bins
        spatial_feat = True  # Spatial features on or off
        hist_feat = True  # Histogram features on or off
        hog_feat = True  # HOG features on or off
        y_start_stop = [None, None]  # Min and max in y to search in slide_window()

        cars, noncars = self.tools.get_car_noncar_images()

        # use extract_features() here
        car_features = self.extract_features(cars, color_space=color_space,
                                             spatial_size=spatial_size, hist_bins=hist_bins,
                                             orient=orient, pix_per_cell=pix_per_cell,
                                             cell_per_block=cell_per_block,
                                             hog_channel=hog_channel, spatial_feat=spatial_feat,
                                             hist_feat=hist_feat, hog_feat=hog_feat)

        notcar_features = self.extract_features(noncars, color_space=color_space,
                                                spatial_size=spatial_size, hist_bins=hist_bins,
                                                orient=orient, pix_per_cell=pix_per_cell,
                                                cell_per_block=cell_per_block,
                                                hog_channel=hog_channel, spatial_feat=spatial_feat,
                                                hist_feat=hist_feat, hog_feat=hog_feat)

        print("car_features and notcar_features are successfully extracted")

        # Create an array stack of feature vectors
        X = np.vstack((car_features, notcar_features)).astype(np.float64)

        # Define the labels vector
        y = np.hstack((np.ones(len(car_features)), np.zeros(len(notcar_features))))

        # Split up data into randomized training and test sets
        rand_state = np.random.randint(0, 100)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=rand_state)

        # Fit a per-column scaler
        X_scaler = StandardScaler().fit(X_train)
        # Apply the scaler to X
        X_train = X_scaler.transform(X_train)
        X_test = X_scaler.transform(X_test)

        print('Using:', orient, 'orientations', pix_per_cell,
              'pixels per cell and', cell_per_block, 'cells per block')
        print('Feature vector length:', len(X_train[0]))
        # Use a linear SVC
        svc = LinearSVC()
        # Check the training time for the SVC
        t = time.time()
        svc.fit(X_train, y_train)
        t2 = time.time()
        print(round(t2 - t, 2), 'Seconds to train SVC...')
        # Check the score of the SVC
        print('Test Accuracy of SVC = ', round(svc.score(X_test, y_test), 4))
        # Check the prediction time for a single sample
        t = time.time()

        self.tools.dump_parameters([svc, X_scaler, orient, pix_per_cell, cell_per_block, spatial_size, hist_bins],
                                   "svm_params")
        print("SVM Parameters successfully dumped")

        cf = currentframe()
        filename = getframeinfo(cf).filename
        print("Proceed in file {}, line {}".format(filename, cf.f_lineno ))

        # image = mpimg.imread('bbox-example-image.jpg')
        # draw_image = np.copy(image)
        # Uncomment the following line if you extracted training
        # data from .png images (scaled 0 to 1 by mpimg) and the
        # image you are searching is a .jpg (scaled 0 to 255)
        # image = image.astype(np.float32)/255

        # windows = slide_window(image, x_start_stop=[None, None], y_start_stop=y_start_stop,
        #                        xy_window=(96, 96), xy_overlap=(0.5, 0.5))
        #
        # hot_windows = search_windows(image, windows, svc, X_scaler, color_space=color_space,
        #                              spatial_size=spatial_size, hist_bins=hist_bins,
        #                              orient=orient, pix_per_cell=pix_per_cell,
        #                              cell_per_block=cell_per_block,
        #                              hog_channel=hog_channel, spatial_feat=spatial_feat,
        #                              hist_feat=hist_feat, hog_feat=hog_feat)
        #
        # window_img = draw_boxes(draw_image, hot_windows, color=(0, 0, 255), thick=6)
        # plt.imshow(window_img)
