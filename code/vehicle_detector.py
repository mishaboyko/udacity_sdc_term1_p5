import numpy as np
import cv2
import time

from sklearn.svm import LinearSVC
from skimage.feature import hog
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from scipy.ndimage.measurements import label

from tools import Tools
from image_processing_utilities import ImageProcessingUtilities
from vehicles_tracker import VehiclesTracker

class VehicleDetector:

    def __init__(self):
        self.svc = None
        self.X_scaler = None

        # set default parameters first
        self.color_space = 'YCrCb'      # Can be YCrCb, RGB, HSV, HLS, YUV, BGR2YCrCb, LUV
        self.hog_channel = 'ALL'        # Numbers of HOG Channels to extract. Value range: [0, 1, 2, ALL]
        self.orient = 9                 # HOG orientations
        self.pixel_per_cell = 8         # HOG pixels per cell
        self.cell_per_block = 2         # HOG cells per block
        self.spatial_size = (32, 32)    # Spatial binning dimensions
        self.hist_bins = 32             # Number of histogram bins

        self.image_scaling_checked = False

        self.tools = Tools()
        self.imageProcessing = ImageProcessingUtilities()
        self.vehiclesTracker = VehiclesTracker()

        # Load parameters if corresponding pickle file available
        classifier_params_count = 2
        self.svc, self.X_scaler = self.tools.load_params(classifier_params_count, 'svm_params.pkl')

        # case no parameters have been stored
        if self.svc is None:
            print("Training SVM")
            self.train_classifier()

    def bin_spatial(self, img):
        """
        Function to compute binned color features
        :param img: Image, from which a feature should be extracted
        :return: binned color features for the given image
        """
        # Use cv2.resize().ravel() to create the feature vector
        features = cv2.resize(img, self.spatial_size).ravel()
        # Return the feature vector
        return features

    def color_hist(self, img, bins_range=(0, 256)):
        """
        function to compute color histogram features
        :param img: Image, from which a feature should be extracted
        :param bins_range: range of the histogram bins.
            Case PNG images: matplotlib image reads them in on a scale [0, 1], but cv2.imread() scales them [0, 255].
            Case JPG images: matplotlib image reads them in on a scale [0, 255]
            Note: if you take an image that is scaled [0, 1] and change color spaces using cv2.cvtColor()
                    you'll get back an image scaled [0, 255]
        :return:
        """
        # Compute the histogram of the color channels separately
        channel1_hist = np.histogram(img[:, :, 0], bins=self.hist_bins, range=bins_range)
        channel2_hist = np.histogram(img[:, :, 1], bins=self.hist_bins, range=bins_range)
        channel3_hist = np.histogram(img[:, :, 2], bins=self.hist_bins, range=bins_range)
        # Concatenate the histograms into a single feature vector
        hist_features = np.concatenate((channel1_hist[0], channel2_hist[0], channel3_hist[0]))
        # Return the individual histograms, bin_centers and feature vector
        return hist_features

    def get_hog_features(self, img, vis=False, feature_vec=True):
        """
        Function to return HOG features and visualization
        :param img: Image, from which a feature should be extracted
        :param vis: boolean, if you want an HOG image to be rendered on-the-fly
        :param feature_vec: boolean, whether you want to obtain features as a vector
        :return:
        """
        # Call with two outputs if vis==True
        if vis:
            features, hog_image = hog(img, orientations=self.orient,
                                      pixels_per_cell=(self.pixel_per_cell, self.pixel_per_cell),
                                      cells_per_block=(self.cell_per_block, self.cell_per_block),
                                      block_norm='L2-Hys', visualize=vis,
                                      transform_sqrt=True, feature_vector=feature_vec)
            self.tools.plot_image(hog_image, "HOG image")
            return features
        # Otherwise call with one output
        else:
            features = hog(img, orientations=self.orient,
                           pixels_per_cell=(self.pixel_per_cell, self.pixel_per_cell),
                           cells_per_block=(self.cell_per_block, self.cell_per_block),
                           block_norm='L2-Hys', visualize=vis,
                           transform_sqrt=True, feature_vector=feature_vec)
            return features

    def extract_features(self, imgs, spatial_feat=True, hist_feat=True, hog_feat=True):
        """
        Function to extract features from a list of images

        :param imgs: array with image paths, what should be loaded and processed
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

            # import matplotlib.image as mpimg
            # image = mpimg.imread(file)
            # image = image.astype(np.float32)*255

            # use cv2.imread to scale PNG images in the [0, 255] range
            image = cv2.imread(file)

            if not self.image_scaling_checked:
                self.tools.check_image_scale(image)

            # apply color conversion if other than 'RGB'
            feature_image = self.imageProcessing.convert_color(image, self.color_space)

            if spatial_feat:
                spatial_features = self.bin_spatial(feature_image)
                file_features.append(spatial_features)
            if hist_feat:
                # Apply color_hist()
                hist_features = self.color_hist(feature_image)
                file_features.append(hist_features)
            if hog_feat:
                # Call get_hog_features() with vis=False, feature_vec=True
                if self.hog_channel == 'ALL':
                    hog_features = []
                    for channel in range(feature_image.shape[2]):
                        hog_features.append(self.get_hog_features(feature_image[:, :, channel], vis=False, feature_vec=True))
                    hog_features = np.ravel(hog_features)
                    # self.tools.plot_image(image, "original image")
                else:
                    hog_features = self.get_hog_features(feature_image[:, :, self.hog_channel], vis=False, feature_vec=True)
                # Append the new feature vector to the features list
                file_features.append(hog_features)
            features.append(np.concatenate(file_features))
        # Return list of feature vectors
        return features

    @staticmethod
    def add_heat(heatmap, bbox_list):
        # Iterate through list of bboxes
        for box in bbox_list:
            # Add += 1 for all pixels inside each bbox
            # Assuming each "box" takes the form ((x1, y1), (x2, y2))
            heatmap[box[0][1]:box[1][1], box[0][0]:box[1][0]] += 1

        # Return updated heatmap
        return heatmap  # Iterate through list of bboxes

    @staticmethod
    def apply_threshold(heatmap, threshold):
        # Zero out pixels below the threshold
        heatmap[heatmap <= threshold] = 0
        # Return thresholded map
        return heatmap

    def draw_labeled_bboxes(self, img, labels):
        # Iterate through all detected cars
        detections_per_frame = None
        font = cv2.FONT_HERSHEY_SIMPLEX
        fontScale = 1
        lineType = 2

        for car_number in range(1, labels[1] + 1):
            # Find pixels with each car_number label value
            nonzero = (labels[0] == car_number).nonzero()
            # Identify x and y values of those pixels
            nonzeroy = np.array(nonzero[0])
            nonzerox = np.array(nonzero[1])
            # Define a bounding box based on min/max x and y
            bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
            if detections_per_frame is None:
                detections_per_frame = [bbox]
            else:
                detections_per_frame.append(bbox)
            # Draw the box on the image for debugging purposes
            # cv2.rectangle(img, bbox[0], bbox[1], (124, 252, 0), 6)
            # cv2.putText(img, str(car_number), bbox[0], font, fontScale, (124, 252, 0), lineType)

        if detections_per_frame:
            self.vehiclesTracker.add_detections_in_frame(detections_per_frame)
            cars_bboxes = self.vehiclesTracker.get_vehicles_bboxes()
            for car_num, bbox in enumerate(cars_bboxes.values()):
                cv2.rectangle(img, bbox[0], bbox[1], (0, 0, 255), 6)
                cv2.putText(img, "car "+str(car_num), bbox[0], font, fontScale, (0, 0, 255), lineType)

        # Return the image
        return img

    def train_classifier(self):
        spatial_feat = True     # Spatial features on or off
        hist_feat = True        # Histogram features on or off
        hog_feat = True         # HOG features on or off

        cars, noncars = self.tools.get_car_noncar_images()

        # use extract_features() here
        car_features = self.extract_features(cars, spatial_feat=spatial_feat, hist_feat=hist_feat, hog_feat=hog_feat)

        notcar_features = self.extract_features(noncars, spatial_feat=spatial_feat, hist_feat=hist_feat,
                                                hog_feat=hog_feat)

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
        self.X_scaler = StandardScaler().fit(X_train)
        # Apply the scaler to X
        X_train = self.X_scaler.transform(X_train)
        X_test = self.X_scaler.transform(X_test)

        print('Using:', self.orient, 'orientations', self.pixel_per_cell,
              'pixels per cell and', self.cell_per_block, 'cells per block')
        print('Feature vector length:', len(X_train[0]))
        # Use a linear SVC
        self.svc = LinearSVC()
        # Check the training time for the SVC
        t = time.time()
        self.svc.fit(X_train, y_train)
        t2 = time.time()
        print(round(t2 - t, 2), 'Seconds to train SVC...')
        # Check the score of the SVC
        print('Test Accuracy of SVC = ', round(self.svc.score(X_test, y_test), 4))
        # Check the prediction time for a single sample
        t = time.time()

        self.tools.dump_parameters([self.svc, self.X_scaler], "svm_params")
        print("SVM Parameters successfully dumped")

    def detect_vehicles(self, img, output_image, ystart, ystop, scale):
        """
        Extract features using hog sub-sampling and make predictions

        :param img: an image, where vehicles should be found
        :param output_image: an image with already drawn detected lanes
        :param ystart: value on the y-axis of the image, where the search should start
        :param ystop: value on the y-axis of the image, where the search should end
        :param scale:
        :return:
        """

        #img = img.astype(np.float32) / 255
        #img_tosearch = img[ystart:ystop, :, :]
        img_tosearch = img[ystart:ystop, :, :]
        heat = np.zeros_like(img[:, :, 0]).astype(np.float)

        ctrans_tosearch = self.imageProcessing.convert_color(img_tosearch, conv='YCrCb')
        if scale != 1:
            imshape = ctrans_tosearch.shape
            ctrans_tosearch = cv2.resize(ctrans_tosearch, (np.int(imshape[1] / scale), np.int(imshape[0] / scale)))

        ch1 = ctrans_tosearch[:, :, 0]
        ch2 = ctrans_tosearch[:, :, 1]
        ch3 = ctrans_tosearch[:, :, 2]

        # Define blocks and steps as above
        nxblocks = (ch1.shape[1] // self.pixel_per_cell) - self.cell_per_block + 1
        nyblocks = (ch1.shape[0] // self.pixel_per_cell) - self.cell_per_block + 1
        nfeat_per_block = self.orient * self.cell_per_block ** 2

        # 64 was the original sampling rate, with 8 cells and 8 pix per cell
        window = 64
        nblocks_per_window = (window // self.pixel_per_cell) - self.cell_per_block + 1
        cells_per_step = 2  # Instead of overlap, define how many cells to step
        nxsteps = (nxblocks - nblocks_per_window) // cells_per_step + 1
        nysteps = (nyblocks - nblocks_per_window) // cells_per_step + 1

        # Compute individual channel HOG features for the entire image
        hog1 = self.get_hog_features(ch1, feature_vec=False)
        hog2 = self.get_hog_features(ch2, feature_vec=False)
        hog3 = self.get_hog_features(ch3, feature_vec=False)

        frame_detected_boxes = []

        for xb in range(nxsteps):
            for yb in range(nysteps):
                ypos = yb * cells_per_step
                xpos = xb * cells_per_step
                # Extract HOG for this patch
                hog_feat1 = hog1[ypos:ypos + nblocks_per_window, xpos:xpos + nblocks_per_window].ravel()
                hog_feat2 = hog2[ypos:ypos + nblocks_per_window, xpos:xpos + nblocks_per_window].ravel()
                hog_feat3 = hog3[ypos:ypos + nblocks_per_window, xpos:xpos + nblocks_per_window].ravel()
                hog_features = np.hstack((hog_feat1, hog_feat2, hog_feat3))

                xleft = xpos * self.pixel_per_cell
                ytop = ypos * self.pixel_per_cell

                # Extract the image patch
                subimg = cv2.resize(ctrans_tosearch[ytop:ytop + window, xleft:xleft + window], (64, 64))

                # Get color features
                spatial_features = self.bin_spatial(subimg)
                hist_features = self.color_hist(subimg)

                # Scale features and make a prediction
                test_features = self.X_scaler.transform(
                    np.hstack((spatial_features, hist_features, hog_features)).reshape(1, -1))
                test_prediction = self.svc.predict(test_features)

                if test_prediction == 1:
                    xbox_left = np.int(xleft * scale)
                    ytop_draw = np.int(ytop * scale)
                    win_draw = np.int(window * scale)
                    frame_detected_boxes.append(((xbox_left, ytop_draw + ystart),
                                                 (xbox_left + win_draw, ytop_draw + win_draw + ystart)))
                    # cv2.rectangle(output_image, frame_detected_boxes[-1][0], frame_detected_boxes[-1][1],
                    #               (0, 0, 255), 6)

        # Add heat to each box in box list
        heat = self.add_heat(heat, frame_detected_boxes)

        # Apply threshold to help remove false positives
        heat = self.apply_threshold(heat, 2)

        # Visualize the heatmap when displaying
        heatmap = np.clip(heat, 0, 255)
        # self.tools.plot_image(heatmap)

        # Find final boxes from heatmap using label function
        labels = label(heatmap)
        draw_img = self.draw_labeled_bboxes(output_image, labels)

        # uncomment below to plot heat map per frame
        # import matplotlib.pyplot as plt
        # fig = plt.figure()
        # plt.subplot(121)
        # plt.imshow(draw_img)
        # plt.title('Car Positions')
        # plt.subplot(122)
        # plt.imshow(heatmap, cmap='hot')
        # plt.title('Heat Map')
        # fig.tight_layout()
        # plt.show()

        return draw_img
