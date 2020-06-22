import numpy as np
import collections


# Class to receive the characteristics of each line detection
class Line:
    def __init__(self):
        # was the line detected in the last iteration?
        self.detected = False
        # x values of the last 5 fits of the line. Ringbuffer
        self.recent_xfitted = collections.deque(maxlen=5)
        # average x values of the fitted line over the last 5 iterations
        self.bestx = []
        # polynomial coefficients averaged over the last iteration
        self.best_fit = []
        # polynomial coefficients for the most recent fit
        self.current_fit = [np.array([False])]
        # radius of curvature of the line in some units
        self.radius_of_curvature = None
        # distance in meters of vehicle center from the line
        # negative value = vehicle far too left, positive value = vehicle far too right
        self.line_base_pos = None
        # difference in fit coefficients between last and new fits
        self.diffs = np.array([0, 0, 0], dtype='float')
        # x values for detected line pixels
        self.allx = None
        # y values for detected line pixels
        self.ally = None

    def add_polyfit(self, fit):
        if len(self.best_fit) == 0:
            self.best_fit = fit
            self.current_fit = fit
        else:
            self.best_fit[0] = (self.current_fit[0] + fit[0])/2
            self.best_fit[1] = (self.current_fit[1] + fit[1])/2
            self.best_fit[2] = (self.current_fit[2] + fit[2])/2
            self.current_fit = fit

    def calculate_bestx(self, xfitted):
        # invalidate bestx on each frame
        self.bestx = np.empty(len(xfitted), dtype=object)

        # redefine bestx on each frame
        for x_pos, x_value in enumerate(xfitted):
            self.bestx[x_pos] = x_value
            for xfit in self.recent_xfitted:
                self.bestx[x_pos] = self.bestx[x_pos] + xfit[x_pos]

            # calculate average
            self.bestx[x_pos] = self.bestx[x_pos] / (len(self.recent_xfitted)+1)

    def add_xfitted(self, xfitted):
        self.recent_xfitted.append(xfitted)
        self.calculate_bestx(xfitted)

    def reset_line(self):
        self.detected = False
        self.recent_xfitted = []
        self.bestx = None
        self.best_fit = None
        self.current_fit = [np.array([False])]
        self.radius_of_curvature = None
        self.line_base_pos = None
        self.diffs = np.array([0, 0, 0], dtype='float')
        self.allx = None
        self.ally = None

