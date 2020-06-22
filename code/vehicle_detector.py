from tools import Tools
from sklearn import svm

class VehicleDetector:

    def __init__(self):
        self.tools = Tools()

    def train_classifier(self):
        self.tools.get_car_noncar_images()
