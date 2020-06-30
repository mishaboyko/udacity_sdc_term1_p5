import collections
from vehicle_bbox import VehicleBBox

class VehiclesTracker:

    def __init__(self):
        # x values of the last 5 fits of the line. Ringbuffer
        self.vehicle_bboxes = dict()
        self.deviation_x = 60   # +/- in px
        self.deviation_y = 20   # +/- in px

    def get_most_suitable_tracker_id(self, bbox):
        best_match = {-1: (0, 0)}

        # Iterate through all vehicles bboxes and find the closest bbox to the new one
        for tracker_id in self.vehicle_bboxes.keys():
            # calculate absolute distance between a point of a new bbox and all other already tracked averaged bboxes
            x_diff = abs(self.vehicle_bboxes[tracker_id].averaged_bbox[0][0] - bbox[0][0])
            y_diff = abs(self.vehicle_bboxes[tracker_id].averaged_bbox[0][1] - bbox[0][1])

            if x_diff <= self.deviation_x and y_diff <= self.deviation_y:
                # remove previous best match if the new one is closer to already tracked bbox
                best_match_key = list(best_match.keys())[0]
                if (best_match_key < 0): # or (best_match[best_match_key][0] > x_diff and best_match[best_match_key][1] > y_diff):
                    best_match.clear()
                # set new best match
                best_match[tracker_id] = (x_diff, y_diff)

        # allow multiple matches for one detection
        return list(best_match.keys()) #[0]

    def add_detection(self, bbox):
        """
        Strategy of tracking labels over multiple frames haven't proven itself, since labels are assigned from left top
        to right bottom, new label in each frame.

        New strategy: compare top left corner of each bbox and group them correspondigly.
        :param bbox: a bounding box in format (topLeftPoint(x, y), bottomRightPoint(x, y))
        :return:
        """
        for tracker_id in self.get_most_suitable_tracker_id(bbox):
            if tracker_id in self.vehicle_bboxes:
                self.vehicle_bboxes[tracker_id].add_bbox(bbox)
            else:
                new_vehicle_bbox = VehicleBBox()
                new_vehicle_bbox.add_bbox(bbox)
                self.vehicle_bboxes[len(self.vehicle_bboxes)] = new_vehicle_bbox
        return len(self.vehicle_bboxes)-1

    def add_detections_in_frame(self, detections):
        consequent_detections = []
        detections_to_be_removed = []

        for detection in detections:
            consequent_detections.append(self.add_detection(detection))

        for tracked_id in self.vehicle_bboxes.keys():
            # if a vehicle was in past iteration(s) but not in the current one - remove one bbox occurrence
            if tracked_id not in consequent_detections:
                # remember the ID of the the vehicle_bbox that has to be removed.
                # Do not remove the bbox in this iteration from the item, because if this is a last bbox in the item,
                # you'll manipulate the sequence while iteration, which will result in error.
                detections_to_be_removed.append(tracked_id)

        for detection_to_be_removed in detections_to_be_removed:
            is_empty = self.vehicle_bboxes[detection_to_be_removed].remove_oldest_detection()
            if is_empty:
                self.vehicle_bboxes.pop(detection_to_be_removed)

    def get_vehicles_bboxes(self):
        averaged_vehicles_bboxes = {}
        for car_id in self.vehicle_bboxes.keys():
            bbox = self.vehicle_bboxes[car_id].get_bbox()
            if bbox:
                averaged_vehicles_bboxes[car_id] = bbox
        return averaged_vehicles_bboxes
