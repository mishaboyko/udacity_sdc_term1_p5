import collections


class VehiclesTracker:

    def __init__(self):
        # x values of the last 5 fits of the line. Ringbuffer
        self.recent_vehicles_bboxes = {}
        self.ringbuffer_length = 15
        self.reliable_detection_sequence = 15

    def add_bbox(self, car_number, bbox):
        if car_number in self.recent_vehicles_bboxes:
            # bboxes_previous_frames = self.recent_vehicles_bboxes.get(car_number)
            # bboxes_previous_frames.append(bbox)
            # self.recent_vehicles_bboxes[car_number] = bboxes_previous_frames
            self.recent_vehicles_bboxes[car_number].append(bbox)
        else:
            last_frames_detection = collections.deque(maxlen=self.ringbuffer_length)
            last_frames_detection.append(bbox)
            self.recent_vehicles_bboxes[car_number] = last_frames_detection

    def add_cars_in_frame(self, cars_per_frame):
        for car_in_frame in cars_per_frame:
            self.add_bbox(car_in_frame, cars_per_frame[car_in_frame])
        self.get_cars_sanitized(list(cars_per_frame.keys()))

    def get_cars_sanitized(self, car_ids_current_frame):
        recently_detected_vehicles = list(self.recent_vehicles_bboxes.keys())
        for recently_detected_vehicle in recently_detected_vehicles:
            if recently_detected_vehicle not in car_ids_current_frame:
                # if a vehicle was in past iteration(s) but not in the current one - remove one bbox occurrence
                self.recent_vehicles_bboxes[recently_detected_vehicle].popleft()

                # remove dict item if last bbox, assigned to it, has been removed
                if not self.recent_vehicles_bboxes[recently_detected_vehicle]:
                    self.recent_vehicles_bboxes.pop(recently_detected_vehicle)

    @staticmethod
    def get_averaged_bbox(bboxes):
        point_top_left_x = 0
        point_top_left_y = 0
        point_bottom_right_x = 0
        point_bottom_right_y = 0
        for bbox in bboxes:
            point_top_left_x = point_top_left_x + bbox[0][0]
            point_top_left_y = point_top_left_y + bbox[0][1]

            point_bottom_right_x = point_bottom_right_x + bbox[1][0]
            point_bottom_right_y = point_bottom_right_y + bbox[1][1]
        point_top_left_x = point_top_left_x/len(bboxes)
        point_top_left_y = point_top_left_y/len(bboxes)

        point_bottom_right_x = point_bottom_right_x/len(bboxes)
        point_bottom_right_y = point_bottom_right_y/len(bboxes)

        return (int(point_top_left_x), int(point_top_left_y)), (int(point_bottom_right_x), int(point_bottom_right_y))

    def get_smoothed_vehicles_bboxes(self):
        smoothed_bboxes = {}
        for car_id in self.recent_vehicles_bboxes:
            if len(self.recent_vehicles_bboxes[car_id]) >= self.reliable_detection_sequence:
                smoothed_bboxes[car_id] = self.get_averaged_bbox(self.recent_vehicles_bboxes[car_id])
        return smoothed_bboxes
