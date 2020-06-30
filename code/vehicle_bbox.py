import collections


class VehicleBBox:

    def __init__(self):
        # x values of the last N fits of the line. Ringbuffer
        self.ringbuffer_length = 8
        self.reliable_detection_sequence = 2
        self.recent_bboxes = collections.deque(maxlen=self.ringbuffer_length)
        self.averaged_bbox = ((0, 0), (0, 0))
        # give a measurement a chance not to be deleted if a vehicle haven't been detected in 2 consequent frames
        self.remove_in_frames = -1
        self.reset_remove_in_frames()
        self.bbox_hight = 100    # in px
        self.bbox_length = 200    # in px

    def reset_remove_in_frames(self):
        self.remove_in_frames = 2

    def update_averaged_bbox(self):
        point_top_left_x = 0
        point_top_left_y = 0
        point_bottom_right_x = 0
        point_bottom_right_y = 0
        for bbox in self.recent_bboxes:
            point_top_left_x = point_top_left_x + bbox[0][0]
            point_top_left_y = point_top_left_y + bbox[0][1]

            point_bottom_right_x = point_bottom_right_x + bbox[1][0]
            point_bottom_right_y = point_bottom_right_y + bbox[1][1]

        point_top_left_x = point_top_left_x / len(self.recent_bboxes)
        point_top_left_y = point_top_left_y / len(self.recent_bboxes)
        point_bottom_right_x = point_bottom_right_x / len(self.recent_bboxes)
        point_bottom_right_y = point_bottom_right_y / len(self.recent_bboxes)

        self.averaged_bbox = (int(point_top_left_x), int(point_top_left_y)), \
                             (int(point_bottom_right_x), int(point_bottom_right_y))

    def add_bbox(self, bbox):
        self.recent_bboxes.append(bbox)
        self.update_averaged_bbox()
        # reset counter if the vehicle is detected again
        self.reset_remove_in_frames()

    def get_bbox(self):
        if len(self.recent_bboxes) >= self.reliable_detection_sequence:
            # make frame of a fixed size
            return (self.averaged_bbox[0][0], self.averaged_bbox[0][1]), \
                   (self.averaged_bbox[0][0]+self.bbox_length, self.averaged_bbox[0][1]+self.bbox_hight)
        else:
            return {}

    def remove_oldest_detection(self):
        # decrease counter first
        if self.remove_in_frames == 0:
            self.recent_bboxes.popleft()
            self.reset_remove_in_frames()
        else:
            self.remove_in_frames = self.remove_in_frames - 1
        return not self.recent_bboxes
