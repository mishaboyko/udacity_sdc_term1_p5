import collections


class DetectionTracker:
    def __init__(self):
        # number of last frames
        self.ringbuffer_length = 16
        # history of classifier's detections, collected over the last N frames
        self.prev_frames = collections.deque(maxlen=self.ringbuffer_length)
