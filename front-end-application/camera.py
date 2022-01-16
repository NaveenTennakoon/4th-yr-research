import cv2

from threading import Thread

class VideoStream:
    def __init__(self, src=0):
		# initialize the video camera stream and read the first frame from the stream
        self.stream = cv2.VideoCapture(src)
        self.stream.set(3, 640)
        self.stream.set(4, 360)
        self.stream.set(cv2.CAP_PROP_FPS, 30)

        (self.grabbed, self.frame) = self.stream.read()

        self.stopped = False
        self.record = False
        self.captured_frames = []

    def start(self):
		# start the thread to read frames from the video stream
        self.stopped = False
        Thread(target=self.update, args=()).start()
        return self

    def update(self):
		# keep looping infinitely until the thread is stopped
        while True:
            # if the thread indicator variable is set, stop the thread
            if self.stopped:
                return

            # otherwise, read the next frame from the stream
            (self.grabbed, self.frame) = self.stream.read()

            if self.record:
                self.captured_frames.append(self.frame)

    def read(self):
        # return the frame most recently read
        return self.frame

    def start_recording(self):
        self.captured_frames = []
        self.record = True

    def stop_recording(self):
        # indicate that recording and the thread should be stopped
        self.record = False
        self.stopped = True
        
        return self.captured_frames