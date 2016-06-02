import logging
import time
import cv2
from queue import Queue
from utils import image_utils
from worker.GoogleCloudWorker import GoogleCloudWorker
from worker.RestEndpointWorker import RestEndpointWorker


class ImageRecognitionService():
    ''' start the main thread is primarily for showing the processed images '''

    def __init__(self, detectiontype='text', restinterface='true'):
        self._logger = logging.getLogger(__name__)
        logging.basicConfig(level=logging.INFO,
                            format='[%(levelname)s] (%(threadName)-10s) %(message)s',)

        self._detectiontype = detectiontype
        self._restinterface = restinterface

        # communication channels with the worker thread
        self._raw_channel = None
        self._processed_channel = None

        # class wide webcam interface
        if restinterface != 'true':
            self._capture = cv2.VideoCapture(0)
        else:
            self._capture = None

    def _getframe(self):
        ''' use the opencv library to take a frame from the webcam '''

        ret, frame = self._capture.read()
        time.sleep(2)
        return frame


    def run(self):
        print('Image recognition service has been started')

        # instanciate the channels
        self._raw_channel = Queue()
        self._processed_channel = Queue()

        if self._restinterface == 'true':
            # Rest API server / eventloop
            rest_eventloop = RestEndpointWorker(8080, '0.0.0.0', self._raw_channel, self._processed_channel)
            rest_eventloop.daemon = True
            rest_eventloop.start()

        else:
            # instanciate locale image processing module / Google Image Recognition Worker
            googlecloudworker = GoogleCloudWorker(self._raw_channel, self._processed_channel, self._detectiontype)
            googlecloudworker.daemon = True
            googlecloudworker.start()

        while True:

            if self._restinterface != 'true':
                # add a captured frame to the raw_queue
                self._raw_channel.put(self._getframe())

            # take a picture from the processed queue, put in the marks and show it
            if not self._processed_channel.empty():
                self._logger.info('Number of processed images in the queue: {}'
                                  .format(self._processed_channel.qsize()))
                response = self._processed_channel.get()

                # the first few responses are empty because of the camera warmup time
                if bool(response[0]['responses'][0]):
                    if self._detectiontype == 'text':
                        processed_frame = image_utils.drawtextrectangle(response[0], response[1])

                    else:
                        faces = response[0]['responses'][0]['faceAnnotations']
                        processed_frame = image_utils.drawmostlikelyemotion(faces, response[1])

                    cv2.imshow('image', processed_frame)
                    cv2.waitKey(500)

        if self._restinterface != 'true':
            self._capture.release()
        cv2.destroyAllWindows()
