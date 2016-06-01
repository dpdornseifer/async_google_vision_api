import logging
import cv2
import time
from queue import Queue
from GoogleCloudWorker import GoogleCloudWorker
from RestEndpointWorker import RestEndpointWorker


class ImageRecognitionService():
    ''' start the main thread is primarily for showing the processed images '''

    def __init__(self, detectiontype='text', restinterface='true'):
        self._logger = logging.getLogger(__name__)
        logging.basicConfig(level=logging.INFO)

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

    @staticmethod
    def _getemotion(face):
        ''' filter return for possible to very_likely emotions '''

        indicators = ['POSSIBLE', 'LIKELY', 'VERY_LIKELY']
        google_face_emotions = ['joyLikelihood', 'sorrowLikelihood', 'angerLikelihood', 'surpriseLikelihood',
                    'underExposedLikelihood', 'blurredLikelihood', 'headwearLikelihood']

        # detected emotions ordered by likelihood
        detected_emotions = {'VERY_LIKELY': [], 'LIKELY': [], 'POSSIBLE': []}

        for emotion in google_face_emotions:
            if face[emotion] in indicators:
                detected_emotions[face[emotion]] += [emotion]

        return detected_emotions

    def _drawmostlikelyemotion(self, json_response, image):
        ''' draw a simple emoticon '''

        faces = json_response['responses'][0]['faceAnnotations']

        for face in faces:
            x = face['fdBoundingPoly']['vertices'][0].get('x', 0.0)
            y = face['fdBoundingPoly']['vertices'][0].get('y', 0.0)
            w = face['fdBoundingPoly']['vertices'][1].get('x', 0.0) - x
            h = face['fdBoundingPoly']['vertices'][2].get('y', 0.0) - y

            x_center = int(x + w / 2)
            y_center = int(y + h / 2)
            y_bottom = y + h

            self._logger.debug('Center of the face is at: x {} / y {}'.format(x_center, y_center))

            cv2.circle(image, (x_center, y_center), int(w / 2), (0, 255, 0), 2)

            # place mood under the circle
            detected_emotions = self._getemotion(face)
            very_likely_emotions = ''
            likely_emotions = ''
            possible_emotions = ''

            for emotion in detected_emotions['VERY_LIKELY']:
                very_likely_emotions += ' ' + emotion + '\n'

            for emotion in detected_emotions['LIKELY']:
                likely_emotions += ' ' + emotion + '\n'

            for emotion in detected_emotions['POSSIBLE']:
                possible_emotions += ' ' + emotion + '\n'

            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(image, very_likely_emotions, (x, y_bottom + 20), font, 1, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.putText(image, likely_emotions, (x, y_bottom + 50), font, 1, (255, 255, 0), 2, cv2.LINE_AA)
            cv2.putText(image, possible_emotions, (x, y_bottom + 80), font, 1, (0, 0, 255), 2, cv2.LINE_AA)

        return image

    def _drawtextrectangle(self, json_response, image):
        ''' draw a simple rectangle around the text '''

        texts = json_response['responses'][0]['textAnnotations']

        for text in texts:
            box = [(v.get('x', 0.0), v.get('y', 0.0)) for v in text['boundingPoly']['vertices']]
            self._logger.debug('box: {}'.format(box))
            cv2.rectangle(image, box[0], box[2], (0, 255, 0), 2)

        return image

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

                    # right now just text and face recognitin are supported
                    if self._detectiontype == 'text':
                        processed_frame = self._drawtextrectangle(response[0], response[1])

                    else:
                        processed_frame = self._drawmostlikelyemotion(response[0], response[1])

                    cv2.imshow('image', processed_frame)
                    cv2.waitKey(500)

        if self._restinterface != 'true':
            self._capture.release()
        cv2.destroyAllWindows()
