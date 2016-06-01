import cv2
import asyncio
import aiohttp
import logging
import base64
import time
import json
import numpy as np
from queue import Queue
from threading import Thread


class ImageRecognitionService():
    ''' start open cv related methods in the main thread '''

    def __init__(self):
        self._logger = logging.getLogger(__name__)
        logging.basicConfig(level=logging.DEBUG)
        self._capture = cv2.VideoCapture(0)

        # communication channels with the worker thread
        self._raw_channel = None
        self._processed_channel = None

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
            x = int(face['fdBoundingPoly']['vertices'][0].get('x', 0.0))
            y = int(face['fdBoundingPoly']['vertices'][0].get('y', 0.0))
            w = face['fdBoundingPoly']['vertices'][1].get('x', 0.0) - x
            h = face['fdBoundingPoly']['vertices'][2].get('y', 0.0) - y

            x_center = int(x + w / 2)
            y_center = int(y + h / 2)
            y_bottom = int(y + h)

            self._logger.debug('Center of the face is at: x {} / y {}'.format(x_center, y_center))

            cv2.circle(image, (x_center, y_center), int(w / 2), (0, 255, 0), 2)

            # place mood under the circle
            detected_emotions = self._getemotion(face)
            very_likely_emotions = ''
            likely_emotions = ''
            possible_emotions = ''

            for emotion in detected_emotions['VERY_LIKELY']:
                very_likely_emotions += emotion + '\n'

            for emotion in detected_emotions['LIKELY']:
                likely_emotions += emotion + '\n'

            for emotion in detected_emotions['POSSIBLE']:
                possible_emotions += emotion + '\n'

            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(image, very_likely_emotions, (x, y_bottom + 20), font, 1, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.putText(image, likely_emotions, (x, y_bottom + 50), font, 1, (255, 255, 0), 2, cv2.LINE_AA)
            cv2.putText(image, possible_emotions, (x, y_bottom + 80), font, 1, (0, 0, 255), 2, cv2.LINE_AA)

        return image

    def _drawrectangle(self, json_response, image):
        faces = json_response['responses'][0]['faceAnnotations']

        for face in faces:
            box = [(v.get('x', 0.0), v.get('y', 0.0)) for v in face['fdBoundingPoly']['vertices']]
            self._logger.debug('box: {}'.format(box))
            cv2.rectangle(image, box[0], box[2], (0, 255, 0), 2)

        return image

    def _drawtextrectangle(self, json_response, image):
        faces = json_response['responses'][0]['textAnnotations']

        for face in faces:
            box = [(v.get('x', 0.0), v.get('y', 0.0)) for v in face['boundingPoly']['vertices']]
            self._logger.debug('box: {}'.format(box))
            cv2.rectangle(image, box[0], box[2], (0, 255, 0), 2)

        return image


    def run(self):
        print('Image recognition service has been started')

        # instanciate the channels
        self._raw_channel = Queue()
        self._processed_channel = Queue()

        # Google Image Recognition Worker
        googlecloudworker = GoogleCloudWorker(self._raw_channel, self._processed_channel)
        googlecloudworker.daemon = True
        googlecloudworker.start()

        while True:
            # add a captured frame to the raw_queue
            self._raw_channel.put(self._getframe())

            # take a picture from the processed queue, put in the marks and show it
            if not self._processed_channel.empty():
                self._logger.info('Number of processed images in the queue: {}'
                                  .format(self._processed_channel.qsize()))
                response = self._processed_channel.get()

                # the first few responses are empty because of the camera warmup time
                if bool(response[0]['responses'][0]):
                    processed_frame = self._drawtextrectangle(response[0], response[1])
                    #processed_frame = self._drawmostlikelyemotion(response[0], response[1])

                    cv2.imshow('image', processed_frame)
                    cv2.waitKey(500)

        self._capture.release()
        cv2.destroyAllWindows()


class GoogleCloudWorker(Thread):
    ''' worker class which is started in its own thread and is responsible for handling all the requests to the Google
    vision API '''

    def __init__(self, raw_channel, processed_channel):
        Thread.__init__(self)
        self._loop = asyncio.get_event_loop()
        self._raw_channel = raw_channel
        self._processed_channel = processed_channel
        self._logger = logging.getLogger(__name__)
        self._API_KEY = ''
        self._GOOGLE_VISION_API_ENDPOINT = 'https://vision.googleapis.com/v1/images:annotate?key={}'\
            .format(self._API_KEY)

    @staticmethod
    def _buildrequest(image_str, max_results, detection_type):
        request = {'requests':
            [
                {
                    'image':
                        {
                            'content': base64.b64encode(image_str).decode('UTF-8')
                        },
                    'features':
                        [
                            {
                                'type': detection_type,
                                'maxResults': max_results
                            }
                        ],
                    'imageContext': {
                        'languageHints': [
                            'en'
                        ]
                    }
                }
            ]
        }
        return request

    @asyncio.coroutine
    def _executefacerecognition(self, session, image, max_results=4):
        # convert cv2 to .jpg to make it compatible for google vision api
        image_buf = cv2.imencode('.jpg', image)[1]
        image_str = np.array(image_buf).tostring()

        request = self._buildrequest(image_str, max_results, 'FACE_DETECTION')

        response = yield from session.post(self._GOOGLE_VISION_API_ENDPOINT, data=json.dumps(request))
        response_json = yield from response.json()
        self._logger.debug('response: {}'.format(response_json))

        return response_json, image

    @asyncio.coroutine
    def _executetextrecognition(self, session, image, max_results=4):

        # do image preprocessing to increase accuracy
        image_prep = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # adaptive
        #image_prep = cv2.medianBlur(image_prep, 5)
        #image_prep = cv2.adaptiveThreshold(image_prep, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

        # otsu
        #image_prep = cv2.GaussianBlur(image_prep, (5, 5), 0)
        image_prep = cv2.threshold(image_prep, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]

        # convert cv2 to .jpg to make it compatible for google vision api
        image_buf = cv2.imencode('.jpg', image_prep)[1]
        image_str = np.array(image_buf).tostring()

        request = self._buildrequest(image_str, max_results, 'TEXT_DETECTION')

        response = yield from session.post(self._GOOGLE_VISION_API_ENDPOINT, data=json.dumps(request))
        response_json = yield from response.json()
        self._logger.debug('response: {}'.format(response_json))

        return response_json, image_prep

    def run(self):
        print('GoogleCloudWorker: Vision API worker has been started')

        while True:

            if not self._raw_channel.empty():
                self._logger.info('GoogleCloudWorker: There are {} unprocessed pictures in the channel'
                                  .format(self._raw_channel.qsize()))
                image = self._raw_channel.get()
                with aiohttp.ClientSession(loop=self._loop) as session:
                    response = self._loop.run_until_complete(
                        #self._executefacerecognition(session, image)
                        self._executetextrecognition(session, image)
                    )
                self._processed_channel.put(response)

if __name__ == '__main__':
    imageservice = ImageRecognitionService()
    imageservice.run()
