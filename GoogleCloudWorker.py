import asyncio
import json
import logging
import aiohttp
import cv2
import numpy as np
import base64
from threading import Thread


class GoogleCloudWorker(Thread):
    ''' worker class which is started in its own thread and is responsible for handling all the requests to the Google
    vision API '''

    def __init__(self, raw_channel, processed_channel, detectiontype='text'):
        Thread.__init__(self)
        self._loop = asyncio.get_event_loop()

        self._detectiontype = detectiontype
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
    def _executetextrecognition(self, session, image, max_results=4, threshold='otsu'):

        # do image preprocessing to increase accuracy
        image_prep = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        if threshold == 'adaptive':
            # adaptive threshold
            image_prep = cv2.medianBlur(image_prep, 5)
            image_prep = cv2.adaptiveThreshold(image_prep, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

        else:
            # otsu
            image_prep = cv2.GaussianBlur(image_prep, (5, 5), 0)
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

                if self._detectiontype == 'text':
                    recognitiontype = self._executetextrecognition
                else:
                    recognitiontype = self._executefacerecognition

                with aiohttp.ClientSession(loop=self._loop) as session:
                    response = self._loop.run_until_complete(
                        recognitiontype(session, image)
                    )
                self._processed_channel.put(response)
