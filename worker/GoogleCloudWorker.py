import asyncio
import json
import logging
import aiohttp
import cv2
import numpy as np
from threading import Thread
from utils import google_utils, image_utils


class GoogleCloudWorker(Thread):
    """ worker class which is started in its own thread and is responsible for handling all the requests to the Google
    vision API """

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

    @asyncio.coroutine
    def _executefacerecognition(self, session, image, max_results=4):
        """ async execution of the face_detection request """

        # convert cv2 to .jpg to make it compatible for google vision api
        image_buf = cv2.imencode('.jpg', image)[1]
        image_str = np.array(image_buf).tostring()

        request = google_utils.buildrequest(image_str, max_results, 'FACE_DETECTION')

        response = yield from session.post(self._GOOGLE_VISION_API_ENDPOINT, data=json.dumps(request))
        response_json = yield from response.json()
        self._logger.debug('response: {}'.format(response_json))

        return response_json, image

    @asyncio.coroutine
    def _executetextrecognition(self, session, image, max_results=4, threshold='otsu'):
        """ async execution of the text_detection request """

        # do image preprocessing to increase accuracy
        image_prep = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        if threshold == 'adaptive':
            image_prep = image_utils.adaptivethreshold(image_prep)

        else:
            image_prep = image_utils.otsuthreshold(image_prep)

        # convert the image to a jpg string
        image_str = image_utils.convertimagetojpgstring(image_prep)

        # build the request
        request = google_utils.buildrequest(image_str, max_results, 'TEXT_DETECTION')

        response = yield from session.post(self._GOOGLE_VISION_API_ENDPOINT, data=json.dumps(request))
        response_json = yield from response.json()
        self._logger.debug('response: {}'.format(response_json))

        return response_json, image

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
