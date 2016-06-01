import asyncio
import json
import logging
import aiohttp
import cv2
import numpy as np
import base64
from threading import Thread, current_thread
from aiohttp import web
from aiohttp.web import Application


class RestEndpointWorker(Thread):

    def __init__(self, port, address, raw_channel, processed_channel):
        super(RestEndpointWorker, self).__init__()
        self._logger = logging.getLogger(__name__)

        self._port = port
        self._address = address

        self._raw_channel = raw_channel
        self._processed_channel = processed_channel

        self._loop = None
        self._tid = None
        self._srv = None
        self._handler = None
        self._app = None
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

    def _imageprep(self, image, max_results=4, threshold='otsu'):

        image_array = np.asarray(bytearray(image), dtype="uint8")
        image_prep = cv2.imdecode(image_array, cv2.IMREAD_GRAYSCALE)

        # image to show the detected artifacts on later in the main thread
        image_show = cv2.imdecode(image_array, cv2.IMREAD_ANYCOLOR)

        # apply some ocr realted optimization on the picture
        if threshold == 'adaptive':
            # adaptive threshold
            #image_prep = cv2.medianBlur(image_prep, 5)
            image_prep = cv2.adaptiveThreshold(image_prep, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

        else:
            # otsu
            #image_prep = cv2.GaussianBlur(image_prep, (5, 5), 0)
            image_prep = cv2.threshold(image_prep, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

        # convert cv2 to .jpg to make it compatible for google vision api
        image_buf = cv2.imencode('.jpg', image_prep)[1]
        image_str = np.array(image_buf).tostring()

        # build request with preprocessed image
        request = self._buildrequest(image_str, max_results, 'TEXT_DETECTION')

        return image_show, request

    @asyncio.coroutine
    def _executetextrecognition(self, session, request, image):
        response = yield from session.post(self._GOOGLE_VISION_API_ENDPOINT, data=json.dumps(request))
        response_json = yield from response.json()
        self._logger.debug('response: {}'.format(response_json))

        return response_json, image

    @asyncio.coroutine
    def _detecttext(self, request):
        ''' execute the object recognition in a different thread to keep the app responsive '''

        data = yield from request.post()

        input_image = data['img'].file
        image = input_image.read()

        # run the picture optimization in a different thread
        future = self._loop.run_in_executor(None, self._imageprep, image)
        image, request = yield from future

        with aiohttp.ClientSession(loop=self._loop) as session:
            # response = (response_json, image)
            response_json, image = yield from self._executetextrecognition(session, request, image)

        # just add a picture to the queue if faces have been recognized, filter for non empty json arrays
        self._processed_channel.put((response_json, image))

        if bool(response_json['responses'][0]):
            recognized_text = response_json['responses'][0]['textAnnotations'][0]
        else:
            recognized_text = {}

        return web.Response(text=json.dumps(recognized_text),
                            status=200,
                            content_type='application/json'
                            )

    @asyncio.coroutine
    def init(self, loop):
        app = Application(loop=loop)

        app.router.add_route('POST', '/detecttext', self._detecttext)

        handler = app.make_handler()
        srv = yield from loop.create_server(handler, self._address, self._port)
        self._logger.info("Server started on host {} / port {}".format(self._address, self._port))
        return srv, handler

    def run(self):
        print("Rest Eventloop Server is being started")

        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        self._tid = current_thread()
        self._srv, self._handler = self._loop.run_until_complete(self.init(self._loop))
        self._loop.run_forever()

    def stop(self):
        self._srv.close()
        self._loop.run_until_complete(self._srv.wait_closed())
        self._loop.run_until_complete(self._app.shutdown())
        self._loop.run_until_complete(self._handler.finish_connections(60.0))
        self._loop.run_until_complete(self._app.cleanup())
        self._loop.stop()
        self._loop.run_forever()
        self._loop.close()
