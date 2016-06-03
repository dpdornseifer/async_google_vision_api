import asyncio
import json
import logging
import aiohttp
from threading import Thread
from threading import current_thread
from aiohttp import web
from utils import google_utils
from utils import image_utils


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
    def _imageprep(image, max_results=4, threshold='otsu'):
        """ applies the given optimization method to the raw image """

        image_prep, image_show = image_utils.convertimagetoopencvarray(image)

        if threshold == 'adaptive':
            image_prep = image_utils.adaptivethreshold(image_prep)

        else:
            image_prep = image_utils.otsuthreshold(image_prep)

        image_str = image_utils.convertimagetojpgstring(image_prep)
        request = google_utils.buildrequest(image_str, max_results, 'TEXT_DETECTION')

        return image_show, request

    @asyncio.coroutine
    def _executetextrecognition(self, session, request, image):
        """ execute the post request to the Google Cloud Vision API in an async way """

        response = yield from session.post(self._GOOGLE_VISION_API_ENDPOINT, data=json.dumps(request))
        response_json = yield from response.json()
        self._logger.debug('response: {}'.format(response_json))

        return response_json, image

    @asyncio.coroutine
    def _detecttext(self, request):
        """ does the image preparation and finally executes the post to the ML backend """

        # get the image from the post request
        data = yield from request.post()
        input_image = data['img'].file
        image = input_image.read()

        # run the picture optimization in a different thread
        future = self._loop.run_in_executor(None, self._imageprep, image)
        image, request = yield from future

        # execute the post request in its own async context
        with aiohttp.ClientSession(loop=self._loop) as session:
            response_json, image = yield from self._executetextrecognition(session, request, image)

        # just add a picture to the queue if faces have been recognized, filter for non empty json arrays
        self._processed_channel.put((response_json, image))

        # process the response, sent and empty json object if no text has been detected by the Vision API
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
        app = web.Application(loop=loop)

        # the only route available /detectext
        app.router.add_route('POST', '/detecttext', self._detecttext)

        handler = app.make_handler()
        srv = yield from loop.create_server(handler, self._address, self._port)
        self._logger.info("Server started on host {} / port {}".format(self._address, self._port))
        return srv, handler

    def run(self):
        """ run the eventloop in a different thread than the main loop which is used for OpenCV methods """

        print("Rest Eventloop Server is being started")
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        self._tid = current_thread()
        self._srv, self._handler = self._loop.run_until_complete(self.init(self._loop))
        self._loop.run_forever()

    def stop(self):
        """ shutdown the eventloop and the connections in the right way"""

        self._srv.close()
        self._loop.run_until_complete(self._srv.wait_closed())
        self._loop.run_until_complete(self._app.shutdown())
        self._loop.run_until_complete(self._handler.finish_connections(60.0))
        self._loop.run_until_complete(self._app.cleanup())
        self._loop.stop()
        self._loop.run_forever()
        self._loop.close()
