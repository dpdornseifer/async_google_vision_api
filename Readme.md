# Async Google Cloud Vision API 

The purpose of this app is to visualize the results of the [Google Cloud Vision API](https://cloud.google.com/vision) 
whether it is text or face detection. 
The application is capable of taking pictures of a local webcam device via `opencv3` or via a REST interface.
The application is using Python 3.4 and its coroutines to be non-blocking. 
Beside that, the image display based on `cv2.imshow('image', processed_frame)` is running in the main thread while 
the Google Cloud Worker or the rest interface are running in their own threads. 

Basically the face / text detection backend can be replaced by the [Microsoft Computer Vision API](https://www.microsoft.com/cognitive-services/en-us/computer-vision-api)
The only change would have to be made while building the request for the services. 


## How to run it
1. Create a virtualenv with Python 3.4
2. Clone the repo
3. Install the requirements via `pip3`
4. Make sure that `OpenCV3` is installed and the Python binding are available 
You should have a look [here](https://github.com/dpdornseifer/async_face_recognition#install-the-opencv3-dependencies) 
where I explain/give some hints how you can install OpenCV3 on Mac.
4. Put in the Google API key for the service you want to use e.g. `self._API_KEY = ''` for the worker/GoogleCloudWorker 
file
5. Start one of the following services and have fun :) 


## Face Detection 
Face detection is using the local webcam `cv2.VideoCapture(0)` 
`imageservice = ImageRecognitionService(restinterface='false', detectiontype='face')`

As soon as there has been an hit, an new OpenCV window will be opened and the detected face(s) will be shown with 
the most likely mood. 

## Text Detection 

### Image Pre-Processing
In general it's very important to prepare the images where you want to apply the OCR algorithms on (e.g. look at 
[Image Preprossesing for license plate detection](http://stackoverflow.com/a/28936254/2054009) 
or [OpenCV Image Thresholding](http://docs.opencv.org/3.1.0/d7/d4d/tutorial_py_thresholding.html#gsc.tab=0)). 


```
def adaptivethreshold(image):
    image_prep = cv2.medianBlur(image, 5)
    image_prep = cv2.adaptiveThreshold(image_prep, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
return image
```
    
```
def otsuthreshold(image):
    image_prep = cv2.GaussianBlur(image, (5, 5), 0)
    image_prep = cv2.threshold(image_prep, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
return image
```


In both cases the `adaptivethrshold` or the `otsuthreshold` and or blur filters can be fine-tuned. If you want to see how
the artifacts look like on the filtered image you can send the prepared picture `image_prep` into the `self._processed_channel`. 

When using the rest API these image preparations are executed in a single thread via `loop.run_in_executor()` as 
you can see in the code.

```
future = self._loop.run_in_executor(None, self._imageprep, image)
         image, request = yield from future
```


### Webcam
To use the local webcam for text detection or API testing define the service like shown here:
`imageservice = ImageRecognitionService(restinterface='false', detectiontype='text')`. 
The framerate can be adjusted by adjusting the `sleep()` time in `_getframe()`. If there are different 
camera devices available you can select the right own by changing the `cv2.VideoCapture(0)` integer parameter.

As soon as the API has found some text on picture, an OpenCV window will be opened and the text regions will 
be marked with a green frame.




### Rest Interface
To use the rest interface define the service with the `restinterface=true` flag like shown here: 
`imageservice = ImageRecognitionService(restinterface='true', detectiontype='text')`

The image should be posted as a form parameter (`img`) to `0.0.0.0:8080/detecttext` by default.
If a text has been detected, the overall description and the overall coordinates, basically the first detection 
result of the Google API, is sent back. 

```
{
    "locale": "en",
    "description": "11\nGood mornin\nid.\nWhat is your main focus for today?\n",
    "boundingPoly": {
        "vertices": [
            {
                "x": 73,
                "y": 61
            },
            {
                "x": 622,
                "y": 61
            },
            {
                "x": 622,
                "y": 618
            },
            {
                "x": 73,
                "y": 618
            }
        ]
    }
}
```

Also if there is a hit, the posted picture will be shown in an new OpenCV window with the detected text areas highlighted.
 
