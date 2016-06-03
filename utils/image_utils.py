import cv2
import numpy as np


def convertimagetoopencvarray(image):
    """ convert image from jpg to and numpy array to be able to apply the opencv methods on the image """

    image_array = np.asarray(bytearray(image), dtype="uint8")
    image_prep = cv2.imdecode(image_array, cv2.IMREAD_GRAYSCALE)

    # image to show the detected artifacts on later in the main thread
    image_show = cv2.imdecode(image_array, cv2.IMREAD_ANYCOLOR)

    return image_prep, image_show


def convertimagetojpgstring(image):
    """ opencv arrays have to be converted into an jpg string to be compatible to Google Vison API """

    image_buf = cv2.imencode('.jpg', image)[1]
    image_str = np.array(image_buf).tostring()

    return image_str


def adaptivethreshold(image):
    """ apply the adaptive threshold to the grayscale image to optimize for OCR """

    image_prep = cv2.medianBlur(image, 5)
    image_prep = cv2.adaptiveThreshold(image_prep, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

    return image_prep


def otsuthreshold(image):
    """ apply the OTSU threshold to the grayscale image optimize for OCR """

    image_prep = cv2.GaussianBlur(image, (5, 5), 0)
    image_prep = cv2.threshold(image_prep, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

    return image_prep


def getemotion(face):
        """ filter return for possible to very_likely emotions """

        indicators = ['POSSIBLE', 'LIKELY', 'VERY_LIKELY']
        google_face_emotions = ['joyLikelihood', 'sorrowLikelihood', 'angerLikelihood', 'surpriseLikelihood',
                    'underExposedLikelihood', 'blurredLikelihood', 'headwearLikelihood']

        # detected emotions ordered by likelihood
        detected_emotions = {'VERY_LIKELY': [], 'LIKELY': [], 'POSSIBLE': []}

        for emotion in google_face_emotions:
            if face[emotion] in indicators:
                detected_emotions[face[emotion]] += [emotion]

        return detected_emotions


def drawmostlikelyemotion(faces, image):
    """ draw a simple emoticon """

    for face in faces:
        x = face['fdBoundingPoly']['vertices'][0].get('x', 0.0)
        y = face['fdBoundingPoly']['vertices'][0].get('y', 0.0)
        w = face['fdBoundingPoly']['vertices'][1].get('x', 0.0) - x
        h = face['fdBoundingPoly']['vertices'][2].get('y', 0.0) - y

        x_center = int(x + w / 2)
        y_center = int(y + h / 2)
        y_bottom = y + h

        cv2.circle(image, (x_center, y_center), int(w / 2), (0, 255, 0), 2)

        # place mood under the circle
        detected_emotions = getemotion(face)
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


def drawtextrectangle(json_response, image):
    """ draw a simple rectangle around the text """

    texts = json_response['responses'][0]['textAnnotations']

    for text in texts:
        box = [(v.get('x', 0.0), v.get('y', 0.0)) for v in text['boundingPoly']['vertices']]
        cv2.rectangle(image, box[0], box[2], (0, 255, 0), 2)

    return image
