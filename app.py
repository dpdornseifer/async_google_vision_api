from ImageRecognitionService import ImageRecognitionService

if __name__ == '__main__':
    imageservice = ImageRecognitionService(restinterface='false', detectiontype='face')
    imageservice.run()
