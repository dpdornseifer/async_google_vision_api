import base64


def buildrequest(image_str, max_results, detection_type):
    ''' returns the Google Vision API request '''

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
