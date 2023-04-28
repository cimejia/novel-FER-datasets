import cv2
from model import FacialExpressionModel
import numpy as np

facec = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
model = FacialExpressionModel('models/FER2013/new_fer_best_model.h5')
#model = FacialExpressionModel('models/NHFI/new_best_model_tl.h5')
#model = FacialExpressionModel('models/AFFECTNET/best_model_fs.h5')
#model = FacialExpressionModel('models/MIXED/best_model_mixed.h5')
#model = FacialExpressionModel('vit_large_patch16_48.h5')

font = cv2.FONT_HERSHEY_SIMPLEX

class VideoCamera(object):
    def __init__(self):
        self.video = cv2.VideoCapture('videos/videoplayback.mp4')
        #self.video = cv2.VideoCapture('videos/video.mp4')

    def __del__(self):
        self.video.release()

    # returns camera frames along with bounding boxes and predictions
    def get_frame(self):
        _, fr = self.video.read()
        gray_fr = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
        faces = facec.detectMultiScale(gray_fr, 1.3, 5)
        #faces = facec.detectMultiScale(gray_fr, scaleFactor=1.1, minNeighbors=5, minSize=(30,30), maxSize=(300,300))

        for (x, y, w, h) in faces:
            fc = gray_fr[y:y+h, x:x+w]
            #fc = fr[y:y+h, x:x+w]
            #FER2013
            roi = cv2.resize(fc, (48, 48))
            pred = model.predict_emotion(roi[np.newaxis, :, :, np.newaxis])
            #AFFECTNET
            #roi = cv2.resize(fc, (48, 48))
            #roi = cv2.resize(fc, (224, 224))
            #pred = model.predict_emotion(roi[np.newaxis, :, :, 3])

            if pred == "Angry" or pred == "Disgust":
                cv2.putText(fr, pred, (x, y), font, 1, (0, 0, 255), 2)
                cv2.rectangle(fr,(x,y),(x+w,y+h),(0, 0, 255),1)
            elif pred == "Happy":
                cv2.putText(fr, pred, (x, y), font, 1, (0, 255, 0), 2)
                cv2.rectangle(fr,(x,y),(x+w,y+h),(0, 255, 0),1)
            else:
                cv2.putText(fr, pred, (x, y), font, 1, (255, 255, 0), 2)
                cv2.rectangle(fr,(x,y),(x+w,y+h),(255, 255, 0),1)

        _, jpeg = cv2.imencode('.jpg', fr)
        return jpeg.tobytes()