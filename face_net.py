#import necessary libraries
import os 
import cv2
from PIL import Image
import requests
import os
import matplotlib.pyplot as plt
import dlib
import pickle 
import numpy as np
from architecture import * 
from sklearn.preprocessing import Normalizer
from tensorflow.keras.models import load_model


######pathsandvairables#########
face_data = 'Faces/'
required_shape = (160,160)
face_encoder = InceptionResNetV2()
path = "Project_Extra/facenet_keras_weights.h5"
face_encoder.load_weights(path)
hog_face_detector = dlib.get_frontal_face_detector()
encodes = []
encoding_dict = dict()
l2_normalizer = Normalizer('l2')
###############################


def normalize(img):
    mean, std = img.mean(), img.std()
    return (img - mean) / std


for face_names in os.listdir(face_data):
    person_dir = os.path.join(face_data,face_names)

    for image_name in os.listdir(person_dir):
        image_path = os.path.join(person_dir,image_name)

        img_BGR = cv2.imread(image_path)
        img_RGB = cv2.cvtColor(img_BGR, cv2.COLOR_BGR2RGB)

        faces_hog = hog_face_detector(img_RGB, 1)
        for face in faces_hog:
            x = face.left()
            y = face.top()
            w = face.right() - x
            h = face.bottom() - y
            x,y = abs(x), abs(y)
            face = img_RGB[y:y+h, x:x+w]
        face = normalize(face)
        face = cv2.resize(face, required_shape)
        face_d = np.expand_dims(face, axis=0)
        encode = face_encoder.predict(face_d)[0]
        encodes.append(encode)

    if encodes:
        encode = np.sum(encodes, axis=0 )
        encode = l2_normalizer.transform(np.expand_dims(encode, axis=0))[0]
        encoding_dict[face_names] = encode
    
path = 'encodings/encodings.pkl'
with open(path, 'wb') as file:
    pickle.dump(encoding_dict, file)


print("[INFO] : Encodings Generated Successfully")



