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
from tkinter.filedialog import askopenfilename
from scipy.spatial.distance import cosine

hog_face_detector = dlib.get_frontal_face_detector()
face_encoder = InceptionResNetV2()
path_m ='Project_Extra/facenet_keras_weights.h5'
face_encoder.load_weights(path_m)
encoder=face_encoder
recognition_t=0.8
encodings_path ='encodings/encodings.pkl'
required_size = (160,160)
l2_normalizer = Normalizer('l2')

def normalize(img):
    mean, std = img.mean(), img.std()
    return (img - mean) / std

def get_encode(face_encoder, face, size):
	face = normalize(face)
	face = cv2.resize(face, size)
	encode = face_encoder.predict(np.expand_dims(face, axis=0))[0]
	return encode


def load_pickle(path):
	with open(path, 'rb') as f:
		encoding_dict = pickle.load(f)
	return encoding_dict

encoding_dict = load_pickle(encodings_path)


def perform_recognition(path):

	frame =cv2.imread(path)
	faces_hog = hog_face_detector(frame, 1)
	for face in faces_hog:
		x = face.left()
		y = face.top()
		w = face.right() - x
		h = face.bottom() - y
		x,y = abs(x), abs(y)
		face = frame[y:y+h, x:x+w]
		x1=x+w
		y1=y+h
		encode = get_encode(encoder, face, required_size)
		encode = l2_normalizer.transform(encode.reshape(1, -1))[0]
		name = 'unknown'
		pt_1=(x,y)
		pt_2=(x1,y1)
		distance = float("inf")
		for db_name, db_encode in encoding_dict.items():
			dist = cosine(db_encode, encode)

			if dist < recognition_t and dist < distance:
				name = db_name
				distance = dist
		# print("Cosine Similarity : ",dist)
		if name == 'unknown':
			return name
		else:
			return name


def main():
	path=askopenfilename()
	get_result=perform_recognition(path)
	print("\nResult")
	print(get_result)


# main()