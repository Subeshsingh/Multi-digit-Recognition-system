import pefile
import os
import array
import math
import pickle
from sklearn.externals import joblib
import sys
import argparse
import os, sys, shutil, time
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import keras
from scipy import misc
from keras.utils import np_utils
import graphics
import os
import cv2

image_size = (54,128)
max_digits = 7


from flask import Flask, request, jsonify, render_template,abort,redirect,url_for
from werkzeug import secure_filename
from sklearn.externals import joblib






app=Flask(__name__)


APP_ROOT=os.path.dirname(os.path.abspath(__file__))
@app.route('/')
def index():
    return render_template('index.html')





@app.route('/uploader', methods = ['GET', 'POST'])
def upload_file():
   if request.method == 'POST':
    f= request.files['file']
    file_name_input=f.filename
    target=os.path.join(APP_ROOT,'static/')
    # print(os.path)
    # path='../inputs/custom/'+file_name_input
    # path1='../../inputs/custom/'+file_name_input
    # f.save(path)
    if(not os.path.isdir(target)):
    	os.mkdir(target)
    else:
    	print("directory not created")
    path3="/".join([target,file_name_input])
    f.save(path3)
    # print('jdfjsd')
    ans=Predict(path3)
    # print('hidshf')
    # print(ans)
    # print(type(ans))
    # ansstr=''.join(str(x) for x in ans)
    ansstr=''
    for x in ans:
    	ansstr=ansstr+str(x)
    print(ansstr)
    return render_template('result.html',seq=ansstr,image_name=file_name_input)



def loadImages(filename):
    Ximg = []
    # for filename in os.listdir(path):
    if filename.endswith('png') or filename.endswith('jpg'):
        # rawimage = misc.imread(path+filename)
        rawimage = misc.imread(filename)
        img = misc.imresize(rawimage, size=image_size, interp='bilinear')
        Ximg.append(img)
    return np.array(Ximg)

def standardize(img):
    s = img - np.mean(img, axis=(2,0,1), keepdims=True)
    s /= np.std(s, axis=(2,0,1), keepdims=True)
    return s

def Predict(path3):
	model_yaml = open('../checkpoints/model.yaml','r')
	model = keras.models.model_from_yaml(model_yaml.read())
	model_yaml.close()
	model.load_weights('../checkpoints/model.hdf5')

	vision = model.layers[1]
	counter = model.layers[3]
	detector = model.layers[4]
	print('r@hul')
	# Ximg = loadImages('../inputs/custom/')
	Ximg = loadImages(path3)
	print('cool')
	Xs = np.array([standardize(x) for x in Ximg])

	h = vision.predict(Xs)
	ycount = counter.predict(h)
	ycount = np.argmax(ycount, axis=1)
	

	ylabel = []
	for i in range(len(ycount)):
	    # generate range for each count
	    indices = np.arange(ycount[i])
	    # one hot encoding for each index
	    indices = np_utils.to_categorical(indices, max_digits)
	    # tile h to match shape of indices matrix
	    hs = np.tile(h[i], (ycount[i],1))
	    print("djflks ",hs)
	    print(indices)
	    # predict labels for the sample
	    sample_seq = detector.predict([hs, indices])
	    print(sample_seq)
	    sample_seq = np.argmax(sample_seq,1)

	    ylabel.append(sample_seq)

	# plt.figure(figsize=(12,3))
	ans=list()
	for i in range(len(Ximg)):
	    # plt.subplot(5,4,i+1)
	    # plt.imshow(Ximg[i])
	    # plt.axis('off')
	    # plt.title("{}".format(ylabel[i]))
	    print(ylabel[i])
	    ans.append(ylabel[i])
	return ans




if __name__ == '__main__':
    app.run(debug=True)