from model import Model
from generator import Generator
import numpy as np
import math
import os
import shutil 
from keras.models import load_model
import matplotlib.image as mpimg
from skimage.transform import resize
import json

width, height = 75, 75
weight_path = "D:/model_best_classifier.h5"

model = Model(width,height)
model = model.build_model()
model.load_weights(weight_path)
model.summary()
print(len(model.layers))

count = 0
total = 0
filename_to_label = {}
label_json = "D:/New_Model_data/test_New_Data.json"
image_dir = "D:/New_Model_data/New_Data/"
wrong_dir = "D:/Wrong_Classified1/"
wrong_names = []
wrong_label = []
with open (label_json, 'r') as fp:
    filename_to_label = json.load(fp)

for filename, label in filename_to_label.items():
    #print(filename)
    img = mpimg.imread(os.path.join(image_dir, filename))
    total = total+1
    if len(img.shape) == 3 and img.shape[2] == 3:
        img = np.dot(img[...,:3], [0.2989, 0.5870, 0.1140]) # Convert to greyscale
    im = resize(img, (75,75), order=1,mode='constant',preserve_range=True)
    im = np.array(im,dtype=np.float32) # convert to single precision
    img = (im - np.mean(im)) / ( (np.std(im) + 0.0001) )
    img = img.reshape((75,75,1))
    img = np.expand_dims(img, axis=0)

    prediction = model.predict(img)
    print(prediction, ' ', label)
    human_readable = prediction[0]
    print(max(human_readable))
    m = np.argmax(human_readable)
    print(m)
    if(m == 2 and label == "No_Helmet"):
        count = count+1
    elif(m == 1 and label == "Occluded"):
        count = count+1
    elif(m == 0 and label == "With_Helmet"):
        count = count+1
    else:
        path = os.path.join(wrong_dir,str(m))
        wrong_names.append(filename)
        wrong_label.append(m)
        shutil.copy(os.path.join(image_dir,filename),path)
            
    print('')
    print("number of test_images:", total)
    print('')
    print("number of correct predictions:",count)
    print('')
    z = dict(zip(wrong_names,wrong_label))
    print(z)

    
    

