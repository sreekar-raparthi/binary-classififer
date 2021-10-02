from keras.preprocessing import sequence
import numpy as np
import h5py
import json
import os
import random
import matplotlib.image as mpimg
from skimage.transform import resize

class Generator():

    def __init__(self,image_dir = "D:/Data/",label_json = "D:/triple_class.json",batch_size = '32'):
        self.image_dir = image_dir
        with open (label_json, 'r') as fp:
            self.filename_to_label = json.load(fp)
        self.total_samples = len(self.filename_to_label)

        filenames = list(self.filename_to_label.keys())
        random.shuffle(filenames)
        self.randomfilenames = filenames
 
        self.batch_size = batch_size

    def load_img(self, filename):
        if not os.path.isfile(filename):
            print (filename + "does not exist")
            dummy = np.zeros((75,75,1))
            return dummy
        else:
            img = mpimg.imread(filename)
            if len(img.shape) == 3 and img.shape[2] == 3:
                img = np.dot(img[...,:3], [0.2989, 0.5870, 0.1140]) # Convert to greyscale
            im = resize(img, (75,75), order=1,mode='constant',preserve_range=True)
            im = np.array(im,dtype=np.float32) # convert to single precision
            img = (im - np.mean(im)) / ( (np.std(im) + 0.0001) )
            return img.reshape((75,75,1))

    def flow(self):
        images = []
        labels = []

        total_count = 0

        while True:
            for filename in self.randomfilenames:
                current_img = self.load_img(self.image_dir + filename)
                label = self.filename_to_label[filename]
                images.append(current_img)

                if (label == "No_Helmet"):
                    label = [0,0,1]
                    labels.append(label)


                if (label == "With_Helmet"):
                    label = [1,0,0]
                    labels.append(label)

                if (label == "Occluded"):
                    label = [0,1,0]
                    labels.append(label)

                

                
                total_count += 1

                    
                if total_count>=self.batch_size:
                    images = np.asarray(images)
                    labels = np.asarray(labels)
                    yield [images,labels]
                        
                    total_count = 0
                    images = []
                    labels = []

            ''' images = np.asarray(images)
            labels = np.asarray(labels)
            return [images,labels]'''

                