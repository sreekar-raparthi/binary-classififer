# binary-classififer
A simple classifier to differentiate bikers with and with out helmet
Create a virtual environment using command 
conda create - n {envname} python = {version}

Activate virtual environment with command conda activate {envname}

Install the libraries scikit-learn, pandas, Tensorflow, keras, h5py, scikit-image, matplotlib
conda install {library names comma seperated}

Run train.py file using 
python train.py

Check the training process using tensorboard to check for overfitting or underfitting

When you reach the maximum accuracy stop the training. A .h5 file is generated which is the weight file.

Use the weight file in the inference.py to test the predictions of real time data


