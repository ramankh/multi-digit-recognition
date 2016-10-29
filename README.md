# Multi-digit Number Recognition
This project is a Multi-digit Number Recognition model which is able to recognize numbers in a photograph. To reach this goal we have implemented a convolutional neural network, and trained the model from images originaly captured from google street view. 


### Version
1.1.0

### Language
 python 2.7.12 or newer 

### Libraries

The following libraries should be installed in order to get the project up and running. 

* Tensorflow
* Matplotlib
* Numpy
* opencv

### Data Sets

You can find the preprocessed picture data sets stored in the pickles in following links:
* [Training Dataset](https://drive.google.com/open?id=0B5Rfp1TOIC8XS19rdEF4ckRHVTg)
* [Test Dataset](https://drive.google.com/open?id=0B5Rfp1TOIC8XZVNXN0tkdkw4Rlk)

### How to run
To run the project first you need to clone the project from the following link:
[Link](https://github.com/ramankh/multi-digit-recognition.git)
Then you need to put the data set files which you already have downloaded in the same directory of the project
if you want to run the model just for test purpose you can change the train flag to false, otherwise leave it to True.

You can run the project in terminal:

```python
python project.py
```

License
----

MIT
