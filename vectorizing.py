import cv2
import numpy as np
import pickle
import h5py

number_of_images = 13068 #33402   #
#dic_pick = pickle.load( open( "train_metadata.pickle", "rb" ) )
dic_pick = pickle.load( open( "test_metadata.pickle", "rb" ) )
length = [len(x) for x in dic_pick["label"]]
inpt= np.ndarray((number_of_images,784))
labels = np.ndarray((number_of_images, 7))
labels_one_hot = np.ndarray((number_of_images, 5, 10)) #(0-9 and blank)
length_one_hot = np.ndarray((number_of_images,7)) # (0-5 and more than 5)


labels_one_hot.fill(0)



labels.fill(-1)

i=0
while i<number_of_images:
	#img = cv2.imread("cropped/{}.png".format(i+1), cv2.CV_LOAD_IMAGE_GRAYSCALE)
	img = cv2.imread("cropped_test/{}.png".format(i+1), cv2.CV_LOAD_IMAGE_GRAYSCALE)
	inpt[i] = np.reshape(img, -1)
	# Iterate over all image labels and
	# creat label classes array for each image
	# creat one hot class representation which last item
	# is the blank digit
	# creat one hot class for numbers length which
	# in this experiment is limited to 5 digits

	for index,item in enumerate(dic_pick["label"][i]):
		print "i={}--index={}".format(i,index)
		if index<5:
			labels[i][index] = item
			if item == 10:
				labels_one_hot[i][index][0] = 1
			else:
				labels_one_hot[i][index][item] = 1

	digit_numbers = len(dic_pick["label"][i])
	# find length of labels and save theme one-hot coded
	if digit_numbers<=5:
		length_one_hot[i][digit_numbers] = 1
	else:
		length_one_hot[i][6] = 1

	i+=1


#print labels_one_hot.shape

char_one_lbl = np.array([labels_one_hot[i][0][:] for i in range(number_of_images-1)])
char_two_lbl = np.array([labels_one_hot[i][1][:] for i in range(number_of_images-1)])
char_three_lbl = np.array([labels_one_hot[i][2][:] for i in range(number_of_images-1)])
char_four_lbl = np.array([labels_one_hot[i][3][:] for i in range(number_of_images-1)])
char_five_lbl = np.array([labels_one_hot[i][4][:] for i in range(number_of_images-1)])

data_dict = {
	'data': inpt,
	'labels': labels,
	'length': length,
	'length_lbl': length_one_hot,
	'ch_one_lbl' : char_one_lbl,
	'ch_two_lbl' : char_two_lbl,
	'ch_three_lbl' : char_three_lbl,
	'ch_four_lbl' : char_four_lbl,
	'ch_five_lbl' : char_five_lbl
	}

#pickle_out = open("data.pickle", "wb")
pickle_out = open("test_data.pickle", "wb")
pickle.dump(data_dict, pickle_out, protocol=2)
pickle_out.close()

