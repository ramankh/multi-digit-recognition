import cv2
import numpy as np
import pickle
import h5py

number_of_images = 13068 #33402
#dic_pick = pickle.load( open( "train_metadata.pickle", "rb" ) )
dic_pick = pickle.load( open( "test_metadata.pickle", "rb" ) )
length = [len(x) for x in dic_pick["label"]]
inpt= np.ndarray((number_of_images,784))
labels = np.ndarray((number_of_images, 7))
labels_one_hot = np.ndarray((number_of_images, 5, 11)) #(0-9 and blank)
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
			labels_one_hot[i][index][item] = 1
	digit_numbers = len(dic_pick["label"][i])
	if digit_numbers<=5:
		length_one_hot[i][digit_numbers] = 1
	else:
		length_one_hot[i][6] = 1
	while (5-digit_numbers)>0:
		labels_one_hot[i][digit_numbers][10]=1
		digit_numbers+=1

	i+=1


#print labels_one_hot.shape

char_one_lbl = np.array([labels_one_hot[i][0][:] for i in range(number_of_images-1)])
char_two_lbl = np.array([labels_one_hot[i][1][:] for i in range(number_of_images-1)])
char_three_lbl = np.array([labels_one_hot[i][2][:] for i in range(number_of_images-1)])
char_four_lbl = np.array([labels_one_hot[i][3][:] for i in range(number_of_images-1)])
char_five_lbl = np.array([labels_one_hot[i][4][:] for i in range(number_of_images-1)])

#print char_one_lbl.shape
#print char_one_lbl[61]

#cv2.imwrite("Temp/{}.png".format(1), img)
'''
f = h5py.File("mydata.hdf5", "w")

f["data"] = inpt
f["labels"] = labels
f["length"] = length
f["length_lbl"] = length_one_hot
f["ch_one_lbl"] = char_one_lbl
f["ch_two_lbl"] = char_two_lbl
f["ch_three_lbl"] = char_three_lbl
f["ch_four_lbl"] = char_four_lbl
f["ch_five_lbl"] = char_five_lbl

f.close()

f_in = h5py.File("mydata.hdf5", "r")
arr = f_in["data"]
labs = f_in["labels"]
arr2 = np.array(labs)
arr3 = np.array(f_in["length"])
f_in.close()
'''


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

