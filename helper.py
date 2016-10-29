import numpy as np
#import h5py
import random
import pickle

class GetData:

	def __init__(self, use_hdf5=False, path = ""):
		self.use_hdf5 = use_hdf5
		self.path = path
		self.loadData()

	def loadData(self):
		if self.use_hdf5 == True:
			self.f_in = h5py.File(self.path, "r")
		else:
			self.open_file = open(self.path,'rb')
			self.f_in = pickle.load(self.open_file)

		self.open_file.close()

		self.x = np.array(self.f_in["data"])
		self.y_one = np.array(self.f_in["ch_one_lbl"])
		self.y_two = np.array(self.f_in["ch_two_lbl"])
		self.y_three = np.array(self.f_in["ch_three_lbl"])
		self.y_four = np.array(self.f_in["ch_four_lbl"])
		self.y_five = np.array(self.f_in["ch_five_lbl"])
		self.y_len  = np.array(self.f_in["length_lbl"])

	def returnAll(self):
		return [self.x,self.y_one,self.y_two,self.y_three,self.y_four,self.y_five,self.y_len]

	def nextBatch(self,number=10):
		indices = random.sample(range(len(self.x)-1),number*2)
		train_indices = indices[0:number]
		valid_indices = indices[number:number*2]

		train_data = [np.array([self.x[i] for i in train_indices]),\
			   np.array([np.argmax(self.y_one[i]) for i in train_indices]),\
			   np.array([np.argmax(self.y_two[i]) for i in train_indices]),\
			   np.array([np.argmax(self.y_three[i]) for i in train_indices]),\
			   np.array([np.argmax(self.y_four[i]) for i in train_indices]),\
			   np.array([np.argmax(self.y_five[i]) for i in train_indices]),\
			   np.array([np.argmax(self.y_len[i]) for i in train_indices])]
		valid_data = [np.array([self.x[i] for i in valid_indices]),\
			   np.array([np.argmax(self.y_one[i]) for i in valid_indices]),\
			   np.array([np.argmax(self.y_two[i]) for i in valid_indices]),\
			   np.array([np.argmax(self.y_three[i]) for i in valid_indices]),\
			   np.array([np.argmax(self.y_four[i]) for i in valid_indices]),\
			   np.array([np.argmax(self.y_five[i]) for i in valid_indices]),\
			   np.array([np.argmax(self.y_len[i]) for i in valid_indices])]

		return train_data, valid_data

	def nextTestBatch(self,number=10):
		indices = random.sample(range(len(self.x)-1),number)

		test_data = [np.array([self.x[i] for i in indices]),\
			   np.array([np.argmax(self.y_one[i]) for i in indices]),\
			   np.array([np.argmax(self.y_two[i]) for i in indices]),\
			   np.array([np.argmax(self.y_three[i]) for i in indices]),\
			   np.array([np.argmax(self.y_four[i]) for i in indices]),\
			   np.array([np.argmax(self.y_five[i]) for i in indices]),\
			   np.array([np.argmax(self.y_len[i]) for i in indices])]
		return test_data

	def getOne(self):
		index = random.sample(range(len(self.x)-1),1)
		ind = index[0]
		ind = 0
		return [np.array([self.x[ind]]),\
			   np.array([np.argmax(self.y_one[ind])]),\
			   np.array([np.argmax(self.y_two[ind])]),\
			   np.array([np.argmax(self.y_three[ind])]),\
			   np.array([np.argmax(self.y_four[ind])]),\
			   np.array([np.argmax(self.y_five[ind])]),\
			   np.array([np.argmax(self.y_len[ind])])]

	def getHun(self,number=1000, batch_size=100):
		all_indices = range(number)

		batch_indices = random.sample(all_indices,batch_size*2)
		train_indices = batch_indices[0:batch_size]
		valid_indices = batch_indices[batch_size:]
		train_data = [np.array([self.x[i] for i in train_indices]),\
			   np.array([np.argmax(self.y_one[i]) for i in train_indices]),\
			   np.array([np.argmax(self.y_two[i]) for i in train_indices]),\
			   np.array([np.argmax(self.y_three[i]) for i in train_indices]),\
			   np.array([np.argmax(self.y_four[i]) for i in train_indices]),\
			   np.array([np.argmax(self.y_five[i]) for i in train_indices]),\
			   np.array([np.argmax(self.y_len[i]) for i in train_indices])]
		valid_data = [np.array([self.x[i] for i in valid_indices]),\
			   np.array([np.argmax(self.y_one[i]) for i in valid_indices]),\
			   np.array([np.argmax(self.y_two[i]) for i in valid_indices]),\
			   np.array([np.argmax(self.y_three[i]) for i in valid_indices]),\
			   np.array([np.argmax(self.y_four[i]) for i in valid_indices]),\
			   np.array([np.argmax(self.y_five[i]) for i in valid_indices]),\
			   np.array([np.argmax(self.y_len[i]) for i in valid_indices])]

		return train_data, valid_data




