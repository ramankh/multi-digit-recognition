import numpy as np
#import h5py
import random
import pickle

class GetData:

	def __init__(self, use_hdf5=False, path = "", test_size=500):
		self.use_hdf5 = use_hdf5
		self.path = path
		self.test_size = test_size
		self.loadData()

	def loadData(self):
		if self.use_hdf5 == True:
			self.f_in = h5py.File(self.path, "r")
		else:
			self.open_file = open(self.path,'rb')
			self.f_in = pickle.load(self.open_file)

		self.open_file.close()
		raw_data = [np.array(self.f_in["data"]),\
					np.array(self.f_in["ch_one_lbl"]),\
					np.array(self.f_in["ch_two_lbl"]),\
					np.array(self.f_in["ch_three_lbl"]),\
					np.array(self.f_in["ch_four_lbl"]),\
					np.array(self.f_in["ch_five_lbl"]),\
					np.array(self.f_in["length_lbl"])]

		indices = random.sample(range(len(self.f_in["data"])-1),self.test_size)

		self.x = np.delete(raw_data[0], indices,0)
		self.y_one = np.delete(raw_data[1], indices,0)
		self.y_two = np.delete(raw_data[2], indices,0)
		self.y_three = np.delete(raw_data[3], indices,0)
		self.y_four = np.delete(raw_data[4], indices,0)
		self.y_five = np.delete(raw_data[5], indices,0)
		self.y_len  = np.delete(raw_data[6], indices,0)

		self.test_data = [np.array([raw_data[0][i] for i in indices]),\
			   np.array([np.argmax(raw_data[1][i]) for i in indices]),\
			   np.array([np.argmax(raw_data[2][i]) for i in indices]),\
			   np.array([np.argmax(raw_data[3][i]) for i in indices]),\
			   np.array([np.argmax(raw_data[4][i]) for i in indices]),\
			   np.array([np.argmax(raw_data[5][i]) for i in indices]),\
			   np.array([np.argmax(raw_data[6][i]) for i in indices])]

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
		return train_data, valid_data



	def getOne(self):
		return np.array([self.x[0]]),\
			   np.array([np.argmax(self.y_one[0])]),\
			   np.array([np.argmax(self.y_two[0])]),\
			   np.array([np.argmax(self.y_three[0])]),\
			   np.array([np.argmax(self.y_four[0])]),\
			   np.array([np.argmax(self.y_five[0])]),\
			   np.array([np.argmax(self.y_len[0])])

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




