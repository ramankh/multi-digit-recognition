
from helper import GetData
import numpy as np
import matplotlib as mp
import matplotlib.pyplot as plt


data = GetData(use_hdf5=False, path="data.pickle")
test_data = GetData(use_hdf5=False, path="test_data.pickle")

data = test_data.returnAll()
l1 = data[1] #label numbr 2
print len(l1)
print l1[0]
#arr = [x for x in l1 if x[10]==1]
#print len(arr)
print "OOOOOOOOOOOOOOOOOOO"

ind = 13066
print data[1][ind]
print data[2][ind]
print data[3][ind]
print data[4][ind]
print data[5][ind]
print "_____________________"
combine = np.concatenate((data[1],data[2],data[3],data[4],data[5]))
gg= [x[1] for x in combine if x[1]==1]
print len(gg)
Zero_count = len([x[0] for x in combine if x[0]==1])
one_count = len([x[1] for x in combine if x[1]==1])
two_count = len([x[2] for x in combine if x[2]==1])
three_count = len([x[3] for x in combine if x[3]==1])
four_count = len([x[4] for x in combine if x[4]==1])
five_count = len([x[5] for x in combine if x[5]==1 ])
six_count = len([x[6] for x in combine if x[6]==1 ])
seven_count = len([x[7] for x in combine if x[7]==1 ])
eight_count = len([x[8] for x in combine if x[8]==1 ])
nine_count = len([x[9] for x in combine if x[9]==1 ])
#blank_count = len([x[10] for x in combine if x[10]==1 ])
print "number of 0: {}".format(Zero_count)
print "number of 1: {}".format(one_count)
print "number of 2: {}".format(two_count)
print "number of 3: {}".format(three_count)
print "number of 4: {}".format(four_count)
print "number of 5: {}".format(five_count)
print "number of 6: {}".format(six_count)
print "number of 7: {}".format(seven_count)
print "number of 8: {}".format(eight_count)
print "number of 9: {}".format(nine_count)
#print "number of blanks: {}".format(blank_count)

plt.ion()
print data[1][ind]
print data[2][ind]
print data[3][ind]
print data[4][ind]
print data[5][ind]
print data[6][ind]
plt.imshow(data[0][ind].reshape((28,28)), interpolation="nearest", cmap="gray")


raw_input("Press any key to exit")


