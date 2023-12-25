import numpy as np

########## Global Data ##########
training_set = []
training_labels = []
validation_set = []
validation_labels = []

########## Reading the Files ##########
def fetchLabels(filename, arr):
	file = open(filename, "r")
	for line in file.readlines():
		label = line.split()
		arr += [float(label[0])]

def fetchSet(filename, arr):
	file = open(filename, "r")
	for line in file.readlines():
		x1,x2,x3 = line.split(sep=',')
		arr += [[float(x1),float(x2),float(x3)]]

def fetchFiles():
	fetchSet("training_set_v2", training_set)
	fetchLabels("training_labels_v2", training_labels)
	
	fetchSet("validation_set_v2", validation_set)
	fetchLabels("validation_labels_v2", validation_labels)

fetchFiles()

########## Classification Error ##########
def f(x, y, constants):
	a0, a1, a2 = constants
	return (a0*np.cbrt(x-5)) + (a1*np.cbrt(y+5)) + a2

def classficationError(data, labels, constants):
	error = 0

	for idx, [x,y,z] in enumerate(data):
		function_z = f(x, y, constants)
		if not ((z >= function_z and labels[idx] == 1) or (z < function_z and labels[idx] == -1)):
			error += 1

	return error/len(data)

constants = [1.0348937638125875, -1.043782372998818, 0.007752099457335265] # plug [a0, a1, a2] here
print("Classification error on training set: " + str(classficationError(training_set, training_labels, constants)))
print("Classification error on validation set: " + str(classficationError(validation_set, validation_labels, constants)))