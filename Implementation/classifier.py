import numpy as np

################## Class Definitions ##################

# point just holds that point coordinates and label in a single object and defines the euclidean distance function between other Points
class Point:
	def __init__(self, x, y, z, label):
		self.x = x
		self.y = y
		self.z = z
		self.label = label

	def __str__(self):
		return f"({self.x}, {self.y}, {self.z}): {self.label}"

	def euclidDist(self, otherPoint):
		x = otherPoint.x
		y = otherPoint.y
		z = otherPoint.z

		return np.sqrt(
			((self.x - x) ** 2) +
			((self.y - y) ** 2) +
			((self.z - z) ** 2)
		)

# distance is mainly used to not lose track of what is the srcPoint when sorting the distances in ascneding order
class Distance:
	def __init__(self, srcPoint, classificationPoint):
		self.srcPoint = srcPoint
		self.classificationPoint = classificationPoint
		self.dist = srcPoint.euclidDist(classificationPoint)

	def __lt__(self, other): # used for sorting based on the euclidean distance between the srcPoint and the classificationPoint
		return self.dist < other.dist

################## Reading the Files ##################
training_set = []
validation_set = []

def fetchFile(data, labels, arr):
	file = open(data, "r")
	for line in file.readlines():
		x1,x2,x3 = line.split(sep=',')
		arr.append(Point(float(x1),float(x2),float(x3), 1))

	file = open(labels, "r")
	for idx, line in enumerate(file.readlines()):
		label = line.split()
		arr[idx].label = int(label[0])

def fetchFiles():
	fetchFile("training_set_v2", "training_labels_v2", training_set)
	fetchFile("validation_set_v2", "validation_labels_v2", validation_set)

fetchFiles()

################## K-Nearest Classifier ##################
def getSortedDistances(point):
	dist = []
	for srcPoint in training_set:
		dist.append(Distance(srcPoint, point))

	dist.sort()
	return dist

def classifyPoint(point, k):
	dist = getSortedDistances(point)
	
	label1 = label2 = 0
	for i in range(k):
		if dist[i].srcPoint.label == 1:
			label1 += 1
		else:
			label2 += 1

	if label1 > label2:
		return 1
	else:
		return -1

def kNearestClassifierError(k, data):
	errorCnt = 0
	for idx, point in enumerate(data):
		classification = classifyPoint(point, k)
		if classification != data[idx].label:
			errorCnt += 1

	return errorCnt/len(data)

kValues = [7, 19, 31]
for k in kValues:
	print("k=" + str(k))
	print("Classification error on training set: " + f"{kNearestClassifierError(k, training_set):.4f}")
	print("Classification error on validation set: " + f"{kNearestClassifierError(k, validation_set):.4f}")