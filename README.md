## A University Artificial Intelligence Coursework that consists of mainly 2 parts:
  1. Implementing a K-Nearest Classifier
  2. Implementing a Genetic Algorithm to perform a regression

> **_NOTE:_**  To run all the Python files in this project, use the command ```python <file_name>.py```

## Dataset Description
  1. The dataset consists of 900 labeled data points. Of these, 360 are for training and 540 are for validation
  2. Each data point denotes a location in 3D space, represented by coordinates (x, y, z)
  3. Every data point has either a label of 1 or -1
  4. There are 4 files for the dataset
     + training_set - contains the (x, y, z) coordinates of all points in the training set
     + training_labels - contains the labels of each data point in the training set, either 1 or -1
     + validation_set - contains the (x, y, z) coordinates of all points in the validation set
     + validation_labels - contains the labels of each data point in the validation set, either 1 or -1

## K-Nearest Classifier
  1. The K-Nearest Classifier code is in the ```classifier.py``` file
  2. The Output will display the Classification Error on the training and validation sets when k is set to 7, 19, and 31
  ![image](https://github.com/youssef-gerges-ramzy-mokhtar/Artificial-Intelligence-Coursework/assets/113933501/6bca4337-5352-4417-a2c4-ea2bfa3179a1)

## Genetic Algorithm
  1. The Genetic Algorithm code is in the ```regression.py``` file
  2. The genetic algorithm task requires the computation of three constant values: a0, a1, and a2. These constants are used in a plane equation, which is then used to classify points within the dataset. We select these constants in a way that minimizes classification error. For additional information, please refer to the coursework specification in the ```scc361.pdf``` file.
  3. The output will include the a0, a1, a2 values that the genetic algorithm has reached, as well as a graph demonstrating the population's overall fitness over time. <br />
    ![image](https://github.com/youssef-gerges-ramzy-mokhtar/Artificial-Intelligence-Coursework/assets/113933501/352553b7-0240-47cc-93b7-3faf62b5785c)
    ![image](https://github.com/youssef-gerges-ramzy-mokhtar/Artificial-Intelligence-Coursework/assets/113933501/08af1d15-17a6-4659-9495-0feed1ad75ea)
  
## Additional Notes
  1. There is a 3rd additional python file called ```boundary.py``` this file is used to compute the classification error on the training and validation sets, by using the a0, a1, a2 values generated from the genetic algorithm. You will need to manually set the a0, a1, a2 values in the ```boundary.py``` file
  2. Lastly, for a more technical understanding of the project, please refer to the ```scc361.pdf``` and ```Report.pdf``` files
