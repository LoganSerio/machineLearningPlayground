from scipy.spatial import distance

#a is point from training data, b is point from testing data
def euc(a,b):
    return distance.euclidean(a,b)

class BarebonesKNN():
    def fit(self,x_train,y_train):
        self.x_train = x_train
        self.y_train = y_train

    def predict(self,x_test):
        predictions = []
        for row in x_test:
            label = self.closest(row)
            predictions.append(label)
        return predictions
    
    def closest(self, row):
        best_dist = euc(row, self.x_train[0])
        best_index = 0
        for i in range(1,len(self.x_train)):
            dist = euc(row,self.x_train[i])
            if dist < best_dist:
                best_dist = dist
                best_index = i
        return self.y_train[best_index]

# import dataset
from sklearn import datasets
iris = datasets.load_iris()

# Call the features x and the labels y
x = iris.data
y = iris.target

# taking features and labels and partitioning them into two sets train and test
from sklearn.cross_validation import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size = .5)


#from sklearn.neighbors import KNeighborsClassifier
my_classifier = BarebonesKNN()

# train classifier using training data
my_classifier.fit(x_train,y_train)

# predicts the testing data based on the training data
predictions = my_classifier.predict(x_test)

from sklearn.metrics import accuracy_score
print(accuracy_score(y_test,predictions))
