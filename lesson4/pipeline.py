# import dataset
from sklearn import datasets
iris = datasets.load_iris()

# Call the features x and the labels y
x = iris.data
y = iris.target

# taking features and labels and partitioning them into two sets train and test
from sklearn.cross_validation import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size = .5)

# tree classifier code
# from sklearn import tree
# my_classifier = tree.DecisionTreeClassifier()

from sklearn.neighbors import KNeighborsClassifier
my_classifier = KNeighborsClassifier()

# train classifier using training data
my_classifier.fit(x_train,y_train)

# predicts the testing data based on the training data
predictions = my_classifier.predict(x_test)

from sklearn.metrics import accuracy_score
print(accuracy_score(y_test,predictions))
