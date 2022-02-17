from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score

#load the dataset
iris = datasets.load_iris()

classes = ['Iris Setosa','Iris Versicolour','Iris Virginica']
X = iris.data
y = iris.target

#split the data
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2)

model = svm.SVC()
model.fit(X_train,y_train)

predictions = model.predict(X_test)
accuracy = accuracy_score(predictions,y_test)

print(y_test)
print(predictions)
print(accuracy)

classes = ['Iris Setosa','Iris Versicolour','Iris Virginica']

for i in range(len(predictions)):
  print(classes[predictions[i]]+",",end =" ")
