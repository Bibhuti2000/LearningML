from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from matplotlib import pyplot as plt
#load data from datasets
boston = datasets.load_boston()
#split it into features and labels
X = boston.data
y = boston.target
print(X.shape)
print(y.shape)

for i in range(len(X.T)):
   plt.scatter(X.T[i],y)
   plt.show()

#split the data into two parts for cross validation
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2)

#laod the model
model = linear_model.LinearRegression()
#train the model
resultant =  model.fit(X_train,y_train)

prediction = resultant.predict(X_test)

print(y_test)
print(prediction)
print("  ")
print('the r^2 value : ', model.score(X,y))
print("  ")
print('coeff : ', model.coef_)
print("  ")
print('intercepts : ',  model.intercept_)