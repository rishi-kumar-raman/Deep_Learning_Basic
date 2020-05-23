# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('C:/Users/vaiko/Desktop/Rishi_Raman_Deep_Learning/P16-Deep-Learning-AZ/Deep_Learning_A_Z/Volume 1 - Supervised Deep Learning/Part 1 - Artificial Neural Networks (ANN)/Mine/Test.csv')
X = dataset.iloc[:, 3:13].values
y = dataset.iloc[:, 13].values

# Dummies to be added to data
from sklearn.preprocessing import LabelEncoder
lb=LabelEncoder()
X[:,1] = lb.fit_transform(X[:,1])

lb2=LabelEncoder()
X[:,2] = lb2.fit_transform(X[:,2])


from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer([('encoder', OneHotEncoder(), [1])], remainder='passthrough')
X = np.array(ct.fit_transform(X), dtype=np.float)
# Encoding Y data
from sklearn.preprocessing import LabelEncoder
y = LabelEncoder().fit_transform(y)

X= X[:,1:]

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Fitting classifier to the Training set
# Create your classifier here


### ANN ##########
import keras 
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout

classifier =  Sequential()


classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu', input_dim = 11))
classifier.add(Dropout(p=0.1))

classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu'))
classifier.add(Dropout(p=0.1))

classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))

classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

classifier.fit(X_train,y_train,batch_size=10,epochs=100)
# Predicting the Test set results
y_pred = classifier.predict(X_test)
y_pred=(y_pred>0.5)


new_prediction=classifier.predict(sc.transform(np.array([[0,0,600,1,40,3,60000,2,1,1,50000]])))
y_new_pred=(new_prediction>0.5)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix,accuracy_score
cm = confusion_matrix(y_test, y_pred)
accuracy_score(y_test, y_pred)



# Evaluating the ANN:
import keras 
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score

def build_classifier():
    classifier =  Sequential()
    classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu', input_dim = 11))
    classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu'))
    classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))
    classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
    return classifier
    
classifier=KerasClassifier(build_fn=build_classifier,batch_size=10,epochs=100)
accuracies=cross_val_score(estimator=classifier,X=X_train,y=y_train,cv=10,n_jobs=1)

mean=accuracies.mean()
variance=accuracies.std()



# Tuning for the ANN classifier
#################################Tuning for the ANN classifier####################################################
import keras                                                                                                     
from keras.models import Sequential                                                                               
from keras.layers import Dense                                                                                   
from keras.wrappers.scikit_learn import KerasClassifier                                                                              
from sklearn.model_selection import GridSearchCV                                                                                 
                                                                                                                                                            
def build_classifier(optimizer):                                                                                                                                                            
    classifier =  Sequential()                                                                                   
    classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu', input_dim = 11))                                                                              
    classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu'))                        
    classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))                     
    classifier.compile(optimizer = optimizer, loss = 'binary_crossentropy', metrics = ['accuracy'])                 
    return classifier                                                                                            
                                                                                                                 
classifier=KerasClassifier(build_fn=build_classifier)                                   
parameters={'batch_size':[25,32],
            'epochs':[100, 500],
            'optimizer':['adam','rmsprop']}



##################################################################################################################