# 1.Load data va chia Train, Val and test
#from keras import Sequential
from keras.layers import Dense

from tensorflow.keras import Sequential
from sklearn.model_selection import train_test_split
from numpy import loadtxt 
dataset = loadtxt('pima-indians-diabetes.data.csv',delimiter=',')

X = dataset[:,0:8]
y = dataset[:,8]

# chia data thanh train +val , va test.
#test_size = 0.2 : la 20% data --> train + val = 80%
X_train_val, X_test, y_train_val, y_test =train_test_split(X,y,test_size = 0.2)

# train = 80%, val = 20%
X_train,X_val,y_train,y_val = train_test_split(X_train_val,y_train_val,test_size =0.2)

    # build model
model = Sequential()
model.add(Dense(16,input_dim=8,activation = 'relu'))
model.add(Dense(8,activation = 'relu'))
model.add(Dense(1,activation = 'sigmoid'))
    #show model 
model.summary()
    #compile model
model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['Accurary'])
    #train model

    #batch-size: so luong phan tu duoc chon vao model train
    #Epoch : so lan duyet qua phan tu " batch-size"

model.fit(X_train, y_train, epochs =100,batch_size =8,validation_data=(X_val,y_val))


///