import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import Sequential,layers,losses,optimizers
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

#数据预处理
def preProcessData(dataset):
    inputData = pd.read_csv(dataset, header=None)

    inputData=np.array(inputData,dtype=float)
    feature=np.delete(inputData,0,1)
    for i in range(len(feature)):
        for j in range(784):
            feature[i][j]=feature[i][j]/256
            if feature[i][j]==0:
                feature[i][j]=0.01

    target = np.zeros((len(feature), 10))
    for i in range(len(feature)):
        for j in range(10):
            target[i][j] = 0.01
            if inputData[i][0] == j:
                target[i][j] = 0.99

    return feature,target
#建立网络
def LeNet(featureTrain,targetTrain,featureTest,targetTest,trainData):
    network=Sequential([
        layers.Conv2D(6,kernel_size=5,strides=1),
        layers.MaxPooling2D(pool_size=2,strides=2),
        layers.ReLU(),
        layers.Conv2D(16,kernel_size=5,strides=1),
        layers.MaxPooling2D(pool_size=2,strides=2),
        layers.ReLU(),
        layers.Flatten(),
        layers.Dense(120,activation='sigmoid'),
        layers.Dense(84,activation='sigmoid'),
        layers.Dense(10,activation='softmax')
    ])
    network.build(input_shape=(4,28,28,1))
    optimizer=optimizers.Adam(lr=1e-4)
    lossTotal=[]
    criteon=losses.CategoricalCrossentropy(from_logits=False)
    for step,(featureTrain,targetTrain) in enumerate(trainData):
        featureTrain=tf.reshape(featureTrain,(-1,28,28))
        with tf.GradientTape() as tape:
            output=network(featureTrain)
            loss=criteon(targetTrain,output)
        grads=tape.gradient(loss,network.trainable_variables)
        optimizer.apply_gradients(zip(grads,network.trainable_variables))
        if step%100==0:
            print("step:",step,loss)
            lossTotal.append(loss)
            featureTest=tf.reshape(featureTest,(-1,28,28))
            output=network(featureTest)
            predict=tf.argmax(output,1)
            accurate,total=0.,0.
            total+=featureTest.shape[0]
            targetValue=tf.argmax(targetTest,1)
            for i in range(len(predict)):
                if predict[i]==targetValue[i]:
                    accurate=float(accurate+1)
            print("\n","accuracy",float(accurate/total))

dataset1="mnist_train.csv"
featureTrain,targetTrain=preProcessData(dataset1)
trainData=tf.data.Dataset.from_tensor_slices((featureTrain,targetTrain))
trainData=trainData.batch(512)
trainData=trainData.repeat(20)
dataset2="mnist_test.csv"
featureTest,targetTest=preProcessData(dataset2)
print("train data:")
print(trainData)
LeNet(featureTrain,targetTrain,featureTest,targetTest,trainData)


# array=np.array([[1,2,3,4],[4,5,6,7],[7,8,9,10]])
# array=array.reshape((3,4))
# print(array)
