#using dataset MNIST
import tensorflow as tf
import pandas as pd
import numpy as np
from tensorflow.keras import Sequential,layers,losses,optimizers
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

#preprocessing
def preProcess(dataset):
    inputData=pd.read_csv(dataset,header=None)
    inputData=np.array(inputData,dtype=float)
    feature=np.delete(inputData,0,1)
    for i in range(len(feature)):
        for j in range(len(feature[i])):
            feature[i][j]=feature[i][j]/256
            if feature[i][j]==0:
                feature[i][j]=0.01
    target=np.zeros((len(feature),10))
    for i in range(len(feature)):
        for j in range(10):
            target[i][j]=0.01
            if inputData[i][0]==j:
                target[i][j]=0.99
    return feature,target

#BP network constructing
def BPNet3Layers(featureTrain,targetTrain,featureTest,targetTest,trainData):
    network=Sequential([
        layers.Dense(784,activation='sigmoid'),
        layers.Dense(100,activation='sigmoid'),
        layers.Dense(10,activation='sigmoid')
    ])
    network.build(input_shape=(None,1*784))
    optimizer=optimizers.Adam(lr=1e-4)
    lossTotal=[]
    criteon = losses.CategoricalCrossentropy(from_logits=False)
    for step,(featureTrain,targetTrain) in enumerate(trainData):
       with tf.GradientTape() as tape:
            outputTrain=network(featureTrain)
            loss=criteon(targetTrain,outputTrain)
       grads=tape.gradient(loss,network.trainable_variables)
       optimizer.apply_gradients(zip(grads,network.trainable_variables))
       if step%100==0:
            print("step:",step,"losses:",loss)
            lossTotal.append(loss)
            outputTest=network(featureTest)
            predict=tf.argmax(outputTest,1)
            targetValue=tf.argmax(targetTest,1)
            accurate,totalLines=0.,0.
            totalLines+=predict.shape[0]
            for i in range(len(predict)):
                if predict[i]==targetValue[i]:
                    accurate=accurate+1
            print("\n","accuracy:",accurate/totalLines)

dataset1="mnist_train.csv"
featureTrain,targetTrain=preProcess(dataset1)
trainData=tf.data.Dataset.from_tensor_slices((featureTrain,targetTrain))
trainData=trainData.batch(512)
trainData=trainData.repeat(20)
dataset2="mnist_test.csv"
featureTest,targetTest=preProcess(dataset2)
BPNet3Layers(featureTrain,targetTrain,featureTest,targetTest,trainData)
