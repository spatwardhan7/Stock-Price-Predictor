"""
Following code in QSTK/qstklearn/kdtknn.py
"""
import math,random,sys,bisect,time
import numpy,scipy.spatial.distance
from scipy.spatial import cKDTree
import numpy as np

class LinRegLearner():
    def __init__(self):
        """
        Basic Setup
        """
    def addEvidence(self,dataX,dataY):

        #print "Lin Reg Learner : Shape of dataX:" +str(dataX.shape)
        
        list0 = [0]
        list1 = [1]
        list2 = [2]
        list3 = [3]
        list4 = [4]
        list5 = [5]
        

        x0 = [[l[i] for i in list0] for l in dataX]
        x1 = [[l[i] for i in list1] for l in dataX]
        x2 = [[l[i] for i in list2] for l in dataX]
        x3 = [[l[i] for i in list3] for l in dataX]
        x4 = [[l[i] for i in list4] for l in dataX]
        x5 = [[l[i] for i in list5] for l in dataX]

        x0 = np.squeeze(x0[0:])
        x1 = np.squeeze(x1[0:])
        x2 = np.squeeze(x2[0:])
        x3 = np.squeeze(x3[0:])
        x4 = np.squeeze(x4[0:])
        x5 = np.squeeze(x5[0:])

        x0 = np.array(x0)
        x1 = np.array(x1)
        x2 = np.array(x2)
        x3 = np.array(x3)
        x4 = np.array(x4)
        x5 = np.array(x5)

        stk = np.vstack([x0,x1,x2,x3,x4,x5, np.ones(len(dataX))]).T
        self.x, res,rank,s = numpy.linalg.lstsq(stk,dataY)
        
    def query(self,Xtest):

        '''
        list1 = [0]
        list2 = [1]

        x1 = [[l[i] for i in list1] for l in Xtest]
        x2 = [[l[i] for i in list2] for l in Xtest]

        x1 = np.squeeze(x1[0:])
        x2 = np.squeeze(x2[0:])
    
        return (x1*self.x[0] + x2*self.x[1] + self.x[2])
        '''
        #print "Lin Reg Learner :Shape of x" +str(self.x.shape)
        #print "lin Reg Learner: Shape of Xtest" +str(Xtest.shape)
        return np.dot(self.x,Xtest)
        
def getTrainingData(fname,trainingThreshold):
    data = np.recfromcsv(fname, delimiter=',', names=['X1','X2','Y'])

    trainingSet = data[0:int(len(data)*trainingThreshold)]
    testingSet = data[int(len(data)*trainingThreshold):]

    list1 = [0,1]
    list2 = [2]
    
    Xtrain = [[l[i] for i in list1] for l in trainingSet]
    Ytrain = [[l[i] for i in list2] for l in trainingSet]

    Xtest = [[l[i] for i in list1] for l in testingSet]
    Ytest = [[l[i] for i in list2] for l in testingSet]
    
    return Xtrain,Ytrain,Xtest,Ytest
    
if __name__=="__main__":
    Xtrain,Ytrain,Xtest,Ytest = getTrainingData('data-classification-prob.csv',0.6)
    learner = LinRegLearner()
    learner.addEvidence(Xtrain,Ytrain)
    Y = learner.query(Xtest)
    print Y
