
''' Python imports '''
import datetime as dt

''' 3rd party imports '''
import numpy as np
import pandas as pand
import matplotlib.pyplot as plot

''' QSTK imports '''
from QSTK.qstkutil import DataAccess as da
from QSTK.qstkutil import qsdateutil as du

from QSTK.qstkfeat.features import featMA, featRSI,featBollinger,featMomentum
from QSTK.qstkfeat.classes import class_fut_ret
import QSTK.qstkfeat.featutil as ftu

from LinRegLearner import *

def getData(fname):
    data = np.recfromcsv(fname, delimiter=',')

    list1 = [6]
    rawY = np.zeros(len(data))
    rawY = [[l[i] for i in list1] for l in data]
    rawY = np.array(rawY)    
    return rawY

def getFeature_std(rawY,lookback,lookforward):
    feature = np.zeros(len(rawY) - lookback - lookforward)

    index = 0
    for i in range (lookforward,len(rawY) - lookback):
        feature[index] = np.std(rawY[i:i+lookback])
        index += 1

    return feature

def getFeature_slope(rawY,lookback,lookforward):
    feature = np.zeros(len(rawY) - lookback - lookforward)

    x = np.array(range(1,22))
    A = np.vstack([x, np.ones(len(x))]).T

    index = 0
    for i in range(lookforward,len(rawY) - lookback):
        #m,c = np.linalg.lstsq(A, rawY[i:i+lookback])[0]
        #print "M = "+ str(m)
        feature[index] = np.linalg.lstsq(A, rawY[i:i+lookback])[0][0]
        #print "feature[0][0] "+ str(feature[index])

    return feature    

def getFeature_amp(rawY,lookback,lookforward):
    feature = np.zeros(len(rawY) - lookback - lookforward)

    index = 0
    for i in range(lookforward,len(rawY) - lookback):
        feature[index] = max(rawY[i:i+lookback]) - min(rawY[i:i+lookback])
        index += 1

    return feature

def getFeature_delta(rawY,lookback,lookforward):
    feature = np.zeros(len(rawY) - lookback - lookforward)

    index = 0
    for i in range (lookforward,len(rawY) - lookback):
        feature[index] = np.average(rawY[i:i+lookback]) - rawY[i]
        index += 1

    return feature

def getFeature_oneDayPriceChange(rawY,lookback,lookforward):
    feature = np.zeros(len(rawY) - lookback - lookforward)

    index = 0
    for i in range (lookforward,len(rawY) - lookback):
        feature[index] = rawY[i] - rawY[i+1]
        index += 1

    return feature

def getFeature_twoDayPriceChange(rawY,lookback,lookforward):
    feature = np.zeros(len(rawY) - lookback - lookforward)

    index = 0
    for i in range (lookforward,len(rawY) - lookback):
        feature[index] = (rawY[i] - rawY[i+1]) - (rawY[i+1] - rawY[i+2])
        index += 1

    return feature
    
def getActualY(rawY,lookback,lookforward):
    Y_actual = np.zeros(len(rawY) - lookback - lookforward)

    index = 0
    for i in range (lookforward,len(rawY) - lookback):
        Y_actual[index] = rawY[i- lookforward] - rawY[i]
        index += 1

    return Y_actual

def drawCharts(rmsValues,rmsValues_inSample, yLabel, title,name):
    plot.clf();
    plot.plot(range(1,101),rmsValues,c='blue',label="Yactual")
    plot.plot(range(1,101),rmsValues_inSample,c='red', label="Ypredict")
    plot.xlabel('Y->')
    #plot.ylabel("RMS Values")
    plot.ylabel(yLabel)
    #plot.title("In Sample RMS vs. Out of Sample RMS")
    plot.title(title)
    plot.legend()
    #plot.savefig("RMSCompare.pdf", format='pdf')
    plot.savefig(name, format='pdf')

def drawFive(feature1,feature2,feature3,feature4,feature5,name):
    plot.clf();
    plot.plot(range(1,101),feature1,c='blue',label="feature1")
    plot.plot(range(1,101),feature2,c='red', label="feature2")
    plot.plot(range(1,101),feature3,c='green', label="feature3")
    plot.plot(range(1,101),feature4,c='yellow', label="feature4")
    plot.plot(range(1,101),feature5,c='black', label="feature5")
    plot.xlabel('days->')
    #plot.ylabel("RMS Values")
    plot.ylabel('value')
    #plot.title("In Sample RMS vs. Out of Sample RMS")
    plot.title('First Five Features')
    plot.legend()
    #plot.savefig("RMSCompare.pdf", format='pdf')
    plot.savefig(name, format='pdf')


def calculateCoef(Ytest,Y):
    Ysqueeze = np.squeeze(Ytest[0:])
    corrcoef = numpy.corrcoef(Ysqueeze,Y)[1,0]
    return corrcoef

def calculateRMS(Ytest, Ycalc):
    sum = 0
    for i in range(0,len(Ycalc)):
        sum +=math.pow((Ycalc[i] - Ytest[i]),2)

    mean = sum/len(Ytest)
    rms = math.sqrt(mean)

    return rms


def drawScatterChart(Y_actual,Y_pred,name):
    plot.clf()
    X = list(xrange(len(Y_actual)))
    plot.scatter(X,Y_pred,color='red')
    plot.scatter(X,Y_actual,color='blue')
    plot.title(" Y actual vs  Y predicted")
    plot.savefig(name, format='pdf')



if __name__ == '__main__':
    

    lookback = 21
    lookforward = 5

    for i in range(0,100):
        if(i < 10):
            fname = "ML4T-00"+str(i)+".csv"
        else:
            fname = "ML4T-0"+str(i)+".csv"

        #fname = 'ML4T-000.csv'
        print fname
        rawY = getData(fname)



        if( i == 0):
            feature1 = np.zeros(len(rawY) - lookback - lookforward)
            feature2 = np.zeros(len(rawY) - lookback - lookforward)
            feature3 = np.zeros(len(rawY) - lookback - lookforward)
            feature4 = np.zeros(len(rawY) - lookback - lookforward)
            feature5 = np.zeros(len(rawY) - lookback - lookforward)
            feature6 = np.zeros(len(rawY) - lookback - lookforward)
            Y = np.zeros(len(rawY) - lookback - lookforward)


        X1       = np.zeros(len(rawY) - lookback - lookforward)
        X2       = np.zeros(len(rawY) - lookback - lookforward)
        X3       = np.zeros(len(rawY) - lookback - lookforward)
        X4       = np.zeros(len(rawY) - lookback - lookforward)
        X5       = np.zeros(len(rawY) - lookback - lookforward)
        X6       = np.zeros(len(rawY) - lookback - lookforward)
        Y_actual = np.zeros(len(rawY) - lookback - lookforward)

        X1 = getFeature_oneDayPriceChange(rawY,lookback,lookforward)
        X2 = getFeature_std(rawY,lookback,lookforward)
        X3 = getFeature_twoDayPriceChange(rawY,lookback,lookforward)
        X4 = getFeature_amp(rawY,lookback,lookforward)
        X5 = getFeature_delta(rawY,lookback,lookforward)
        X6 = getFeature_slope(rawY,lookback,lookforward)
        
        Y_actual = getActualY(rawY,lookback,lookforward)

        if( i == 0):
            feature1 = np.array(X1)
            feature2 = np.array(X2)
            feature3 = np.array(X3)
            feature4 = np.array(X4)
            feature5 = np.array(X5)
            feature6 = np.array(X6)
            Y = np.array(Y_actual)
            
        else:
            feature1 = np.append(feature1,X1)
            feature2 = np.append(feature2,X2)
            feature3 = np.append(feature3,X3)
            feature4 = np.append(feature4,X4)
            feature5 = np.append(feature5,X5)
            feature6 = np.append(feature6,X6)
            Y = np.append(Y, Y_actual)


   
    Xtrain = np.c_[feature1,feature2,feature3,feature4,feature5,feature6]
    #print Xtrain.shape

    print "Adding evidence"
    learner = LinRegLearner()
    learner.addEvidence(Xtrain, Y)
    print "Done add evidence"

    #------------------------------------------------------------------#
    fname = 'ML4T-135.csv'
    rawY_test1 = getData(fname)

    feature1_test1 = np.zeros(len(rawY_test1) - lookback - lookforward)
    feature2_test1 = np.zeros(len(rawY_test1) - lookback - lookforward)
    feature3_test1 = np.zeros(len(rawY_test1) - lookback - lookforward)
    feature4_test1 = np.zeros(len(rawY_test1) - lookback - lookforward)
    feature5_test1 = np.zeros(len(rawY_test1) - lookback - lookforward)
    feature6_test1 = np.zeros(len(rawY_test1) - lookback - lookforward)    
    Y_actual_test1 = np.zeros(len(rawY_test1) - lookback - lookforward)    

    feature1_test1 = getFeature_oneDayPriceChange(rawY_test1,lookback,lookforward)
    feature2_test1 = getFeature_std(rawY_test1,lookback,lookforward)
    feature3_test1 = getFeature_twoDayPriceChange(rawY_test1,lookback,lookforward)
    feature4_test1 = getFeature_amp(rawY_test1,lookback,lookforward)
    feature5_test1 = getFeature_delta(rawY_test1,lookback,lookforward)
    feature6_test1 = getFeature_slope(rawY_test1,lookback,lookforward)
    Y_actual_test1 = getActualY(rawY_test1,lookback,lookforward)

    Xtest_test1 = np.c_[feature1_test1,feature2_test1,feature3_test1,feature4_test1,feature5_test1,feature6_test1, np.ones(len(feature1_test1))]

    Y_pred_test1 = np.zeros(len(feature1_test1))

    for i in range (0, len(feature1_test1)):
        Y_pred_test1[i] = learner.query(Xtest_test1[i])

    #print Y_actual_test1
    #print Y_pred_test1

    coeff_test1 = calculateCoef(Y_actual_test1,Y_pred_test1)
    rms_test1 = calculateRMS(Y_actual_test1,Y_pred_test1)
    print "Coeff for 135: " +str(coeff_test1)
    print "RMS for 135: " +str(rms_test1)
    drawCharts(Y_actual_test1[0:100],Y_pred_test1[0:100],"Y values", "Y actual Vs Y predict (last 100) file 135","135_last100.pdf")
    drawCharts(Y_actual_test1[len(Y_actual_test1)-100:len(Y_actual_test1)],Y_pred_test1[len(Y_pred_test1)-100:len(Y_pred_test1)],"Y values", "Y actual Vs Y predict (first 100) file 135","135_first100.pdf")
    drawScatterChart(Y_actual_test1,Y_pred_test1,"135_ScatterPlot.pdf")
    drawFive(feature1_test1[len(feature1_test1)-100:len(feature1_test1)],feature2_test1[len(feature2_test1)-100:len(feature2_test1)],feature3_test1[len(feature3_test1)-100:len(feature3_test1)],feature4_test1[len(feature4_test1)-100:len(feature4_test1)],feature5_test1[len(feature5_test1)-100:len(feature5_test1)],"135_drawfive.pdf")
    #------------------------------------------------------------------#
    fname = 'ML4T-292.csv'
    rawY_test2 = getData(fname)

    feature1_test2 = np.zeros(len(rawY_test2) - lookback - lookforward)
    feature2_test2 = np.zeros(len(rawY_test2) - lookback - lookforward)
    feature3_test2 = np.zeros(len(rawY_test2) - lookback - lookforward)
    feature4_test2 = np.zeros(len(rawY_test2) - lookback - lookforward)
    feature5_test2 = np.zeros(len(rawY_test2) - lookback - lookforward)
    feature6_test2 = np.zeros(len(rawY_test2) - lookback - lookforward)    
    Y_actual_test2 = np.zeros(len(rawY_test2) - lookback - lookforward)    

    feature1_test2 = getFeature_oneDayPriceChange(rawY_test2,lookback,lookforward)
    feature2_test2 = getFeature_std(rawY_test2,lookback,lookforward)
    feature3_test2 = getFeature_twoDayPriceChange(rawY_test2,lookback,lookforward)
    feature4_test2 = getFeature_amp(rawY_test2,lookback,lookforward)
    feature5_test2 = getFeature_delta(rawY_test2,lookback,lookforward)
    feature6_test2 = getFeature_slope(rawY_test2,lookback,lookforward)
    Y_actual_test2 = getActualY(rawY_test2,lookback,lookforward)

    Xtest_test2 = np.c_[feature1_test2,feature2_test2,feature3_test2,feature4_test2,feature5_test2,feature6_test2, np.ones(len(feature1_test2))]

    Y_pred_test2 = np.zeros(len(feature1_test2))

    for i in range (0, len(feature1_test2)):
        Y_pred_test2[i] = learner.query(Xtest_test2[i])

    #print Y_actual_test2
    #print Y_pred_test2

    coeff_test2 = calculateCoef(Y_actual_test2,Y_pred_test2)
    rms_test2 = calculateRMS(Y_actual_test2,Y_pred_test2)
    print "Coeff for 292: " +str(coeff_test2)
    print "RMS for 292: " +str(rms_test2)
    drawCharts(Y_actual_test2[0:100],Y_pred_test2[0:100],"Y values", "Y actual Vs Y predict (last 100) file 292","292_last100.pdf")
    drawCharts(Y_actual_test2[len(Y_actual_test2)-100:len(Y_actual_test2)],Y_pred_test2[len(Y_pred_test2)-100:len(Y_pred_test2)],"Y values", "Y actual Vs Y predict (first 100) file 292","292_first100.pdf")
    drawScatterChart(Y_actual_test2,Y_pred_test2,"292_ScatterPlot.pdf")
    drawFive(feature1_test2[len(feature1_test2)-100:len(feature1_test2)],feature2_test2[len(feature2_test2)-100:len(feature2_test2)],feature3_test2[len(feature3_test2)-100:len(feature3_test2)],feature4_test2[len(feature4_test2)-100:len(feature4_test2)],feature5_test2[len(feature5_test2)-100:len(feature5_test2)],"292_drawfive.pdf")
    #---------------------------------------------------------------------#    
