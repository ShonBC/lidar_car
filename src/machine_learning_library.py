"""
(15 pts) Using the image data uploaded on ELMS, perform classification using
the following algorithms for non-separable data:
(a) Perceptron algorithm.
(b) Pocket algorithm.
Use your chosen algorithm to find the best separator you can using the train-
ing data only (you can create your own features). The output is +1 if the
example is a 1 and -1 for a 5.
(a) Give separate plots of the training and test data, together with the sep-
arators.
(b) Compute Ein on your training data and Etest, the test error on the test
data.
(c) Obtain a bound on the true out-of-sample error. You should get two
bounds, one based on Ein and one based on Etest. Use a tolerance δ =
0.05. Which is the better bound?
(d) Now repeat using a 2nd order polynomial transform.
(e) As your final deliverable to a customer, would you use the model with
or without the 2nd order polynomial transform? Explain.
"""

from math import sqrt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
import pandas
import matplotlib.pyplot as plt

def intensity(x):
    intn = np.mean(x)
    return intn

def symmetry(x):
    rev = np.flip(x) 
    symm = - np.mean(np.abs(x - rev))
    return symm

def Pocket(feature, label, weight, test_feature, test_label, max_itter = 100, show_plot=False):
    iteration = 0
    weights = weight
    label[label==5] = -1
    while(iteration < max_itter):
        iteration += 1
        w = weights
        misClassifications=0
        for i in range(0,len(feature)):
            currentX = feature[i].reshape(-1,feature.shape[1])
            currentY = label[i]
            if currentY != np.sign(np.dot(currentX, w.T)):
                w = w + currentY*currentX
                misClassifications=1
            # print(misClassifications)
            if misClassifications==1:
                break
        Ein_w = 0
        Ein_weights = 0
        for i in range(0,len(feature)):
            currentX = feature[i].reshape(-1,feature.shape[1])
            currentY = label[i]    
            if currentY != np.sign(np.dot(currentX, w.T)):
                Ein_w +=1
            if currentY != np.sign(np.dot(currentX, weights.T)):
                Ein_weights +=1
        if Ein_w < Ein_weights:
            weights = w

    # In-Sample Error using training data
    Ein = Ein_w / len(label)    
    w_poc = weights

    # Out of Sample Error using test data
    Eout = 0
    for i in range(0,len(test_feature)):
        currentX = test_feature[i].reshape(-1,test_feature.shape[1])
        currentY = test_label[i]    
        if currentY != np.sign(np.dot(currentX, w_poc.T)):
            Eout +=1

    Eout = Eout / len(test_label)
    
    if show_plot:
        x_plot = np.linspace(0, 0.4, 100)
        y_plot = - w_poc[1]/w_poc[2]*x_plot - w_poc[0]/w_poc[2]
        plt.scatter(feature[:6742,1],feature[:6742,2],c='b',label='1',marker='+')
        plt.scatter(feature[6742:,1],feature[6742:,2],c='r',label='5',marker='o')
        plt.plot(x_plot,y_plot,'-') 

    return w_poc, Ein, Eout

def Perceptron(feature, label, max_itter = 1000, show_plot=False):
    w_per = np.array([0, 0, 0])
    for epoch in range(max_itter):
        i = np.random.randint(low=0, high=len(feature))
        s = np.dot(w_per, np.array(feature[i]))

        if label[i] * s <= 1:
            w_per = w_per + (label[i] * feature[i])

    Ein = 0
    for i in range(0,len(feature)):
        currentX = feature[i].reshape(-1, feature.shape[1])
        currentY = label[i]    
        if currentY != np.sign(np.dot(currentX, w_per.T)):
            Ein +=1

    Ein = Ein / len(label)

    return w_per, Ein

def LinReg(feature, label, show_plot=False):
    lin_reg = LinearRegression()
    lin_reg.fit(feature, label)

    """ Very close result to np.matmul()"""
    # w_lin = lin_reg.coef_

    w_lin = np.matmul(np.matmul(np.linalg.inv(np.matmul(feature.T,feature)),feature.T),label)

    if show_plot:
        x_plot = np.linspace(0, 0.4, 100)
        y_plot = - w_lin[1]/w_lin[2]*x_plot - w_lin[0]/w_lin[2]
        plt.scatter(feature[:6742,1],feature[:6742,2],c='b',label='1',marker='+')
        plt.scatter(feature[6742:,1],feature[6742:,2],c='r',label='5',marker='o')
        plt.plot(x_plot,y_plot,'-') 
        plt.xlabel("Intensity")
        plt.ylabel("Symmetry")
        plt.title('Linear Regression')
        plt.legend(['Linear Regression'], loc='upper right')

    return w_lin

def CalcError(feature, label, weight):
    error = 0
    for i in range(0,len(feature)):
        currentX = feature[i].reshape(-1, feature.shape[1])
        currentY = label[i]    
        if currentY != np.sign(np.dot(currentX, weight.T)):
            error +=1

    error = error / len(label)

    return error

def ImportData(file):
    # Define Training Data
    data = pandas.read_csv(file)
    data = (data.to_numpy())
    label = (data.T[0]).T
    # Set all 5 labels equal to -1
    label[label==5] = -1
    # Regularize data to be between 0 and 1
    x = (data.T[1:]).T/255

    # Define training features vector (X_train) pad with data set with a 1
    feature = []
    for curX in x:
        cur_intn = intensity(curX.reshape((28, 28)))
        cur_symm = symmetry(curX.reshape((28, 28)))
        feature = np.append(feature,(1, cur_intn, cur_symm))
    feature = np.reshape(feature, (len(label), 3))

    return feature, label

def ShowData(feature, label, weight, seperator_color, title, algorithm):
    x_plot = np.linspace(0, 0.4, 100)
    y_plot = - weight[1]/weight[2]*x_plot - weight[0]/weight[2]
    idx = np.where(label == -1)
    plt.scatter(feature[:min(idx[0]),1],feature[:min(idx[0]),2],c='b',label='1',marker='+')
    plt.scatter(feature[min(idx[0]):,1],feature[min(idx[0]):,2],c='r',label='5',marker='o')
    plt.plot(x_plot,y_plot,f'-{seperator_color}') 
    plt.xlabel("Intensity")
    plt.ylabel("Symmetry")
    plt.title(title)
    plt.legend([algorithm], loc='upper right')

def TrainErrorBound(features, e_in):
    """Use the VC generalization bound (inequality 2.12 in LFD book) to obtain a bound on the true out-of-sample error based on E_in
    Growth function (m_H) is bounded by m_H <= N^(d_vc)+1. For linear model, d_vc=d+1, where d is number of features.
    Substitute all these into the inequality 2.12.

    Args:
        features ([np.array]): Features to represent data
        e_in ([int]): Insample error for training data

    Returns:
        [int]: Max out of sample error the model can achieve
    """

    # For linear model, d_vc=d+1, where d is number of features.
    d_vc = len(features[0])
    # N is the number of data points
    n = len(features)
    tolerance = 0.05
    # Growth function (m_H) is bounded by m_H <= N^(d_vc)+1
    m_H = n**(d_vc) + 1
    a = (8 / n) * np.log((4 * m_H * 2 * n) / tolerance)
    e_out_bound = e_in + sqrt(a)

    return e_out_bound
    
def TestErrorBound(features, e_in, m = 1):   
    """For a bound based on E_test, use the Hoeffding bound (inequality 2.1 in LFD book). 
    Here M=1 (# of times model has been trained), N=2072 (test size). \delta=0.5 (tolerance) is the same for both cases.

    Args:
        features ([np.array]): Features to represent data
        e_in ([int]): Insample error for training data
        m ([int]): M is number of times model has been trained on various data sets.

    Returns:
        [int]: Max out of sample error the model can achieve.
    """
    
    # N is the number of data points
    n = len(features)
    tolerance = 0.05
    a = (1 / (2 * n)) * np.log((2 * m) / tolerance)
    e_out_bound = e_in + sqrt(a)

    return e_out_bound

if __name__ == '__main__':

    train_feature, train_label = ImportData('MNIST_data/mnist_train_binary.csv')
    test_feature, test_label = ImportData('MNIST_data/mnist_test_binary.csv')

    w_per, e_in_per = Perceptron(train_feature, train_label)
    e_out_per = CalcError(test_feature, test_label, w_per)
    train_e_bound_per = TrainErrorBound(train_feature, e_in_per)
    test_e_bound_per = TestErrorBound(test_feature, e_in_per)
    # print(f'Perceptron: w = {w_per} Ein = {e_in_per} Eout = {e_out_per}')

    w_poc, e_in_poc, e_out_poc = Pocket(train_feature, train_label, w_per, test_feature, test_label)
    train_e_bound_poc = TrainErrorBound(train_feature, e_in_poc)
    test_e_bound_poc = TestErrorBound(test_feature, e_in_poc)
    # print(f'Pocket: w = {w_poc} Ein = {e_in} Eout = {e_out}')

    # Part (a) Perceptron
    # ShowData(train_feature, train_label, w_per, 'g', 'Perceptron Training Data', 'Perceptron')
    # ShowData(test_feature, test_label, w_per, 'g', 'Perceptron Test Data', 'Perceptron')
    # plt.show()

    # Part (a) Pocket
    # ShowData(train_feature, train_label, w_poc, 'g', 'Perceptron + Pocket Training Data', 'Perceptron + Pocket')
    # ShowData(test_feature, test_label, w_poc, 'g', 'Perceptron + Pocket Test Data', 'Perceptron + Pocket')
    # plt.show()

    # Part (b) Calculate Ein and Eout
    print(f'Perceptron: w = {w_per} Ein = {e_in_per} Eout = {e_out_per} In-Sample Error Bound: {train_e_bound_per} Test Error Bound: {test_e_bound_per}')
    print(f'Perceptron + Pocket: w = {w_poc} Ein = {e_in_poc} Eout = {e_out_poc} In-Sample Error Bound: {train_e_bound_poc} Test Error Bound: {test_e_bound_poc}')

    # Part (d) 2nd Order Linear Regression
    w_lin = LinReg(train_feature, train_label)
    e_in_lin = CalcError(train_feature, train_label, w_lin)
    e_out_lin = CalcError(test_feature, test_label, w_lin)
    train_e_bound_lin = TrainErrorBound(train_feature, e_in_lin)
    test_e_bound_lin = TestErrorBound(test_feature, e_in_lin)
    
    w_poc_lin, e_in_poc_lin, e_out_poc_lin = Pocket(train_feature, train_label, w_lin, test_feature, test_label)
    train_e_bound_poc_lin = TrainErrorBound(train_feature, e_in_poc_lin)
    test_e_bound_poc_lin = TestErrorBound(test_feature, e_in_poc_lin)

    ShowData(train_feature, train_label, w_poc_lin, 'g', 'Linear Regression + Pocket Training Data', 'Linear Regression + Pocket')
    
    print(f'Linear Regression: w = {w_lin} Ein = {e_in_lin} Eout = {e_out_lin} In-Sample Error Bound: {train_e_bound_lin} Test Error Bound: {test_e_bound_lin}')
    print(f'Linear Regression + Pocket: w = {w_poc_lin} Ein = {e_in_poc_lin} Eout = {e_out_poc_lin} In-Sample Error Bound: {train_e_bound_poc_lin} Test Error Bound: {test_e_bound_poc_lin}')
    plt.show()
