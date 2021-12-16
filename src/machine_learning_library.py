from math import sqrt
from os import curdir
from keras.metrics import MeanSquaredError
import numpy as np
from numpy.lib.shape_base import split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier
from sklearn.multioutput import MultiOutputRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn import svm
from sklearn.neural_network import MLPClassifier
import pandas
import matplotlib.pyplot as plt
from sklearn.linear_model import Perceptron
from keras.models import Sequential
from keras.layers import Dense
from keras.losses import MeanAbsoluteError
import tensorflow as tf

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

    # """ Very close result to np.matmul()"""
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
    """Import data and group the lidar protion into 4. 
    The robot position/orientation and local goal position/orientation 
    were grouped into deltas. The data was then regularized by averaging 
    and dividing by the max in their respective columns to scale them 
    between 0 and 1.

    Args:
        file (str): Relative path to csv file

    Returns:
        [list]: Features and their respective Labels
    """

    # Define Training Data
    data = pandas.read_csv(file)
    data = (data.to_numpy())

    # Define labels
    lin_vel = (data.T[-2]).T
    ang_vel = (data.T[-1]).T
    label = []
    for i in range(len(lin_vel)):
        label = np.append(label, [lin_vel[i], ang_vel[i]])
    label = np.reshape(label, (len(data), 2))

    # Define features vector (X_train/ X_test) pad with data set with a 1
    ld_end = 1080 # Last index of Lidar data
    ld_split = int(ld_end / 4) # Split lidar data into 67.5 degree segments
    lidar_1 = (data.T[:ld_split]).T # Lidar first 67.5 degree data
    lidar_2 = (data.T[ld_split:ld_split * 2]).T # Lidar next 67.5 degree to 135 degree data
    lidar_3 = (data.T[ld_split * 2:ld_split * 3]).T # Lidar first 135 degree to 202.5 degree data
    lidar_4 = (data.T[ld_split * 3:ld_split * 4]).T # Lidar first 202.5 degree to 270 degree data
    goal_l = [data.T[ld_end + 5], data.T[ld_end + 6]] # Local goal [x, y] feature
    goal_lq = [data.T[ld_end + 7], data.T[ld_end + 8]] # Local goal quaternion [qk, qr] feature
    robot_pos = [data.T[ld_end + 9], data.T[ld_end + 10]] # Robot pos [x, y] feature
    robot_q = [data.T[ld_end + 11], data.T[ld_end + 12]] # Robot orientation quaternion [qk, qr] feature
    
    pos_delta = np.subtract(goal_l, robot_pos) # Local goal - Robot pos [xl - xr, yl - yr] feature
    ori_delta = np.subtract(goal_lq, robot_q) # Local goal quaternion - Robot orientation quaternion [qk_l - qk_r, qr_l - qr_r] feature
    l_1 = []
    l_2 = []
    l_3 = []
    l_4 = []
    feature = []
    for i in range(len(data)): # Regularize data by taking the average 
        l_1 = np.append(l_1, sum(lidar_1[i]) / len(lidar_1[0]))
        l_2 = np.append(l_2, sum(lidar_2[i]) / len(lidar_2[0]))
        l_3 = np.append(l_3, sum(lidar_3[i]) / len(lidar_3[0]))
        l_4 = np.append(l_4, sum(lidar_4[i]) / len(lidar_4[0]))
    
    # Regularize by dividing my max value of each set of lidar data
    l_1 = l_1 / np.amax(l_1)
    l_2 = l_2 / np.amax(l_2)
    l_3 = l_3 / np.amax(l_3)
    l_4 = l_4 / np.amax(l_4)

    for i in range(len(data)): # Link all data in an array and reshape to match the number of features padded data set with a 1
        feature = np.append(feature, (1, l_1[i], l_2[i], l_3[i], l_4[i], pos_delta[0][i], pos_delta[1][i], ori_delta[0][i], ori_delta[1][i]))
    feature = np.reshape(feature, (len(label), 9))

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

def Test(x, y, y_test):
    lin_reg = LinearRegression()
    lin_reg.fit(x, y)
    w =lin_reg()
    # pocket
    # loop:
    # pick misclas point
    # error 
    # update weights
    # compare error with previous w error
    # repeat until t_max


    y_preditct = lin_reg.predict()
    print(np.sum((y_preditct - y_test)**2) / np.sum(y_test**2))

def RegModel(x, y):
    """Train a model using Decision Tree Regression.

    Args:
        x (list): List of feature data
        y (list): List of label data

    Returns:
        [DecisionTreeRegressor]: Trained models using max dpeth of 0.1 and 10
    """

    regr_1 = DecisionTreeRegressor(max_depth=0.1)
    regr_2 = DecisionTreeRegressor(max_depth=10)
    regr_1.fit(x, y)
    regr_2.fit(x, y)

    return regr_1, regr_2

if __name__ == '__main__':

    train_feature, train_label = ImportData('src/data/train_data/corridor_CSV/July22_1.csv')
    test_feature, test_label = ImportData('src/data/test_data/July22_68.csv')

    # Linear Regression
    lin_reg = LinearRegression()
    lin_reg.fit(train_feature, train_label)
    w_lin = lin_reg.coef_
    print(f"Linear Regression Weights: {w_lin.T}")
    print(lin_reg.score(train_feature, train_label))
    lin_reg_labels = lin_reg.predict(test_feature)
    lin_reg_err = mean_squared_error(test_label, lin_reg_labels)
    print(f"Lin_reg_err: {lin_reg_err}")

    # Decision Tree Regression
    reg_1, reg_2 = RegModel(train_feature, train_label)
    reg_1_labels = reg_1.predict(test_feature)
    # w_reg1 = reg_1.coef_
    reg_1_err = mean_squared_error(test_label, reg_1_labels)
    reg_1_train_accuracy = reg_1.score(train_feature, train_label)
    reg_1_test_accuracy = reg_1.score(test_feature, test_label)
    print(f"Reg_err_1: {reg_1_err}")
    print(f"Reg_1_test_accuracy: {reg_1_train_accuracy}") # https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeRegressor.html#sklearn.tree.DecisionTreeRegressor.score
    print(f"Reg_1_test_accuracy: {reg_1_test_accuracy}")
    reg_2_labels = reg_2.predict(test_feature)
    reg_2_err = mean_squared_error(test_label, reg_2_labels)
    reg_2_train_accuracy = reg_2.score(train_feature, train_label)
    reg_2_test_accuracy = reg_2.score(test_feature, test_label)
    print(f"Reg_err_2: {reg_2_err}")
    print(f"Reg_2_test_accuracy: {reg_2_train_accuracy}")
    print(f"Reg_2_test_accuracy: {reg_2_test_accuracy}")

    # # Multi-Layer-Perceptron-Classifier
    # mlp = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(15,), random_state=1)
    # mlp.fit(train_label, train_label) 

    # Stochastic Gradient Decent
    # sgd_model = SGDClassifier(loss="hinge", penalty="l2", max_iter=5)
    # MultiOutputRegressor(sgd_model).fit(train_feature, train_label)
    # w_sgd = lin_reg.coef_
    # print(w_sgd.T)
    # print(sgd_model.score(train_feature, train_label))

    initializer = tf.keras.initializers.HeUniform()

    model = Sequential([
        Dense(40, input_dim=len(train_feature.T), kernel_initializer = initializer, activation='sigmoid'),
        Dense(20, activation='sigmoid', kernel_initializer = initializer),
        Dense(10, activation='sigmoid', kernel_initializer = initializer),
        Dense(4, activation='sigmoid', kernel_initializer = initializer),
        Dense(2)    
    ])

    model.compile(loss=MeanAbsoluteError(), optimizer='adam', metrics=['accuracy'])
    seq_labels = model.predict(test_feature)
    seq_labels_train = model.predict(train_feature)
    seq_err = mean_squared_error(test_label, seq_labels)
    seq_err_train = mean_squared_error(train_label, seq_labels_train)    
    print(f"Test_Sequential_err: {seq_err}")
    print(f"Train_Sequential_err: {seq_err_train}")
