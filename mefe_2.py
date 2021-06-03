# coding=utf-8
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import decimal

## Helper mathematical activation functions

def safelog(x):
    return(np.log(x + 1e-100))

def sigmoid(x):
    return 1/(1+np.exp(-x))

def sigmoid_der(x):
    return sigmoid(x) *(1-sigmoid (x))

def softmax(A):
    expA = np.exp(A)
    return expA / expA.sum(axis=1, keepdims=True)



def main():
    ## reading data
    data = pd.read_excel('excel nepnew - Haziran1.xlsx', sheet_name='GRUP TAHMİNLİ', header=0,  usecols=('I'))

    # reading input variables as X values
    regressors_data = pd.read_excel('excel nepnew - Haziran1.xlsx', sheet_name='GRUP TAHMİNLİ', header=0, usecols=(
        'TOP/1000', 'LOGTOP' , 'YENILORAN',
        'RUZGAR/1000', 
        'JEO/1000', 'BARAJ/1000', 'BIO/1000', 'AKARS/1000', 
        'DEMAND/1000',
        'LOGRUZGAR', 'LOGJEO', 'LOGBARAJ', 'LOGBIO', 'LOGAKARS', 
        'LOGDEMAND',
        'WEEKEND',  'HOUR',
        'YATSAPMA-168/1000', 'PRICESAPMA-168/1000',
        'YATSAPMA-24/1000', 'PRICESAPMA-24/1000',
        'YATSAPMA-48/1000', 'PRICESAPMA-48/1000',
        'YATSAPMATAHMİN'
        ))
    ## converting dataframe to numpy
    X_data_set = regressors_data.to_numpy()

    ## reading outputs of the train data as Y values
    Y_data_set = data.get("PRICESAPMA/1000")
    Y_data_set = Y_data_set.to_numpy()

    ## one hot encoding of the train outputs
    classnum = 4
    Y_values = np.zeros((len(Y_data_set), classnum))
    for i in range(len(Y_data_set)):
        val = Y_data_set[i]
        if val <= -0.037:
            Y_values[i][0] = 1
        elif val <= 0.01:
            Y_values[i][1] = 1
        elif val <= 0.033:
            Y_values[i][2] = 1
        else:
            Y_values[i][3] = 1

    ## print the class sizes for each group
    sums = Y_values.sum(axis=0)
    print(sums)

    ## dividing Y values into train and test datasets
    ## this was used when model was initialed
    ## for model investigation 0.80 train rate is suggested
    ## for the prediction it should be closer to 1
    trainnum = int(len(Y_data_set)* 0.90)
    Y_train = Y_values[:trainnum]
    Y_test = Y_values[trainnum:]

    ## print the class sizes for each group in the train and test dataset
    sums = Y_train.sum(axis=0)
    print("Train ", sums)
    sums = Y_test.sum(axis=0)
    print("Test ", sums)

    # dividing X values into train and test data with respect to the train rate
    X_train = X_data_set[:trainnum]
    X_test = X_data_set[trainnum:]

    ## initialization of the hidden layer and nodes
    instances = X_train.shape[0]
    attributes = X_train.shape[1]
    hidden_nodes = 15
    output_labels = classnum

    ## initialization of the matrices between input-hidden-output layers
    wh = np.random.rand(attributes, hidden_nodes)
    bh = np.random.randn(hidden_nodes)
    wo = np.random.rand(hidden_nodes, output_labels)
    bo = np.random.randn(output_labels)

    ## learning rate, smaller is better but can cause overfit and long compilation. We used 10e-5
    lr = 10e-5
    error_cost = []

    ## model work 50000 iterations
    iteration = 0
    for epoch in range(50000):
        ## feedforward part
        # Phase 1
        zh = np.dot(X_train, wh) + bh
        ah = sigmoid(zh)
        # Phase 2
        zo = np.dot(ah, wo) + bo
        ao = softmax(zo)

        ## back propagation part
        ## phase 1
        dcost_dzo = ao - Y_train
        dzo_dwo = ah
        dcost_wo = np.dot(dzo_dwo.T, dcost_dzo)
        dcost_bo = dcost_dzo

        ## phases 2
        dzo_dah = wo
        dcost_dah = np.dot(dcost_dzo, dzo_dah.T)
        dah_dzh = sigmoid_der(zh)
        dzh_dwh = X_train
        dcost_wh = np.dot(dzh_dwh.T, dah_dzh * dcost_dah)
        dcost_bh = dcost_dah * dah_dzh

        ## Update Weights with respect to the learning rate
        wh -= lr * dcost_wh
        bh -= lr * dcost_bh.sum(axis=0)
        wo -= lr * dcost_wo
        bo -= lr * dcost_bo.sum(axis=0)

        ## calculate the loss function
        loss = np.sum(-Y_train * np.log(ao))
        error_cost.append(loss)
        iteration = iteration + 1

    ## Graph of the exponentially decay loss function
    plt.plot(range(1, iteration + 1), error_cost, "k-")
    plt.xlabel("Iteration")
    plt.ylabel("Error")
    plt.show()

    # calculate forecast results and confusion matrix for train dataset
    print("TRAIN")
    zh = np.dot(X_train, wh) + bh
    ah = sigmoid(zh)
    zo = np.dot(ah, wo) + bo
    y_predicted = softmax(zo)
    y_predicted_conf = np.argmax(y_predicted, axis=1) + 1
    Y_train_conf = np.argmax(Y_train, axis=1) + 1
    confusion_matrix_train = pd.crosstab(y_predicted_conf, Y_train_conf, rownames=['y_pred'], colnames=['y_truth'])
    print(confusion_matrix_train)
    count = 0
    i = 0
    for i in range(len(y_predicted_conf)):
        if y_predicted_conf[i] == Y_train_conf[i]:
            count = count + 1

    print(" TRAIN - SUCCESS RATE : ", count/len(y_predicted_conf))
    print(" ")

    # calculate forecast results and confusion matrix for test dataset
    print("TEST")
    zh = np.dot(X_test, wh) + bh
    ah = sigmoid(zh)
    zo = np.dot(ah, wo) + bo
    y_predicted_conf_test = np.argmax(softmax(zo), axis=1) + 1
    Y_test_conf_test = np.argmax(Y_test, axis=1) + 1
    confusion_matrix_test = pd.crosstab(y_predicted_conf_test, Y_test_conf_test, rownames=['y_pred'], colnames=['y_truth'])
    print(confusion_matrix_test)
    count = 0
    i = 0
    for i in range(len(y_predicted_conf_test)):
        if y_predicted_conf_test[i] == Y_test_conf_test[i]:
            count = count + 1
    print("TOTAL SUCCESS : ", count)
    print("TOTAL FAIL : ", len(y_predicted_conf_test)-count)
    print("SUCCESS RATE : ", count/len(y_predicted_conf_test))


    ########### FORECAST ##########

    data_tahmin = pd.read_excel('excel nepnew - Haziran1.xlsx', sheet_name='Tahmin Edilecek', header=0)
    # reading X values
    regressors_data_tahmin = pd.read_excel('excel nepnew - Haziran1.xlsx', sheet_name='Tahmin Edilecek', header=0, usecols=(
        'TOP/1000', 'LOGTOP', 'YENILORAN',
        'RUZGAR/1000',
        'JEO/1000', 'BARAJ/1000', 'BIO/1000', 'AKARS/1000',
        'DEMAND/1000',
        'LOGRUZGAR', 'LOGJEO', 'LOGBARAJ', 'LOGBIO', 'LOGAKARS',
        'LOGDEMAND',
        'WEEKEND', 'HOUR',
        'YATSAPMA-168/1000', 'PRICESAPMA-168/1000',
        'YATSAPMA-24/1000', 'PRICESAPMA-24/1000',
        'YATSAPMA-48/1000', 'PRICESAPMA-48/1000',
        'YATSAPMATAHMİN'
        ))

    ## initialization of output parameters for the forecast dimensions
    Y_data_set_tahmin = data_tahmin.get("PRICESAPMA/1000")
    Y_data_set_tahmin = Y_data_set_tahmin.to_numpy()
    Y_tahmin_reel = np.zeros((len(Y_data_set_tahmin), classnum))
    for i in range(len(Y_data_set_tahmin)):
        val = Y_data_set_tahmin[i]
        if val <= -0.037:
            Y_tahmin_reel[i][0] = 1
        elif val <= 0.01:
            Y_tahmin_reel[i][1] = 1
        elif val <= 0.033:
            Y_tahmin_reel[i][2] = 1
        else:
            Y_tahmin_reel[i][3] = 1
    X_tahmin = regressors_data_tahmin
    X_tahmin = X_tahmin.to_numpy()

    ## Calculating and displaying forecast results for the next day
    print("Prediction")
    zh = np.dot(X_tahmin, wh) + bh
    ah = sigmoid(zh)
    zo = np.dot(ah, wo) + bo
    y_predicted_conf_tahmin = np.argmax(softmax(zo), axis=1) + 1
    print("Predicted Classes",y_predicted_conf_tahmin)
    Y_test_conf_tahmin_reel = np.argmax(Y_tahmin_reel, axis=1) + 1
    print("Real Classes",Y_test_conf_tahmin_reel)
    confusion_matrix_tahmin = pd.crosstab(y_predicted_conf_tahmin, Y_test_conf_tahmin_reel, rownames=['y_pred'], colnames=['y_truth'])
    print(confusion_matrix_tahmin)


if __name__ == "__main__":
    main()
