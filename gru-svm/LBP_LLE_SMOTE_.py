import pandas  as pd
import matplotlib.pyplot as plt
import numpy as np
import itertools
import pickle

from imblearn.over_sampling import SMOTE
from sklearn.manifold import LocallyLinearEmbedding
from sklearn import preprocessing
from time import time

plt.rcParams["figure.figsize"] = (16,9)
plt.rcParams['lines.color'] = 'r'
plt.rcParams['axes.grid'] = False 

def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=0)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without finormalization')

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

def LLE_plot(data):
    """
    This function print and plots the result of LLE(Local Linear Embedding) algorithm.
    """
    print("Computing LLE embedding")
    t1 = time()
    for n in range(1, 50):
        plt.figure(figsize=(16,9))
        n_neighbors = n
        print("n_neighbors = %d"%n_neighbors)
        for i in range(10):

            condition = data['label'] == i
            subset_data = data[condition]

            clf = LocallyLinearEmbedding(n_neighbors, n_components=2, method='standard', eigen_solver='dense')
            t0 = time()
            X_lle = clf.fit_transform(subset_data)

            print("Done. Reconstruction error: %g" % clf.reconstruction_error_)
            print("Locally Linear Embedding of the digits (time %.2fs)" %(time() - t0))
            plt.scatter(X_lle[:, 0], X_lle[:, 1], cmap=plt.cm.hot, s=2, label='digit %d'%i)

        plt.ylim([-0.1, 0.1])
        plt.xlim([-0.2, 0.2])
        plt.legend()
        plt.grid()
        plt.savefig("./img/n-neighbor=%d.png"%n_neighbors, dpi=300)

    print("totally consumed time : (%.2fs)" %(time() - t1))


def LLE(data, n_neighbors, label_number):
    """
    play the LLE, return X
    for MNIST dataset, use dense attribute for eigen_solver in LocallyLinearEembedding class
    but if dataset big enough, use other attribute.(arpack, auto etc...)
    """
    print("processing digit: %d"%label_number)
    condition = data['label'] == label_number
    subset_data = data[condition]
    y_lle = np.array(subset_data.iloc[:, data.columns == 'label']) 
    clf = LocallyLinearEmbedding(n_neighbors, n_components=2, method='standard', eigen_solver='dense')
    X_lle = clf.fit_transform(subset_data)
    print(X_lle.shape)
    print(y_lle.shape)
    return X_lle, y_lle


def SMOTE_func(X_train, y_train):
    print("Number transactions X_train dataset: ", X_train.shape)
    print("Number transactions y_train dataset: ", y_train.shape)

    print("Before OverSampling, counts of label '0': {}".format(sum(y_train==0)))
    print("Before OverSampling, counts of label '1': {}".format(sum(y_train==1)))
    print("Before OverSampling, counts of label '2': {}".format(sum(y_train==2)))
    print("Before OverSampling, counts of label '3': {}".format(sum(y_train==3)))
    print("Before OverSampling, counts of label '4': {}".format(sum(y_train==4)))
    print("Before OverSampling, counts of label '5': {}".format(sum(y_train==5)))
    print("Before OverSampling, counts of label '6': {}".format(sum(y_train==6)))
    print("Before OverSampling, counts of label '7': {}".format(sum(y_train==7)))
    print("Before OverSampling, counts of label '8': {}".format(sum(y_train==8)))
    print("Before OverSampling, counts of label '9': {} \n".format(sum(y_train==9)))

    lab_enc = preprocessing.LabelEncoder()
    training_scores_encoded = lab_enc.fit_transform(y_train)

    sm = SMOTE()
    X, Y = sm.fit_sample(X_train, training_scores_encoded)

    print('After OverSampling, the shape of train_X: {}'.format(X.shape))
    print('After OverSampling, the shape of train_y: {} \n'.format(Y.shape))

    print("After OverSampling, counts of label '0': {}".format(sum(Y==0)))
    print("After OverSampling, counts of label '1': {}".format(sum(Y==1)))
    print("After OverSampling, counts of label '2': {}".format(sum(Y==2)))
    print("After OverSampling, counts of label '3': {}".format(sum(Y==3)))
    print("After OverSampling, counts of label '4': {}".format(sum(Y==4)))
    print("After OverSampling, counts of label '5': {}".format(sum(Y==5)))
    print("After OverSampling, counts of label '6': {}".format(sum(Y==6)))
    print("After OverSampling, counts of label '7': {}".format(sum(Y==7)))
    print("After OverSampling, counts of label '8': {}".format(sum(Y==8)))
    print("After OverSampling, counts of label '9': {}".format(sum(Y==9)))
   
    return X, Y
    
def plot_scatter(data, sm, n_neighbors):
    plt.figure()    
    """
    This function makes scatter plot.
    """
    for i in range(10):

        ValueToFind = i
        condition = (data[:,2]==ValueToFind)
        arr_find = data[condition]
        plt.scatter(arr_find[:,0], arr_find[:,1], cmap=plt.cm.hot, label='digit %d'%i, s=2)

    plt.ylim([-0.1, 0.1])
    plt.xlim([-0.2, 0.2])
    plt.legend()
    plt.grid()
    if sm:
        plt.savefig("./img/result/NOSMOTE_n_neighbors=%d.png"%n_neighbors, dpi=300)
    else:
        plt.savefig("./img/result/SMOTE_n_neighbors=%d.png"%n_neighbors, dpi=300)

def LLE_test(X, Y, n_neighbors, label_number):
    """
    play the LLE, return X
    for MNIST dataset, use dense attribute for eigen_solver in LocallyLinearEembedding class
    but if dataset big enough, use other attribute.(arpack, auto etc...)
    """

    print("shape of X", X.shape)
    print("shape of y", Y.shape)
    print("processing digit: %d"%label_number)
    condition = np.where(Y == label_number)
    sub_X = X[condition[0]]
    y_lle = np.where(Y==label_number) 
    clf = LocallyLinearEmbedding(n_neighbors, n_components=2, method='standard', eigen_solver='dense')
    X_lle = clf.fit_transform(sub_X)
    return X_lle, y_lle


def main(method, n_neighbor):
    n = n_neighbor
    if method == 'MNIST':
        data = pd.read_csv('./datasets/MNIST.csv')
        X = np.array(data.iloc[:, data.columns != 'label'])
        y = np.array(data.iloc[:, data.columns == 'label'])

        return  X, y

    elif method == 'LBP':
        data = pd.read_csv('./datasets/LBP.csv')
    elif method == 'CLBP':
        data = pd.read_csv('./datasets/CLBP.csv')
    elif method == 'UCLBP':
        data = pd.read_csv('./datasets/UCLBP.csv')
    elif method == 'binary_clf':
        data = data = pd.read_csv('./datasets/MNIST.csv')
        data_ = pd.read_csv('./datasets/generated_MNIST.csv')
        data_ = data_.sample(n=30000)

        exe = np.array(data.iloc[:, data.columns == 'label'])
        # real = 0, fake = 1
        X = np.array(data.iloc[:, data.columns != 'label'])
        X_ = np.array(data_)
        y = np.zeros_like(exe) 
        y_ = np.ones_like(X_)
        y_ = np.delete(y_, slice(1, 30000), axis=1)

        X = np.concatenate((X, X_))
        y = np.concatenate((y, y_))
        
        return X, y

    elif method == 'binary_clf_LBP':
        data = data = pd.read_csv('./datasets/LBP_MNIST.csv')
        data_ = pd.read_csv('./datasets/LBP_generated_MNIST.csv')
        data_ = data_.sample(n=30000)

        exe = np.array(data.iloc[:, data.columns == 'label'])
        # real = 0, fake = 1
        X = np.array(data.iloc[:, data.columns != 'label'])
        X_ = np.array(data_)
        y = np.zeros_like(exe) 
        y_ = np.ones_like(X_)
        y_ = np.delete(y_, slice(1, 30000), axis=1)

        X = np.concatenate((X, X_))
        Y = np.concatenate((y, y_))
        return X, Y

    elif method == 'binary_clf_CLBP':
        data = data = pd.read_csv('./datasets/CLBP_MNIST.csv')
        data_ = pd.read_csv('./datasets/CLBP_generated_MNIST.csv')
        data_ = data_.sample(n=60000)

        exe = np.array(data.iloc[:, data.columns == 'label'])
        # real = 0, fake = 1
        X = np.array(data.iloc[:, data.columns != 'label'])
        X_ = np.array(data_)
        y = np.zeros_like(exe) 
        y_ = np.ones_like(X_)
        y_ = np.delete(y_, slice(1, 60000), axis=1)

        X = np.concatenate((X, X_))
        Y = np.concatenate((y, y_))
        return X, Y

    elif method == 'binary_clf_UCLBP':
        data = data = pd.read_csv('./datasets/UCLBP_MNIST.csv')
        data_ = pd.read_csv('./datasets/UCLBP_generated_MNIST.csv')
        data_ = data_.sample(n=60000)

        exe = np.array(data.iloc[:, data.columns == 'label'])
        # real = 0, fake = 1
        X = np.array(data.iloc[:, data.columns != 'label'])
        X_ = np.array(data_)
        y = np.zeros_like(exe) 
        y_ = np.ones_like(X_)
        y_ = np.delete(y_, slice(1, 60000), axis=1)

        X = np.concatenate((X, X_))
        Y = np.concatenate((y, y_))
        return X, Y
    
    
    X = np.array(data.iloc[:, data.columns != 'label'])
    y = np.array(data.iloc[:, data.columns == 'label'])

    # LLE_plot(data)
    print("Computing LLE embedding")
    result_X_lle = np.empty([60000, 2])
    result_y_lle = np.empty([60000, 1])
    try:
        print("loading LLE results..")
        tmp1 = './tmp/CLBP/result_X_lle_n=%d.pkl'%n
        tmp2 = './tmp/CLBP/result_y_lle_n=%d.pkl'%n
        with open(tmp1,'rb') as fx: result_X_lle = pickle.load(fx)
        with open(tmp2,'rb') as fy: result_y_lle = pickle.load(fy)

    except:
        for i in range(10):
            # LLE(dataset, n_neighbor, label number)
            X_lle, y_lle = LLE(data, n, i)
            result_X_lle = np.concatenate((result_X_lle, X_lle))
            result_y_lle = np.concatenate((result_y_lle, y_lle))
  
        result_X_lle = np.nan_to_num(result_X_lle)
        result_X_lle.astype(int)
        result_y_lle.astype(int)
        result_X_lle = np.delete(result_X_lle, slice(0, 60000), axis=0)
        result_y_lle = np.delete(result_y_lle, slice(0, 60000), axis=0)
        filename1 = './tmp/CLBP/result_X_lle_n=%d.pkl'%n
        filename2 = './tmp/CLBP/result_y_lle_n=%d.pkl'%n     
        with open(filename1,'wb') as fx: pickle.dump(result_X_lle, fx)
        with open(filename2,'wb') as fy: pickle.dump(result_y_lle, fy)
  
    X, Y = SMOTE_func(result_X_lle, result_y_lle)

    return X, Y

if __name__ == "__main__":
    # execute only if run as a script
    main()




"""
        print("Computing LLE embedding")

        for i in [0, 1]:
            # LLE(dataset, n_neighbor, label number)
            X_lle, y_lle = LLE_test(X, y, n, i)
            result_X_lle = np.concatenate((result_X_lle, X_lle))
            result_y_lle = np.concatenate((result_y_lle, y_lle))
        result_X_lle = np.nan_to_num(result_X_lle)
        result_X_lle.astype(int)
        result_y_lle.astype(int)
        result_X_lle = np.delete(result_X_lle, slice(0, 60000), axis=0)
        result_y_lle = np.delete(result_y_lle, slice(0, 60000), axis=0)
        filename1 = './X_lle_n=%d.pkl'%n
        filename2 = './y_lle_n=%d.pkl'%n     
        with open(filename1,'wb') as fx: pickle.dump(result_X_lle, fx)
        with open(filename2,'wb') as fy: pickle.dump(result_y_lle, fy)  
        X, Y = SMOTE_func(result_X_lle, result_y_lle)
"""


 

