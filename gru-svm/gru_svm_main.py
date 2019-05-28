# A Neural Network Architecture Combining Gated Recurrent Unit (GRU) and
# Support Vector Machine (SVM) for Intrusion Detection in Network Traffic Data
# Copyright (C) 2017  Abien Fred Agarap
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
# ==============================================================================

"""Implementation of the GRU+SVM model for Intrusion Detection"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import pandas as pd
#from models.gru_svm.gru_svm_llesmote import GruSvm
from models.gru_svm.gru_svm import GruSvm
from sklearn.model_selection import train_test_split
import warnings
import LBP_LLE_SMOTE_ as LBP_LLE_SMOTE
from imblearn.over_sampling import SMOTE

warnings.filterwarnings('ignore')

# hyper-parameters for the model
DROPOUT_P_KEEP = 0.85
HM_EPOCHS = 1000
LEARNING_RATE = 0.0007
N_CLASSES = 2
SEQUENCE_LENGTH = 784
SVM_C = 0.5

def parse_args():
    parser = argparse.ArgumentParser(description='GRU+SVM for Intrusion Detection')
    group = parser.add_argument_group('Arguments')
    group.add_argument('-lr', '--LEARNING_RATE', required=False, type=float, default=0e-5,
                       help='LEARNING_RATE, default = 0e-5')
    group.add_argument('-o', '--operation', required=True, type=str,
                       help='the operation to perform: "train" or "test"')
    group.add_argument('-c', '--checkpoint_path', required=True, type=str,
                       help='path where to save the trained model')
    group.add_argument('-l', '--log_path', required=False, type=str,
                       help='path where to save the TensorBoard logs')
    group.add_argument('-m', '--model_name', required=False, type=str,
                       help='filename for the trained model')
    group.add_argument('-r', '--result_path', required=True, type=str,
                       help='path where to save the actual and predicted labels')
    group.add_argument('-s', '--cell_size', required=False, type=int, default=256,
                       help='size of cell, default = 256')
    group.add_argument('-b', '--batch_size', required=False, type=int, default=256,
                       help='size of batch, default = 256')
    group.add_argument('-e', '--experimental_method', required=False, type=str, default='MNIST',
                       help='MNIST, LBP, CLBP, UCLBP, ... default = MNIST')
    group.add_argument('-n', '--n_neighbor', required=False, type=int, default=3,
                       help='LLE n-neighborr parameter, 1 to 50, default = 4')
    group.add_argument('-sm', '--SMOTE', required=False, type=str, default=False,
                       help='applying SMOTE or not applying SMOTE, default is False')
     
    arguments = parser.parse_args()
    return arguments


def main(argv):
    if argv.operation == 'train':
        if argv.experimental_method == 'MNIST':
            X, Y  = LBP_LLE_SMOTE.main('MNIST', argv.n_neighbor)
            Y = Y.ravel()
        elif argv.experimental_method == 'LBP':
            X, Y  = LBP_LLE_SMOTE.main('LBP', argv.n_neighbor)
        elif argv.experimental_method == 'CLBP':
            X, Y  = LBP_LLE_SMOTE.main('CLBP', argv.n_neighbor)
        elif argv.experimental_method == 'UCLBP': 
            X, Y  = LBP_LLE_SMOTE.main('UCLBP', argv.n_neighbor)
        elif argv.experimental_method == 'binary_clf': 
            X, Y  = LBP_LLE_SMOTE.main('binary_clf', argv.n_neighbor)
            Y = Y.ravel()
        elif argv.experimental_method == 'binary_clf_LBP': 
            X, Y  = LBP_LLE_SMOTE.main('binary_clf_LBP', argv.n_neighbor)
            Y = Y.ravel()
        elif argv.experimental_method == 'binary_clf_CLBP': 
            X, Y  = LBP_LLE_SMOTE.main('binary_clf_LBP', argv.n_neighbor)
            Y = Y.ravel()
        elif argv.experimental_method == 'binary_clf_UCLBP': 
            X, Y  = LBP_LLE_SMOTE.main('binary_clf_LBP', argv.n_neighbor)
            Y = Y.ravel()
        else:
            print('expremental_method error')

        train_features, validation_features, train_labels, validation_labels = train_test_split(X, Y, test_size=0.2)

        if argv.SMOTE == True:
            sm = SMOTE()
            train_features, train_labels = sm.fit_sample(train_features, train_labels)
     
        # get the size of the dataset for slicing
        train_size = train_features.shape[0]
        validation_size = validation_features.shape[0]

        # slice the dataset to be exact as per the batch size
        train_features = train_features[:train_size-(train_size % argv.batch_size)]
        train_labels = train_labels[:train_size-(train_size % argv.batch_size)]

        # modify the size of the dataset to be passed on model.train()
        train_size = train_features.shape[0]

        # slice the dataset to be exact as per the batch size
        validation_features = validation_features[:validation_size-(validation_size % argv.batch_size)]
        validation_labels = validation_labels[:validation_size-(validation_size % argv.batch_size)]

        # modify the size of the dataset to be passed on model.train()
        validation_size = validation_features.shape[0]
       
        LEARNING_RATE = argv.LEARNING_RATE 
        # instantiate the model
        model = GruSvm(alpha=LEARNING_RATE, batch_size=argv.batch_size, cell_size=argv.cell_size, dropout_rate=DROPOUT_P_KEEP,
                       num_classes=N_CLASSES, sequence_length=SEQUENCE_LENGTH, svm_c=SVM_C)

        # train the model
        model.train(checkpoint_path=argv.checkpoint_path, log_path=argv.log_path, model_name=argv.model_name,
                    epochs=HM_EPOCHS, train_data=[train_features, train_labels], train_size=train_size,
                    validation_data=[validation_features, validation_labels], validation_size=validation_size,
                    result_path=argv.result_path)
    elif argv.operation == 'test':
        if argv.experimental_method == 'MNIST':
            X, Y  = LBP_LLE_SMOTE.main('MNIST', argv.n_neighbor)
            Y = Y.ravel()
        elif argv.experimental_method == 'LBP':
            X, Y  = LBP_LLE_SMOTE.main('LBP', argv.n_neighbor)
        elif argv.experimental_method == 'CLBP':
            X, Y  = LBP_LLE_SMOTE.main('CLBP', argv.n_neighbor)
        elif argv.experimental_method == 'UCLBP': 
            X, Y  = LBP_LLE_SMOTE.main('UCLBP', argv.n_neighbor)
        elif argv.experimental_method == 'binary_clf': 
            X, Y  = LBP_LLE_SMOTE.main('binary_clf', argv.n_neighbor)
            Y = Y.ravel()
        elif argv.experimental_method == 'binary_clf_LBP': 
            X, Y  = LBP_LLE_SMOTE.main('binary_clf_LBP', argv.n_neighbor)
            Y = Y.ravel()
        else: 
            print('expremental_method error')

        test_features, _, test_labels, _ = train_test_split(X, Y, test_size=0.2)

        print(type(test_labels))

        test_size_ = test_features.shape[0]
        print(test_size_)

        test_features = test_features[:test_size_-(test_size_ % argv.batch_size)]
        test_labels = test_labels[:test_size_-(test_size_ % argv.batch_size)]
 
        GruSvm.predict(batch_size=argv.batch_size, cell_size=argv.cell_size, dropout_rate=DROPOUT_P_KEEP, num_classes=N_CLASSES,
                       test_data=[test_features, test_labels], test_size=test_size_, checkpoint_path=argv.checkpoint_path,
                       result_path=argv.result_path)


if __name__ == '__main__':
    args = parse_args()

    main(argv=args)
