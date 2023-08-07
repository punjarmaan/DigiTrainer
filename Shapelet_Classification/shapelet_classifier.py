from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import os
import csv
import numpy as np
from random import sample
from tslearn.preprocessing import TimeSeriesScalerMinMax
from tslearn.shapelets import LearningShapelets, grabocka_params_to_shapelet_size_dict
from tslearn.utils import ts_size
import tensorflow as tf


np.random.seed(0)


# Retrieve all bicep file names
FILE_PATH = '../../MP/test_vids'
FILES = os.listdir(FILE_PATH)

# Separate train and test files
train_files, test_files = train_test_split(FILES, test_size=0.3, random_state=1234)

# Load csv paths
csv_paths = {
    'curl': '../../bicep_outputs/curl2.csv',
    'upper_arm_torso': '../../bicep_outputs/upper_arm2.csv',
    'torso_lean': '../../bicep_outputs/torso_lean2.csv',
    'wrist_flexion': '../../bicep_outputs/wrist_flexion2.csv'
}

# Retrieve data for test and train files
def retrieve_feature_data(files, data):
    all_data = []

    for file in files:
        current_data = []
        with open(data, mode='r') as f:
            csv_reader = csv.reader(f)
            for row in csv_reader:
                if row[0] == file:
                    current_data = row[2:300]
                    for i in range(len(current_data)):
                        if current_data[i] != '':
                            current_data[i] = float(current_data[i])
                        else:
                            current_data[i] = None

        all_data.append(current_data)
    
    return all_data

def retrieve_target_data(files, data):
    all_data = []

    for file in files:
        current_data = []
        with open(data, mode='r') as f:
            csv_reader = csv.reader(f)
            for row in csv_reader:
                if row[0] == file:
                    current_data = int(row[1])
        all_data.append(current_data)
    
    return all_data
    


x_train_1 = retrieve_feature_data(train_files, csv_paths['curl'])
x_train_2 = retrieve_feature_data(train_files, csv_paths['upper_arm_torso'])
x_train_3 = retrieve_feature_data(train_files, csv_paths['torso_lean'])
x_train_4 = retrieve_feature_data(train_files, csv_paths['wrist_flexion'])

x_test_1 = retrieve_feature_data(test_files, csv_paths['curl'])
x_test_2 = retrieve_feature_data(test_files, csv_paths['upper_arm_torso'])
x_test_3 = retrieve_feature_data(test_files, csv_paths['torso_lean'])
x_test_4 = retrieve_feature_data(test_files, csv_paths['wrist_flexion'])

y_train = retrieve_target_data(train_files, csv_paths['curl'])
y_test = retrieve_target_data(test_files, csv_paths['curl'])

x_train_1 = TimeSeriesScalerMinMax().fit_transform(x_train_1)
x_train_2 = TimeSeriesScalerMinMax().fit_transform(x_train_2)
x_train_3 = TimeSeriesScalerMinMax().fit_transform(x_train_3)
x_train_4 = TimeSeriesScalerMinMax().fit_transform(x_train_4)

x_test_1 = TimeSeriesScalerMinMax().fit_transform(x_test_1)
x_test_2 = TimeSeriesScalerMinMax().fit_transform(x_test_2)
x_test_3 = TimeSeriesScalerMinMax().fit_transform(x_test_3)
x_test_4 = TimeSeriesScalerMinMax().fit_transform(x_test_4)

n_ts, ts_sz = x_train_1.shape[:2]
n_classes = len(set(y_train))

# Set the number of shapelets per size as done in the original paper
shapelet_sizes = grabocka_params_to_shapelet_size_dict(n_ts=n_ts,
                ts_sz=ts_sz,
                n_classes=n_classes,
                l=0.1,
                r=1)

# Define the model using parameters provided by the authors (except that we
# use fewer iterations here)
shp_clf_curl = LearningShapelets(n_shapelets_per_size=shapelet_sizes,
            optimizer=tf.optimizers.Adam(.01),
            batch_size=16,
            weight_regularizer=.01,
            max_iter=200,
            random_state=42,
            verbose=0)

shp_clf_UAT = LearningShapelets(n_shapelets_per_size=shapelet_sizes,
            optimizer=tf.optimizers.Adam(.01),
            batch_size=16,
            weight_regularizer=.01,
            max_iter=200,
            random_state=42,
            verbose=0)

shp_clf_TL = LearningShapelets(n_shapelets_per_size=shapelet_sizes,
            optimizer=tf.optimizers.Adam(.01),
            batch_size=16,
            weight_regularizer=.01,
            max_iter=200,
            random_state=42,
            verbose=0)

shp_clf_WF = LearningShapelets(n_shapelets_per_size=shapelet_sizes,
            optimizer=tf.optimizers.Adam(.01),
            batch_size=16,
            weight_regularizer=.01,
            max_iter=200,
            random_state=42,
            verbose=0)

shp_clf_curl.fit(x_train_1, y_train)
shp_clf_UAT.fit(x_train_2, y_train)
shp_clf_TL.fit(x_train_3, y_train)
shp_clf_WF.fit(x_train_4, y_train)

# Make predictions and calculate accuracy score
pred_labels1 = shp_clf_curl.predict(x_test_1)
pred_labels2 = shp_clf_UAT.predict(x_test_2)
pred_labels3 = shp_clf_TL.predict(x_test_3)
pred_labels4 = shp_clf_WF.predict(x_test_4)

print("Correct classification rate:", accuracy_score(y_test, pred_labels1))
print("Correct classification rate:", accuracy_score(y_test, pred_labels2))
print("Correct classification rate:", accuracy_score(y_test, pred_labels3))
print("Correct classification rate:", accuracy_score(y_test, pred_labels4))


