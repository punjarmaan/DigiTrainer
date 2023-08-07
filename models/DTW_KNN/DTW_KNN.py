from sklearn.model_selection import train_test_split
from dtaidistance import dtw
import os
import csv
import numpy as np
from sklearn.metrics import classification_report
from random import sample


# Retrieve all bicep file names
FILE_PATH = '../../MP/test_vids'
FILES = os.listdir(FILE_PATH)

# Separate train and test files
train_files, test_files = train_test_split(FILES, test_size=0.3, random_state=1234)

# Load csv paths
csv_paths = {
    'curl': '../../bicep_outputs/curl_angles.csv',
    'upper_arm_torso': '../../bicep_outputs/upper_arm_torso_angles.csv',
    'torso_lean': '../../bicep_outputs/torso_lean_angles.csv',
    'wrist_flexion': '../../bicep_outputs/wrist_flexion_angles.csv'
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
                    current_data = row[2:]
                    for i in range(len(current_data)):
                        current_data[i] = float(current_data[i])

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

def get_distance(test_file, train_file, test_index, index_range, r):
    return dtw.distance(test_file[test_index], train_file[index_range[r]], 
                                          window=30, use_pruning=True)


def knn_classifier(k: int, file_idx: int):
    labels = {0: 'Correct', 1: 'Incorrect'}
    angle_types = list(csv_paths.keys())
    num_files = len(train_files)
    indexes = range(0, num_files)

    dtw_distances = {
        'curl': [],
        'UAT': [],
        'TL': [],
        'WF': []
    }

    nearest_ns = {}
    counters = {}
    
    for r in range(num_files):
        dtw_distances['curl'].append(get_distance(x_test_1, x_train_1, file_idx, indexes, r))
        dtw_distances['UAT'].append(get_distance(x_test_2, x_train_2, file_idx, indexes, r))
        dtw_distances['TL'].append(get_distance(x_test_3, x_train_3, file_idx, indexes, r))
        dtw_distances['WF'].append(get_distance(x_test_4, x_train_4, file_idx, indexes, r))
    
    for angle in dtw_distances:
        print(dtw_distances[angle])

    for angle in dtw_distances.keys():
        nearest_ns[angle] = sorted(range(len(dtw_distances[angle])), 
                            key=lambda i: dtw_distances[angle][i], reverse=False)[:k]
    
    print(nearest_ns)
    for l in labels.values():
            counters[l] = 0
        
    for angle in nearest_ns:
        max_val = 0
        for r in nearest_ns[angle]:
            l = labels[y_train[r]]
            counters[l] += 1
            if (counters[l] > max_val):
                max_val = counters[l]
    
    keys = [k for k in counters if counters[k] == max_val]
    
    return(sample(keys, 1)[0])

# for idx in range(len(test_files)):
#     f1_good, f1_bad, f2_good, f2_bad, f3_good, f3_bad, f4_good, f4_bad = [[] for i in range(8)]
    
#     for i in range(len(train_files)):
#         dist1 = DTWDistance(x_test_1[idx], x_train_1[i])
#         dist2 = DTWDistance(x_test_2[idx], x_train_2[i])
#         dist3 = DTWDistance(x_test_3[idx], x_train_3[i])
#         dist4 = DTWDistance(x_test_4[idx], x_train_4[i])

#         if y_train[i] == 1:
#             f1_good.append(dist1)
#             f2_good.append(dist2)
#             f3_good.append(dist3)
#             f4_good.append(dist4)

#         else:
#             f1_bad.append(dist1)
#             f2_bad.append(dist2)
#             f3_bad.append(dist3)
#             f4_bad.append(dist4)

#     good_score = np.mean(f1_good) + np.mean(f2_good) + np.mean(f3_good) + np.mean(f4_good)
#     bad_score = np.mean(f1_bad) + np.mean(f2_bad) + np.mean(f3_bad) + np.mean(f4_bad)
    
#     if good_score < bad_score:
#         predictions.append(0)
#     else:
#         predictions.append(1)

# print(predictions)
# print(y_test)


# print(classification_report(y_test, predictions, target_names=['correct', 'incorrect']))