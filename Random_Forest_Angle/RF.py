import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score
from sklearn import utils
from sklearn import preprocessing
import pickle
import os

# Retrieve all bicep file names
FILE_PATH = '../../MP/test_vids'
FILES = os.listdir(FILE_PATH)

def drop_col(df, col: int):
    result = df.iloc[:,col:]
    return result

def add_frame_headers(df, col_title):
    cols = len(df.columns)
    frames = [f'{col_title}{i}' for i in range(1, cols+1)]
    df.columns = frames

    return df

def get_features(file, col_title):
    df = pd.read_csv(file, index_col=0, header=None)
    cols = len(df.columns)
    frames = [f'{col_title}{i}' for i in range(0, cols)]
    frames[0] = 'target'
    
    df.columns = frames
    return df


def main():

    # curl_df = pd.read_csv('../../bicep_outputs/curl2.csv')
    # torso_lean_df = pd.read_csv('../../bicep_outputs/torso_lean2.csv')
    # upper_arm_df = pd.read_csv('../../bicep_outputs/upper_arm2.csv')
    # wrist_flexion_df = pd.read_csv('../../bicep_outputs/wrist_flexion2.csv')

    
    # Separating features and target (x, y)

    x_curl_df = get_features('../../bicep_outputs/curl2.csv', 'c')
    x_upper_arm_df = get_features('../../bicep_outputs/upper_arm2.csv', 'ua')
    x_torso_lean_df = get_features('../../bicep_outputs/torso_lean2.csv', 'tl')
    x_wrist_flexion_df = get_features('../../bicep_outputs/wrist_flexion2.csv', 'wf')

    curl_features = drop_col(x_curl_df, 1)
    upper_arm_features = drop_col(x_upper_arm_df, 1)
    torso_lean_features = drop_col(x_torso_lean_df, 1)
    wrist_flexion_features = drop_col(x_wrist_flexion_df, 1)

    y_targets = x_curl_df['target'].values.tolist()

    x_df = pd.concat([curl_features, upper_arm_features, torso_lean_features, wrist_flexion_features], axis=1)


    lab = preprocessing.LabelEncoder()
    y_trans = lab.fit_transform(y_targets)
    print(y_trans)

    x_train, x_test, y_train, y_test = train_test_split(x_df, y_trans, test_size=0.4, random_state=1234)
    

    pipelines = {
        'hg': make_pipeline(StandardScaler(), HistGradientBoostingClassifier()),
    }

    fit_models = {}

    for algo, pipeline in pipelines.items():
        model = pipeline.fit(x_train, y_train)
        fit_models[algo] = model
        print(f"done with {pipeline}")

    for algo, model in fit_models.items():
        yhat = model.predict(x_test)
        print(algo, accuracy_score(y_test, yhat))


    # with open('rf_class.pkl', 'wb') as f:
    #     pickle.dump(fit_models['rf'], f)
    # with open('lr_class.pkl', 'wb') as f:
    #     pickle.dump(fit_models['lr'], f)

    

main()