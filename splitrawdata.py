import os
import pandas as pd
from sklearn.model_selection import train_test_split
import lightgbm as lgb

DATA_PATH = "/data/kondratov/data"
DATA_NUMBER = 1

SIMPLE_FEATURE_COLUMNS = ['ncl[0]', 'ncl[1]', 'ncl[2]', 'ncl[3]', 'avg_cs[0]', 'avg_cs[1]',
    'avg_cs[2]', 'avg_cs[3]', 'MatchedHit_TYPE[0]', 'MatchedHit_TYPE[1]',
    'MatchedHit_TYPE[2]', 'MatchedHit_TYPE[3]', 'MatchedHit_X[0]',
    'MatchedHit_X[1]', 'MatchedHit_X[2]', 'MatchedHit_X[3]',
    'MatchedHit_Y[0]', 'MatchedHit_Y[1]', 'MatchedHit_Y[2]',
    'MatchedHit_Y[3]', 'MatchedHit_Z[0]', 'MatchedHit_Z[1]',
    'MatchedHit_Z[2]', 'MatchedHit_Z[3]', 'MatchedHit_T[0]',
    'MatchedHit_T[1]', 'MatchedHit_T[2]', 'MatchedHit_T[3]',
    'MatchedHit_DX[0]', 'MatchedHit_DX[1]', 'MatchedHit_DX[2]',
    'MatchedHit_DX[3]', 'MatchedHit_DY[0]', 'MatchedHit_DY[1]',
    'MatchedHit_DY[2]', 'MatchedHit_DY[3]', 'MatchedHit_DZ[0]',
    'MatchedHit_DZ[1]', 'MatchedHit_DZ[2]', 'MatchedHit_DZ[3]',
    'MatchedHit_DT[0]', 'MatchedHit_DT[1]', 'MatchedHit_DT[2]',
    'MatchedHit_DT[3]', 'Lextra_X[0]', 'Lextra_X[1]', 'Lextra_X[2]',
    'Lextra_X[3]', 'Lextra_Y[0]', 'Lextra_Y[1]', 'Lextra_Y[2]',
    'Lextra_Y[3]', 'Mextra_DY2[0]', 'Mextra_DY2[1]', 'Mextra_DY2[2]',
    'Mextra_DY2[3]', 'Mextra_DX2[0]', 'Mextra_DX2[1]', 'Mextra_DX2[2]',
    'Mextra_DX2[3]', 'P', 'PT', ]

def split_raw_data(path='muon_light_transformed_weight.hdf', random_state=None, downsample=True):
    data = pd.read_hdf(os.path.join(DATA_PATH, path), random_state=random_state)
    if downsample:
        false_data = data.loc[data.label == 0]
        true_data = data.loc[data.label == 1]
        data = (false_data.append(true_data.sample(len(false_data)))).sample(frac=1)

    train_data, test = train_test_split(data, test_size=0.2, random_state=random_state)
    test.to_csv(os.path.join(DATA_PATH, 'transformed_test.csv.gz'),
                compression='infer')
    test.loc[:, SIMPLE_FEATURE_COLUMNS].to_csv(os.path.join(DATA_PATH,
                                               'transformed_test_features.csv'),
                                               compression='infer')
    del test
    del data
    train, lr_train = train_test_split(train_data, test_size=0.25, random_state=random_state)
    lr_train.to_csv(os.path.join(DATA_PATH, 'lr_train.csv.gz'),
                    compression='infer')
    del lr_train
    train.to_csv(os.path.join(DATA_PATH, 'transformed_train_1.csv.gz'),
                 compression='infer')

def transform_lgb():
    lr_train = pd.read_csv(os.path.join(DATA_PATH, 'lr_train.csv.gz'), compression='infer')
    train = pd.read_csv(os.path.join(DATA_PATH, 'transformed_train_1.csv.gz'), compression='infer')
    test = pd.read_csv(os.path.join(DATA_PATH, 'transformed_test.csv.gz'), compression='infer')
    lgb_train_lr = lgb.Dataset(lr_train.loc[:, SIMPLE_FEATURE_COLUMNS],
                               lr_train.label,
                               weight=lr_train.weight,
                               free_raw_data=False)
    lgb_train_lr.save_binary(os.path.join(DATA_PATH, 'lr_train.bin'))
    lgb_train = lgb.Dataset(train.loc[:, SIMPLE_FEATURE_COLUMNS],
                            train.label,
                            weight=train.weight,
                            free_raw_data=False)
    lgb_train_lr.save_binary(os.path.join(DATA_PATH, 'transformed_train_1.bin'))
    lgb_test = lgb.Dataset(test.loc[:, SIMPLE_FEATURE_COLUMNS],
                           test.label,
                           weight=test.weight,
                           free_raw_data=False)
    lgb_test.save_binary(os.path.join(DATA_PATH, 'transformed_test.bin'))
