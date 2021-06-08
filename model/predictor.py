from __future__ import print_function

import time
import sys
from io import StringIO
import os
import json
import shutil

import argparse
import json
import numpy as np
import pandas as pd
import joblib

from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder, FunctionTransformer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Hyperparameters are described here
    # In this simple example only 4 Hyperparamters are permitted
    parser.add_argument('--n_estimators', type=int, default=100)
    parser.add_argument('--max_depth', type=int, default=10)
    parser.add_argument('--max_features', type=str, default='sqrt')
    parser.add_argument('--random_state', type=int, default=42)
    # SageMaker specific arguments. Defaults are set in the environment variables.
    parser.add_argument('--model_dir', type=str, default=os.environ.get('SM_MODEL_DIR'))
    parser.add_argument('--train', type=str, default=os.environ['SM_CHANNEL_TRAIN'])
    parser.add_argument('--validation', type=str, default=os.environ['SM_CHANNEL_VALIDATION'])
    parser.add_argument('--config', type=str, default=os.environ['SM_CHANNEL_CONFIG'])
    args = parser.parse_args()

    # Take the set of files and read them all into a single pandas dataframe
    train_files = [os.path.join(args.train, file) for file in os.listdir(args.train)]
    val_files = [os.path.join(args.validation, file) for file in os.listdir(args.validation)]
    config_files = [os.path.join(args.config, file) for file in os.listdir(args.config)]
    if len(train_files) == 0:
        raise ValueError(('There are no files in {}.\n' +
                          'This usually indicates that the channel ({}) was incorrectly specified,\n' +
                          'the data specification in S3 was incorrectly specified or the role specified\n' +
                          'does not have permission to access the data.').format(args.train, 'train'))
    if len(val_files) == 0:
        raise ValueError(('There are no files in {}.\n' +
                          'This usually indicates that the channel ({}) was incorrectly specified,\n' +
                          'the data specification in S3 was incorrectly specified or the role specified\n' +
                          'does not have permission to access the data.').format(args.validation, 'validation'))
    if len(config_files) == 0:
        raise ValueError(('There are no files in {}.\n' +
                          'This usually indicates that the channel ({}) was incorrectly specified,\n' +
                          'the data specification in S3 was incorrectly specified or the role specified\n' +
                          'does not have permission to access the data.').format(args.config, 'config'))
        
    # Read DataFrames into array and concatenate them into one DF
    train_data = [pd.read_csv(file) for file in train_files]
    train_data = pd.concat(train_data)
    val_data = [pd.read_csv(file) for file in val_files]
    val_data = pd.concat(val_data)
    config_data = {}
    for file in config_files:
        with open(file) as json_file:
            config_data.update(json.load(json_file))
            
    # Set train and validation
    X_train, y_train = train_data.iloc[:, 1:], train_data.iloc[:, 0]
    X_test, y_test = val_data.iloc[:, 1:], val_data.iloc[:, 0]
    
    # Define preprocessor
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', FunctionTransformer(), config_data['numeric']),
            ('cat', OneHotEncoder(handle_unknown='ignore'), config_data['one_hot_encoding'])
        ]
    )

    # Define model
    rfr = RandomForestRegressor(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth if args.max_depth > 0 else None,
        max_features=args.max_features,
        random_state=args.random_state if args.random_state > 0 else None,
        n_jobs=-1)

    # Create model pipeline and fit
    model = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('regressor', rfr)
        ]
    ).fit(X_train, y_train)
    
    # Monitor metrics
    train_metric = model.predict(X_train)
    val_metric = model.predict(X_test)
    train_mae = mean_absolute_error(y_train, train_metric)
    val_mae = mean_absolute_error(y_test, val_metric)
    print('Train_mae={};'.format(train_mae))
    print('Validation_mae={};'.format(val_mae))
    
    # Save the model and config_data to the model_dir so that it can be loaded by model_fn
    # Save model
    joblib.dump(model, os.path.join(args.model_dir, 'model.joblib'))
    # save the config to the model path to use the header when predicting in predict_fn
    with open(os.path.join(args.model_dir, 'config_data.json'), 'w') as outfile:
        json.dump(config_data, outfile)
        
    # Print Success
    print('Saved model!')

def input_fn(input_data, content_type='text/csv'):
    """Parse input data payload.

    Args:
        input_data (pandas.core.frame.DataFrame): A pandas.core.frame.DataFrame.
        content_type (str): A string expected to be 'text/csv'.

    Returns:
        df: pandas.core.frame.DataFrame
    """
    try:
        if 'text/csv' in content_type == 'text/csv':
            df = pd.read_csv(StringIO(input_data), header=None)
            return df
        elif 'application/json' in content_type:
            df = pd.read_json(StringIO(input_data.decode('utf-8')))
            return df
        elif 'text/html' in content_type:
            df = pd.read_csv(StringIO(input_data.decode('utf-8')), header=None)
            return df
        else:
            df = pd.read_csv(StringIO(input_data.decode('utf-8')), header=None)
            return df
    except:
        raise ValueError(f'{content_type} not supported by script!')

def output_fn(prediction, accept='text/csv'):
    """Format prediction output.

    Args:
        prediction (pandas.core.frame.DataFrame): A DataFrame with predictions.
        accept (str): A string expected to be 'text/csv'.

    Returns:
        df: str (in CSV format)
    """
    df = prediction.to_csv(index=False, header=None)
    return df

def predict_fn(input_data, model):
    """Preprocess input data.

    Args:
        input_data (pandas.core.frame.DataFrame): A pandas.core.frame.DataFrame.
        model (sklearn.ensemble.RandomForestRegressor): A regression model

    Returns:
        output: pandas.core.frame.DataFrame
    """
    # Read your model and config file
    trained_model, config_data = model
    # This is fixing the issue of having columns in training job vs. Clarify expecting to input
    input_data.columns = config_data['header'][1:]
    output = trained_model.predict(input_data)
    return pd.DataFrame(output)

def model_fn(model_dir):
    """Deserialize fitted model.
    
    This simple function takes the path of the model, loads it,
    deserializes it and returns it for prediction.

    Args:
        model_dir (str): A string that indicates where the model is located.

    Returns:
        model: sklearn.ensemble.RandomForestRegressor
    """
    # Load the model and deserialize
    model = joblib.load(os.path.join(model_dir, 'model.joblib'))
    config_file = os.path.join(model_dir, 'config_data.json')
    with open(config_file) as json_file:
        config_data = json.load(json_file)
    return model, config_data