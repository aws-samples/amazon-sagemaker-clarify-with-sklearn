# ### Source the libraries
# 
# First we install the AWS StepFunctions Data Science Python SDK. And then we source in all libraries required to run this notebook.


import sys
get_ipython().system('{sys.executable} -m pip install stepfunctions')

import sagemaker
from sagemaker.image_uris import retrieve
from sagemaker.sklearn.estimator import SKLearn
from sagemaker.processing import Processor, ProcessingInput, ProcessingOutput
from sagemaker.dataset_definition.inputs import S3Input
from sagemaker.inputs import TrainingInput
from sagemaker.s3 import S3Uploader
import stepfunctions
from stepfunctions import steps
from stepfunctions.inputs import ExecutionInput
from stepfunctions.workflow import Workflow

import logging
import time
import boto3
import json
import pandas as pd
from sklearn.model_selection import train_test_split

# ### Set S3 bucket, data prefix and permission
# 
# Here the user should set all the required variables that will be used throughout the notebook:

# the bucket with the project/use case data
S3_BUCKET = 'sagemaker-clarify-demo' # YOUR_S3_BUCKET
# the project/use case prefix
PREFIX = "abalone-clarify-statemachine" # "YOUR_PREFIX"
# the perfix for the training, validation, baseline data for the ML model and clarify
DATA_PREFIX = f'{PREFIX}/prepared_data'
# the step functions execution role
account = boto3.client('sts').get_caller_identity().get('Account')
WORKFLOW_EXEC_ROLE = "arn:aws:iam::{}:role/sagemaker-clarify-demo-StepFunctionsRole".format(account)
STEPFUNCTION_NAME = "abalone-training" # "YOUR_STEPFUNCTION_NAME"
# the target/label column of the data 
TARGET_COL = "Rings" # "YOUR_TARGET_COL"
# the prefix to contain the clarify config file 
CLARIFY_CONFIG_PREFIX = "{}/clarify-explainability".format(PREFIX) # "CLARIFY_CONFIG_PREFIX"
# flag to either generate the training/validation/baseline/config data for the state machine
generate_data = True

# define the required inputs for the SKLearn Base estimator
framework_version = '0.23-1'
instance_type = 'ml.m5.xlarge'
instance_count = 1
# Training job hyperparameters
hyperparameters = {
    'n_estimators': 100,
    'max_depth': 10,
    'max_features': 'sqrt',
    'random_state': 42
}

# This cell will compile your entry code for the Amazon SageMaker Training job
# and then store it back in your model/ folder
!cp model/predictor.py ./
!tar -czvf model.tar.gz predictor.py
!mv predictor.py model/predictor.py
!mv model.tar.gz model/model.tar.gz

# ## Load and upload the data
# 
# Generate the training and validation data and data config the regression model and generate the baseline data for the clarify explainability job.

if generate_data:
    target_col = 'Rings'
    df = pd.read_csv('https://datahub.io/machine-learning/abalone/r/abalone.csv')
    cols = [x if "rings" not in x else "Rings" for x in df.columns]
    df.columns = cols
    # X: input features - this is what your algorithm takes for learning
    # y: target - this is what your algorithm will predict
    X = df.drop(TARGET_COL, axis=1)
    y = df[TARGET_COL]
    # Split the data into 70% training and 30% validation data
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3, random_state=42)
    train = pd.concat([y_train, X_train], axis=1)
    val = pd.concat([y_val, X_val], axis=1)
    # save the data locally and then upload to S3
    train.to_csv('train_data.csv', index=False) # training data
    val.to_csv('val_data.csv', index=False) # validation data
    baseline = train.agg({'Sex': 'mode', 
                      'Length': 'mean', 
                      'Diameter': 'mean', 
                      'Height': 'mean', 
                      'Whole_weight': 'mean', 
                      'Shucked_weight': 'mean', 
                      'Viscera_weight': 'mean', 
                      'Shell_weight': 'mean'}) # used in SageMaker Clarify: only store your features
    baseline.to_csv('baseline.csv', index=False, header=None)
    # write the columns to be one hot encoded and the column names, in the same order as in the training data into a config file
    # this config file will be read during training and prediction to one hot encode the columns 
    config_data = {
        'one_hot_encoding': ['Sex'], 
        'numeric': ['Length', 'Diameter', 'Height', 'Whole_weight', 'Shucked_weight', 'Viscera_weight', 'Shell_weight'],
        'header': train.columns.tolist()
    }
    with open('config_data.json', 'w') as outfile:
        json.dump(config_data, outfile)
    # upload the data to S3
    train_uri = S3Uploader.upload('train_data.csv', f's3://{S3_BUCKET}/{DATA_PREFIX}')
    val_uri = S3Uploader.upload('val_data.csv', f's3://{S3_BUCKET}/{DATA_PREFIX}')
    baseline_uri = S3Uploader.upload('baseline.csv', f's3://{S3_BUCKET}/{DATA_PREFIX}')
    config_uri = S3Uploader.upload('config_data.json', f's3://{S3_BUCKET}/{DATA_PREFIX}')
    model_uri = S3Uploader.upload('model/model.tar.gz', f's3://{S3_BUCKET}/{DATA_PREFIX}')

# ## Create the pipeline
# 
# ### Set training session parameters
# 
# In this section you will set the data source for the model to be run, as well as the Amazon SageMaker SDK session variables.

# Get a SageMaker-compatible role used by this function and the session.
sagemaker_session = sagemaker.Session()
region = sagemaker_session.boto_region_name
role = sagemaker.get_execution_role()
# get the clarify image_uri for a processor job to run clarify
clarify_image_uri = retrieve('clarify', region)
stepfunctions.set_stream_logger(level=logging.INFO)

# Train a SKLearn estimator 
# Set your code folder
source_dir = model_uri
#source_dir = "s3://{}/{}/model/model.tar.gz".format(S3_BUCKET, PREFIX)
entry_point = 'predictor.py'

# training input path
train_uri = "s3://{}/{}/train_data.csv".format(S3_BUCKET, DATA_PREFIX)
# validation input path
val_uri = "s3://{}/{}/val_data.csv".format(S3_BUCKET, DATA_PREFIX)
# baseline (required for Clarify) input path
baseline_uri = "s3://{}/{}/baseline.csv".format(S3_BUCKET, DATA_PREFIX)
# data config input path
config_data_uri = "s3://{}/{}/config_data.json".format(S3_BUCKET, DATA_PREFIX)
# clarify config input path
clarify_config_uri = "s3://{}/{}/analysis_config.json".format(S3_BUCKET, CLARIFY_CONFIG_PREFIX)
# explainability output path
explainability_output_uri = "s3://{}/{}/".format(S3_BUCKET, CLARIFY_CONFIG_PREFIX)

# Set training input for the SKlearn estimator (not necessary but recommended)
train_data = TrainingInput(train_uri, content_type='csv')
validation_data = TrainingInput(val_uri, content_type='csv')
config_data = TrainingInput(config_data_uri, content_type='json')

s3_client = boto3.client('s3')
result = s3_client.get_object(Bucket=config_data_uri.split('/')[2], Key='/'.join(config_data_uri.split('/')[3:])) 
config_dict = json.loads(result['Body'].read().decode('utf-8'))

# ### Set SKLearn container
model = SKLearn(
    entry_point=entry_point,
    source_dir=source_dir,
    hyperparameters=hyperparameters,
    role=role,
    instance_count=instance_count,
    instance_type=instance_type,
    framework_version=framework_version,
    sagemaker_session=sagemaker_session,
    code_location=f's3://{S3_BUCKET}/{PREFIX}/model/',
    output_path=f's3://{S3_BUCKET}/{PREFIX}/model/',
    enable_sagemaker_metrics=True,
    metric_definitions=[
        {
            'Name': 'train:mae',
            'Regex': 'Train_mae=(.*?);'
        },
        {
            'Name': 'validation:mae',
            'Regex': 'Validation_mae=(.*?);'
        }
])

# ### Define Amazon SageMaker Clarify processing container
# 
# In this section we will define the processing container. Amazon SageMaker Clarify runs as a Amazon SageMaker processing job. For this reason we will be integrating a processing job into the AWS StepFunction.

# define the clarify processor inputs and outputs
inputs = [
    ProcessingInput(
        input_name="dataset",
        app_managed=False,
        s3_input=S3Input(
            s3_uri=train_uri,
            local_path="/opt/ml/processing/input/data",
            s3_data_type="S3Prefix",
            s3_input_mode="File",
            s3_data_distribution_type="FullyReplicated",
            s3_compression_type="None"
        )
    ),
    ProcessingInput(
        input_name="analysis_config",
        app_managed=False,
        s3_input=S3Input(
            s3_uri=clarify_config_uri,
            local_path="/opt/ml/processing/input/config",
            s3_data_type="S3Prefix",
            s3_input_mode="File",
            s3_data_distribution_type="FullyReplicated",
            s3_compression_type="None"
        )
    )
]
outputs = [
    ProcessingOutput(
        source="/opt/ml/processing/output",
        destination=explainability_output_uri,
        output_name="analysis_result",
        s3_upload_mode="EndOfJob",
    )
]
# define the clarify processor
processor = Processor(
    image_uri=clarify_image_uri,
    role=role,
    instance_type=instance_type,
    instance_count=1,
    sagemaker_session=sagemaker_session)

# #### Define input schema

# define the execution input schema
schema = {
    "TrainingJobName": str,
    "ModelName": str,
    "EndpointName": str,
    "ClarifyWriteConfigLambda": str,
    "ClarifyJobName": str,
}

# define the execution input, which needs to be passed in this format to the state machine
execution_input = ExecutionInput(schema=schema)

# #### Define AWS StepFunction steps

# Set the model training step
train_step = steps.TrainingStep(
    'Model Training',
    estimator=model,
    data={
        'train': train_data, 
        'validation': validation_data,
        'config': config_data
    },
    job_name=execution_input['TrainingJobName'],
    wait_for_completion=True
)

# Save the model to Sagemaker
model_step = steps.ModelStep(
    'Save SageMaker Model',
    model=train_step.get_expected_model(),
    model_name=execution_input['ModelName']
)

# SageMaker Clarify config
lambda_step = steps.compute.LambdaStep(
    'Write Clarify config file',
    parameters={  
        "FunctionName": execution_input['ClarifyWriteConfigLambda'],
        'Payload': {
            "bucket": S3_BUCKET,
            "data_prefix": DATA_PREFIX,
            "model_name": execution_input['ModelName'],
            "label": TARGET_COL,
            "header": config_dict['header'],
            "clarify_config_prefix": CLARIFY_CONFIG_PREFIX
        }
    }
)

# SageMaker Clarify
clarify_processing = steps.ProcessingStep(
    'SageMaker Clarify Processing',
    processor=processor,
    job_name=execution_input['ClarifyJobName'],
    inputs=inputs,
    outputs=outputs,
    wait_for_completion=True
)

# Create endpoint config for model
endpoint_config = steps.EndpointConfigStep(
    "Create SageMaker Endpoint Config",
    endpoint_config_name=execution_input['ModelName'],
    model_name=execution_input['ModelName'],
    initial_instance_count=instance_count,
    instance_type=instance_type
)

# Create/Update model endpoint
endpoint = steps.EndpointStep(
    'Create/Update SageMaker Endpoint',
    endpoint_name=execution_input['EndpointName'],
    endpoint_config_name=execution_input['ModelName'],
    update=False
)

# Chain together the steps for the state machine
workflow_definition = steps.Chain([
    train_step,
    model_step,
    lambda_step,
    clarify_processing,
    endpoint_config,
    endpoint
])

# Define the Workflow
workflow = Workflow(
    name=STEPFUNCTION_NAME,
    definition=workflow_definition,
    role=WORKFLOW_EXEC_ROLE,
    execution_input=execution_input
)

# Create the workflow
workflow.create()

workflow.update(definition=workflow_definition, role=WORKFLOW_EXEC_ROLE)
time.sleep(10)

gid = time.strftime("%Y-%m-%d-%H-%M-%S", time.gmtime())
inputs = {
    "TrainingJobName": "sagemaker-sklearn-job-{}".format(gid),
    "ModelName": "sagemaker-sklearn-job-{}".format(gid),
    "EndpointName": "sagemaker-sklearn-job-{}".format(gid),
    "ClarifyWriteConfigLambda": "sagemaker-clarify-write-config",
    "ClarifyJobName": "sagemaker-sklearn-job-{}".format(gid),
}

execution = workflow.execute(inputs=inputs)