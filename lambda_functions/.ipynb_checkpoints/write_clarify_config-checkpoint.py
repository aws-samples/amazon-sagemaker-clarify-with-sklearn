import boto3
import json
import os
from io import StringIO

# Pull ressources
client = boto3.client("s3") #low-level functional API

def write_clarify_config(event, context):
    """Lambda function.

    This function will query the data that triggered the event (transformed data),
    then runs this data against the prediction endpoint and returns the prediction.

    Args:
        event (dict): A dict sent from the trigger.
        context (dict): A dict.

    Returns:
        dict: dict.
    """
    print(event)
    bucket = event["bucket"]
    config = {
      "dataset_type": "text/csv",
      "headers": event["header"],
      "label": event["label"],
      "methods": {
        "shap": {
          "baseline": "s3://{}/{}/baseline.csv".format(bucket, event["data_prefix"]),
          "num_samples": 20,
          "agg_method": "mean_abs",
          "use_logit": False,
          "save_local_shap_values": True
        },
        "report": {
          "name": "report",
          "title": "Analysis Report"
        }
      },
      "predictor": {
        "model_name": event["model_name"],
        "instance_type": "ml.m5.xlarge",
        "initial_instance_count": 1,
        "accept_type": "text/csv",
        "content_type": "text/csv"
      }
    }
    clarify_config_file = "{}/analysis_config.json".format(event["clarify_config_prefix"])
    # Write JSON to Amazon S3:
    response = client.put_object(
        Body=json.dumps(config),
        Bucket=bucket,
        Key=clarify_config_file
    )
    # Return prediction on success!
    return {
        "statusCode": 200,
        "body": json.dumps("Job successful!")
    }