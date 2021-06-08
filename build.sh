#!/bin/bash

cd /home/ec2-user/SageMaker/amazon-sagemaker-clarify-with-sklearn/model
tar -czvf model.tar.gz *.py

cd ..
python3 model_training_pipeline.py