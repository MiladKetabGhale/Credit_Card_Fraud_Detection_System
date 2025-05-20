# Fraud Detection System Using Machine Learning and AWS

## Overview
This project implements an automated **fraud detection system** using **XGBoost-based models** combined with various **sampling techniques** to handle class imbalance. The model is trained after preprocessing the dataset with **PySpark** and deployed on **AWS** using a fully automated Bash script with AWS CLI.

## Project Features
- **Data Preprocessing with PySpark**: Scales, normalizes, and transforms the dataset before training.
- **Multiple Sampling Techniques**: Implements **9 sampling methods**, including **SMOTE variants and under-sampling techniques**, to improve fraud detection.
- **Cross-Validation & Hyperparameter Tuning**: Fine-tunes the XGBoost model for optimal performance.
- **AWS-Based Automated Deployment**: Deploys the fraud detection system using AWS **Glue, SageMaker, S3**, and **IAM** through a fully automated **Bash script**.
- **Automated Model Training & Inference**: Uses SageMaker to train and evaluate the model, then deploys an inference script.

## Architecture
The fraud detection system is structured as follows:

1. **Data Preprocessing (Local, PySpark)**
   - Loads and cleans the dataset using PySpark.
   - Applies log normalization and standardization.
   - Splits the dataset into **train, validation, and test** sets.
   - Saves preprocessed data as CSVs.
   
2. **Model Training (Local, XGBoost)**
   - Applies **nine sampling techniques** to balance fraud vs. non-fraud cases.
   - Trains and tunes an XGBoost model using cross-validation.
   - Evaluates performance using precision, recall, F1-score, and G-Mean.
   
3. **Automated AWS Deployment (Bash + AWS CLI)**
   - Uploads preprocessed data and scripts to an S3 bucket.
   - Sets up **AWS Glue** for ETL processing.
   - Deploys a **SageMaker Notebook** for training and evaluation.
   - Runs an **inference script** on SageMaker for real-time fraud detection.

## Data Preprocessing
The dataset is preprocessed using **PySpark**, applying the following transformations:

- **Log Normalization**: `Amount` column is transformed using `log1p()`.
- **Standardization**: Features `V1 - V28` are standardized using `StandardScaler`.
- **Feature Scaling**: Uses **VectorAssembler** to assemble features before scaling.
- **Splitting**: Dataset is randomly split into **train (80%)**, **validation (10%)**, and **test (10%)** sets.
- **Output**: Preprocessed features and labels are saved as CSV files.

## Model Training
The model training phase applies **9 different resampling techniques** to improve fraud detection:

### **Over-Sampling Methods:**
- **Random Over-Sampling (ROS)**
- **SMOTE** (Synthetic Minority Over-sampling Technique)
- **ADASYN** (Adaptive Synthetic Sampling)
- **Borderline-SMOTE**
- **SMOTE-ENN (Hybrid)**
- **SMOTE-Tomek Links (Hybrid)**

### **Under-Sampling Methods:**
- **Random Under-Sampling (RUS)**
- **Tomek Links**
- **NearMiss**

The final model is an **XGBoost-based classifier**, trained and tuned with **cross-validation** to optimize hyperparameters.

## AWS-Based Deployment (Production-Ready Infrastructure)

The fraud detection pipeline is fully automated and deployable on AWS through robust Bash scripting and the AWS CLI. The deployment infrastructure adheres to security best practices and follows an idempotent setup, ensuring reproducibility and fault tolerance.

### Key AWS Services Used
 - **Amazon S3** – Centralized storage for raw data, preprocessed datasets, scripts, model artifacts, and predictions.
 - **AWS Glue** – Serverless data integration and ETL service for schema inference and transformation.
 - **Amazon SageMaker** – Managed platform for model training, evaluation, and real-time inference.
 - **IAM (Identity and Access Management)** – Manages secure access and fine-grained permissions for Glue and SageMaker components.

### Deployment Workflow: Fully Automated in Two Stages

Deployment is managed via two idempotent Bash scripts:

### 1. `s3_provision_encryption_idempotency.sh`

This script sets up a secure and compliant S3 environment. It ensures repeatable infrastructure provisioning without side effects.

#### Main responsibilities:

 - Create an S3 bucket (if it doesn’t exist)
 - Enforce server-side AES-256 encryption
 - Block all public access (in line with security best practices)
 - Scaffold the required S3 directory topology:

### 2. `automated_deployment_script.sh`

This script handles the orchestration of AWS services. It intelligently checks service statuses, ensures successful transitions between stages, and enables full pipeline automation.

#### Main responsibilities:

   - Upload datasets and code artifacts to designated S3 prefixes
   - Create or validate IAM roles:
   - GlueServiceRole
   - SageMakerExecutionRole
   - Configure and execute a Glue Crawler, polling until completion
   - Trigger a Glue ETL Job for data transformation
   - Upload SageMaker assets:
   - Jupyter notebook for training
   - Python inference script for deployment
   - Launch SageMaker training and evaluation
   - Deploy the trained model as a real-time inference endpoint

## Setup and Usage
### **1. Prerequisites**
Ensure the following are installed:
- Python (>=3.8) with required dependencies
- PySpark
- XGBoost
- AWS CLI (configured with IAM credentials)
- Bash (Linux/Mac) or Git Bash (Windows)

### **2. Running the Preprocessing Pipeline Locally**
Run the following command to preprocess the dataset:
```bash
spark-submit ML_Model_Development/PySpark_Preprocessing.py
```
Note that the script expects the original dirty credit card dataset `creditcard.csv` to be in the same directory as the script is located in.
 Running the this will generate cleaned and scaled **train**, **validation**, and **test** CSV files inside the same directory as the script itself.

## Results and Performance
The **best XGBoost model** was evaluated using multiple metrics:
- **Precision / Recall**: Assesses fraud detection trade-offs.
- **F1-Score**: Harmonic mean of precision and recall.
- **Geometric Mean (G-Mean)**: Measures classifier balance on imbalanced data.

Results are logged and available in **SageMaker Notebook outputs**.
