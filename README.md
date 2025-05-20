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

## AWS Deployment
The fraud detection system is deployed on AWS using a **Bash script** that automates the following:

1. **S3 Upload**: Uploads raw data, preprocessed files, and scripts to an S3 bucket.
2. **IAM Role Creation**: Configures roles for Glue and SageMaker.
3. **AWS Glue Processing**: Runs a Glue ETL job to structure and prepare data.
4. **SageMaker Training**: Triggers SageMaker Notebook for model training and evaluation.
5. **Inference Deployment**: Uploads an inference script to SageMaker for real-time fraud detection.

The script includes **status checks** for Glue crawlers and jobs, ensuring they complete successfully before proceeding.

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
spark-submit PySpark_Preprocessing.py
```
Note that this script expects the original dirty credit card dataset `creditcard.csv` to be inside the same directory as the script is localed in.
 Running the this will generate cleaned and scaled **train**, **validation**, and **test** CSV files inside the same directory as the script itself.

### **3. Deploying to AWS**
Execute the Bash script to deploy the entire system:
```bash
chmod +x deploy_fraud_detection.sh
./deploy_fraud_detection.sh
```
This will:
- Upload the data and scripts to S3.
- Set up IAM roles.
- Run AWS Glue and SageMaker services.

## Results and Performance
The **best XGBoost model** was evaluated using multiple metrics:
- **Precision / Recall**: Assesses fraud detection trade-offs.
- **F1-Score**: Harmonic mean of precision and recall.
- **Geometric Mean (G-Mean)**: Measures classifier balance on imbalanced data.

Results are logged and available in **SageMaker Notebook outputs**.

## Repository Structure
```
/
├── data/                                      # Raw and preprocessed datasets
│   ├── creditcard.csv
│   ├── preprocessed_train.csv
│   ├── preprocessed_val.csv
│   ├── preprocessed_test.csv
│
├── scripts/                                   # ML and AWS scripts
│   ├── optimized_fraud_detection_glue_job.py  # PySpark preprocessing script
│   ├── sagemaker_notebook_code.ipynb          # Model training notebook
│   ├── inference.py                           # Model inference script
│   ├── deploy_fraud_detection.sh              # AWS automation script
│
├── README.md                                  # Project documentation
```
