# Fraud Detection System Using Machine Learning and AWS

## Overview
This project builds a fully automated, cloud-deployable fraud detection system that handles extreme class imbalance using advanced resampling techniques, implements logging and monitoring through MLFlow integrated with automated hyperparameter tuning using Optuna.  It deploys the trained model on AWS using a production-grade, idempotent infrastructure scripts.

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

## Results and Evaluations

This system is designed for high-precision fraud detection in the presence of **extreme class imbalance** — only **0.17%** of the dataset consists of fraud cases. The final pipeline combines resampling, threshold tuning, and ensemble voting to maximize detection performance while minimizing false alarms.

### Generalisability of Individual Models

The following resampling methods produced robust base classifiers, tested for generalisation under severe imbalance:

| Sampling Method     | Non-Fraud Accuracy | Fraud Recall | False Positive Rate | False Negative Rate |
|---------------------|--------------------|--------------|----------------------|----------------------|
| SMOTE               | 99.9%              | 81.7%        | < 0.1%               | 18.3%                |
| SMOTE-Tomek Links   | 99.9%              | 81.7%        | < 0.1%               | 18.3%                |
| Random Oversampling | 99.9%              | **85.0%**    | < 0.1%               | **15.0%**            |

> All three models are effective at correctly identifying legitimate transactions (non-fraud) and produce **very low false positives**.  
> **Random Oversampling** achieved the **highest fraud recall** and **lowest false negative rate**.

### Threshold Optimization

Precision–recall trade-offs vary with threshold. Each model was tuned to achieve its best operating point:

| Sampling Method     | Optimal Threshold | Precision | Recall |
|---------------------|-------------------|-----------|--------|
| SMOTE               | 0.55              | 74%       | 82%    |
| SMOTE-Tomek Links   | 0.60              | 79%       | 82%    |
| Random Oversampling | 0.80              | 70%       | **85%**|

### Ensemble Performance

Final fraud detection is driven by a **voting-based ensemble** composed of the most generalisable base models. Evaluation used a **fraud threshold of 0.7**.

#### Summary Table: Soft vs. Hard Voting Ensembles

| Metric                   | Soft Voting Ensemble | Hard Voting Ensemble |
|--------------------------|----------------------|-----------------------|
| **Precision (Fraud)**    | **0.82**             | 0.77                  |
| **Recall (Fraud)**       | 0.82                | **0.82**               |
| **F1-Score (Fraud)**     | **0.82**             | 0.79                  |
| **Macro Avg F1-Score**   | **0.91**             | 0.89                  |
  
> **Soft voting** delivers better **fraud precision** and **F1-score**, making it the preferred choice for deployment to minimize false alarms.

## Inference System and Optimization

This project includes a container-ready, production-grade **inference system** that serves the best-performing fraud model (XGBoost + SMOTE) using a lightweight **FastAPI** backend.

The deployed model achieves a **PR-AUC of 0.8253** and uses the following tuned hyperparameters:

```json
{
  "colsample_bytree": 0.8,
  "gamma": 2,
  "learning_rate": 0.1,
  "max_depth": 7,
  "min_child_weight": 2,
  "n_estimators": 200,
  "reg_alpha": 2,
  "reg_lambda": 100,
  "scale_pos_weight": 0.15,
  "subsample": 0.8
}
```
### Simulation-Based Inference Testing

To ensure robustness and WAN-class readiness, the inference system was subjected to two phases of simulation using a local loopback **FastAPI** deployment and synthetic WAN noise (**60 ± 20 ms latency**, **3% packet loss**, hot reload every **20s**).

#### Goals
- Simulate SageMaker endpoint conditions from Australia  
- Test model reloads under live traffic  
- Validate thread safety, retry logic, and concurrency behavior  
- Quantify latency, throughput, and tail risk  

### Simulation Phase 1: Burst-Join Driver (8,800 requests)

| Metric               | Value      |
|----------------------|------------|
| Total requests       | 8,800      |
| Successful (HTTP 200)| 100%       |
| Mean latency         | 0.553 sec  |
| Median latency       | 0.472 sec  |
| 95th percentile      | 1.287 sec  |
| Max latency          | 4.159 sec  |
| Min latency          | 0.365 sec  |
| Effective throughput | 10.4 req/s |

> Model hot reloads occurred every 20 seconds with **zero disruption to inference flow**.

### Simulation Phase 2: Optimized Fire-and-Forget Driver (29,999 requests)

| Metric               | Before Optimization | After Optimization  | Δ-Factor |
|----------------------|----------------------|----------------------|----------|
| Mean latency         | 0.553 sec            | **0.104 sec**        | 5.3×     |
| Median latency       | 0.472 sec            | **0.104 sec**        | 4.5×     |
| 95th percentile      | 1.287 sec            | **0.175 sec**        | 7.4×     |
| Max latency          | 4.159 sec            | **0.322 sec**        | 12.9×    |
| Total requests       | 8,800                | 30,000               | 3.4×     |
| Effective throughput | 10.4 req/s           | **288–290 req/s**    | 28×      |

> The final configuration processed **30K WAN-class requests with 0% failure** and **p95 latency of 175 ms**, comfortably meeting real-time fraud-check SLAs.

### Key Optimizations (Client-Side)

| Optimization                      | Impact                                 |
|----------------------------------|----------------------------------------|
| Removed backoff delay            | Eliminated 50–100 ms wait per call     |
| Persistent HTTP sessions         | Saved ~20 ms per request (TCP reuse)   |
| Pooled concurrency (100 threads) | Overlapped network wait periods        |
| Fire-and-forget request strategy | Removed `.join()` stalls               |

> None of the optimizations altered the **FastAPI/XGBoost backend** — the gains come entirely from improved client orchestration and concurrency.


