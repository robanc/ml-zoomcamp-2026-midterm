# 4G/5G Mobile Network Signal Level Prediction

**ML Zoomcamp 2026 – Midterm Project**

A machine learning regression project for predicting 4G/5G mobile network signal levels using geographic, mobility, and network characteristics.

## Table of Contents

1. [Problem Description](#1-problem-description)
2. [Exploratory Data Analysis (EDA)](#2-exploratory-data-analysis-eda)
3. [Model Training](#3-model-training)
4. [Training Script (Exporting Notebook to Script)](#4-training-script-exporting-notebook-to-script)
5. [Reproducibility](#5-reproducibility)
6. [Model Deployment](#6-model-deployment)
7. [Dependencies and Environment](#7-dependencies-and-environment)
8. [Containerization](#8-containerization)
9. [Cloud Deployment](#9-cloud-deployment)

---

## 1. Problem Description

### 1.1 Background

Mobile network performance is influenced by geographic location, network technology, mobile operator, and user mobility. As mobile networks evolve from 4G LTE to 5G, understanding variations in signal levels becomes increasingly important for network planning, optimization, and performance analysis.

Telecommunications engineers routinely analyze network measurements to identify coverage limitations, investigate signal degradation, and evaluate network performance under different mobility conditions.

This project applies machine learning to predict mobile network signal levels using geographic, mobility, and network characteristics.

### 1.2 Problem Statement

The objective is to develop a regression model that predicts the mobile network signal level, expressed in decibel-milliwatts (dBm), based on seven input features.

**Input features:**

| Feature | Description |
|---|---|
| Latitude | Geographic latitude of the measurement |
| Longitude | Geographic longitude of the measurement |
| Speed | Device movement speed |
| Operatorname | Anonymized mobile network operator |
| NetworkTech | Network technology (4G or 5G) |
| Mobility | Mobility scenario (Walk, BRT, or Shuttle) |
| Altitude | Altitude associated with the measurement |

**Target variable:** `Level` — recorded mobile network signal level in dBm.

This is a supervised machine learning regression problem because the target is a continuous numerical value.

### 1.3 Why This Problem Matters

Predicting mobile network signal levels can support several telecommunications engineering activities:

- **Network planning:** Estimate signal conditions across different geographic locations.
- **Network optimization:** Identify areas that may require further coverage investigation.
- **Mobility analysis:** Examine how predicted signal levels vary across walking and transportation scenarios.
- **4G/5G performance analysis:** Compare modeled signal behavior across network technologies.
- **Operational decision support:** Provide estimates that may help prioritize field measurements and network investigations.

The model is intended as an analytical aid rather than a replacement for actual radio-frequency measurements or detailed propagation modeling.

### 1.4 Dataset

The project uses a publicly available mobile network measurement dataset hosted on Mendeley Data.

**Dataset source:** https://data.mendeley.com/datasets/dx5xyyfz2y/2

The measurements were collected in an urban environment near Sunway University, Malaysia.

**Dataset characteristics:**

| Attribute | Value |
|---|---|
| Total observations | 30,925 |
| Total columns | 51 |
| Measurement sessions | 23 |
| Network technologies | 4G and 5G |
| Mobile operators | Three anonymized operators |
| Mobility scenarios | Walk, BRT, Shuttle |
| Prediction target | `Level` (dBm) |
| Selected input features | 7 |

The dataset includes multiple measurements within individual collection sessions. To reduce the risk of data leakage, the project uses session-aware splitting so that measurements from the same session do not appear across training, validation, and test partitions.

### 1.5 Expected Outcome

The expected outcome is a trained regression model that:

1. Accepts geographic, mobility, and network information.
2. Predicts the corresponding mobile network signal level in dBm.
3. Provides predictions through a REST API.
4. Can be reproduced and deployed using Python and Docker.

Model performance is evaluated using Root Mean Squared Error (RMSE) and Mean Absolute Error (MAE), both expressed in dB.

The final solution uses a tuned XGBoost regression model, with a held-out test RMSE of **9.1644 dB** and MAE of **6.9597 dB**.




## 2. Exploratory Data Analysis (EDA)

### 2.1 Dataset Overview

Exploratory Data Analysis was performed in `notebook.ipynb` to understand the structure, quality, and characteristics of the mobile network measurement dataset.

The dataset contains:

- **30,925 observations**
- **51 columns**
- **23 measurement sessions**
- Measurements from 4G and 5G networks
- Three anonymized mobile network operators
- Three mobility scenarios: Walk, BRT, and Shuttle

The prediction target is `Level`, representing the recorded signal level in dBm.

### 2.2 Data Quality and Missing Values

The dataset was examined for missing values, feature types, and variables with limited predictive information.

Two features were excluded from the final model:

| Feature | Observation | Decision |
|---|---|---|
| ARFCN | Constant observed value of 9310 | Excluded |
| BANDWIDTH | Constant observed value of 20, with 1,162 missing entries | Excluded |

Other signal measurement variables were also excluded from the input features to avoid relying on potentially unavailable measurements or closely related signal metrics when predicting `Level`.

### 2.3 Target Variable Analysis

The target variable, `Level`, is a continuous numerical measurement expressed in dBm.

Signal levels are represented by negative dBm values. For example, a reading of −80 dBm represents a stronger received signal level than −100 dBm.

The target was analyzed as a regression variable, with model accuracy evaluated using RMSE and MAE.

### 2.4 Feature Selection

Seven features were selected for model development.

**Numerical features:**

- Latitude
- Longitude
- Speed
- Altitude

**Categorical features:**

- Operatorname
- NetworkTech
- Mobility

Numerical features were standardized using `StandardScaler`, while categorical features were encoded using `OneHotEncoder(handle_unknown="ignore")`.

Both transformations were incorporated into a scikit-learn `Pipeline` to ensure consistent preprocessing during training and inference.

### 2.5 Session-Aware Data Splitting

The dataset contains repeated observations collected within measurement sessions.

To reduce the risk of leakage between training and evaluation, the project uses `GroupShuffleSplit` with `SessionID` as the grouping variable.

| Dataset partition | Records | Sessions |
|---|---:|---:|
| Training | 16,768 | 13 |
| Validation | 8,862 | 5 |
| Test | 5,295 | 5 |
| **Total** | **30,925** | **23** |

No measurement session appears in more than one partition.

Both 4G and 5G observations are represented in all three partitions.

This strategy provides a more meaningful evaluation of generalization to unseen measurement sessions than randomly splitting individual records.

### 2.6 EDA Summary

The exploratory analysis informed the modeling approach in several ways:

1. The dataset contains both numerical and categorical predictors, requiring a combined preprocessing pipeline.
2. Constant-valued features were removed because they offered no useful variation in the observed dataset.
3. Related signal measurement variables were excluded to reduce the risk of target leakage.
4. Session-aware splitting was selected to prevent measurements from the same collection session appearing in multiple evaluation partitions.
5. RMSE and MAE were selected as evaluation metrics for the continuous target variable.

The complete exploratory analysis and model development workflow are available in [`notebook.ipynb`](notebook.ipynb).



## 3. Model Training

### 3.1 Modeling Objective

The objective is to train and evaluate regression models that predict mobile network signal level (`Level`) in dBm using geographic, mobility, and network characteristics.

Five regression algorithms were evaluated, including a baseline model, to determine which approach provided the most accurate predictions.

### 3.2 Data Preparation

The dataset was split into training, validation, and test partitions using `GroupShuffleSplit`, with `SessionID` as the grouping variable and `random_state=42`.

The initial split contained:

| Partition | Records | Percentage |
|---|---:|---:|
| Training | 16,768 | 54.2% |
| Validation | 8,862 | 28.7% |
| Test | 5,295 | 17.1% |
| **Total** | **30,925** | **100%** |

The split was performed at the session level to prevent measurements from the same collection session appearing in multiple partitions.

### 3.3 Feature Preprocessing

The preprocessing pipeline uses:

- `StandardScaler` for numerical features: Latitude, Longitude, Speed, and Altitude.
- `OneHotEncoder(handle_unknown="ignore")` for categorical features: Operatorname, NetworkTech, and Mobility.
- `ColumnTransformer` to combine numerical and categorical preprocessing.
- scikit-learn `Pipeline` to integrate preprocessing and model training.

The pipeline ensures that the same feature transformations are applied during training, evaluation, and prediction.

### 3.4 Model Comparison

The following regression models were trained and evaluated using validation RMSE.

| Model | Validation RMSE (dB) |
|---|---:|
| DummyRegressor (Baseline) | 11.0004 |
| Linear Regression | 10.2551 |
| Decision Tree Regressor | 8.8686 |
| Random Forest Regressor | 8.6832 |
| XGBoost Regressor | 8.8540 |
| Tuned Random Forest | 8.4486 |
| **Tuned XGBoost** | **8.3335** |

All trained models improved upon the baseline. The tuned XGBoost model achieved the lowest validation RMSE.

### 3.5 Hyperparameter Tuning

Random Forest and XGBoost were evaluated using multiple hyperparameter configurations.

The best-performing XGBoost configuration was:

| Hyperparameter | Selected value |
|---|---|
| `n_estimators` | 300 |
| `max_depth` | 7 |
| `learning_rate` | 0.03 |
| `min_child_weight` | 5 |
| `subsample` | 0.8 |
| `colsample_bytree` | 0.8 |
| `objective` | `reg:squarederror` |
| `random_state` | 42 |
| `n_jobs` | -1 |

The selected model achieved:

- **Validation RMSE:** 8.3335 dB
- **Validation MAE:** 6.0246 dB

### 3.6 Final Model Selection

The tuned XGBoost model was selected because it achieved the lowest validation RMSE among the evaluated configurations.

After model selection, the final pipeline was retrained using the combined training and validation datasets, totaling 25,630 records.

The held-out test dataset of 5,295 records was reserved for final evaluation.

### 3.7 Final Test Performance

| Metric | Result |
|---|---:|
| Test RMSE | **9.1644 dB** |
| Test MAE | **6.9597 dB** |
| Final training records | 25,630 |
| Test records | 5,295 |
| Selected model | Tuned XGBoost Regressor |

The final test RMSE is higher than the validation RMSE, indicating that prediction errors vary across measurement sessions. This highlights the importance of evaluating the model on previously unseen sessions.

### 3.8 Model Export

The final trained pipeline was serialized using Python's `pickle` module and saved as:

`models/xgboost_signal_model.bin`

The exported pipeline includes both preprocessing transformations and the trained XGBoost regression model.

This model artifact is loaded by the Flask prediction service to generate predictions without retraining.

### 3.9 Training Notebook

The complete experimental workflow, including model comparison, hyperparameter tuning, and final evaluation, is documented in [`notebook.ipynb`](notebook.ipynb).



## 4. Training Script (Exporting Notebook to Script)

### 4.1 Overview

The model development experiments were initially performed in `notebook.ipynb`.

The final training workflow was then implemented in a standalone Python script, [`train.py`](train.py), allowing the selected model to be trained, evaluated, and exported without running Jupyter Notebook.

The training script performs the following operations:

1. Loads the cleaned mobile network measurement dataset.
2. Selects the seven input features and target variable (`Level`).
3. Reproduces the session-aware data splitting using `GroupShuffleSplit` with `random_state=42`.
4. Combines the training and validation partitions for final model training.
5. Creates a preprocessing pipeline for numerical and categorical features.
6. Trains the tuned XGBoost regression model.
7. Evaluates the model against the held-out test dataset.
8. Saves the trained pipeline to `models/xgboost_signal_model.bin`.

### 4.2 Running the Training Script

From the project root directory, activate the Python virtual environment and ensure the dependencies are installed.

Run:

```powershell
python train.py
```

The script reads the dataset from the project's `data/` directory.

The dataset must be downloaded and extracted before running the script. Dataset setup instructions are provided in Section 5 – Reproducibility.

### 4.3 Expected Training Output

A successful training run produces output similar to:

```text
Loaded dataset: (30925, 51)
Training records: 25630
Test records: 5295
Test RMSE: 9.1644 dB
Test MAE:  6.9597 dB
Model saved: .../models/xgboost_signal_model.bin
```

### 4.4 Saved Model Artifact

The final trained model is saved at:

```text
models/xgboost_signal_model.bin
```

The serialized artifact contains the complete preprocessing and prediction pipeline, including:

- Numerical feature scaling
- Categorical feature encoding
- Tuned XGBoost regression model

The exported model is used by the Flask prediction service in `predict.py`.

### 4.5 Training Script Verification

The standalone training script was executed successfully in the project environment.

The resulting model achieved:

| Metric | Result |
|---|---:|
| Test RMSE | 9.1644 dB |
| Test MAE | 6.9597 dB |
| Training records | 25,630 |
| Test records | 5,295 |

The saved model was also loaded successfully and verified to produce consistent predictions.

This confirms that the final model can be trained and exported independently of the experimental notebook.



## 5. Reproducibility

### 5.1 Overview

This project provides the source code, dependency specifications, training script, trained model, and Docker configuration needed to reproduce the machine learning workflow.

The original dataset is not committed to GitHub. Reviewers must download it separately from Mendeley Data.

### 5.2 Clone the Repository

Clone the project and navigate to its root directory:

```powershell
git clone https://github.com/robanc/ml-zoomcamp-2026-midterm.git
cd ml-zoomcamp-2026-midterm
```

### 5.3 Download the Dataset

Download the publicly available dataset from:

https://data.mendeley.com/datasets/dx5xyyfz2y/2

Locate the cleaned dataset file:

`processed_data_cleaned.v2.csv`

Extract the downloaded files into the project's `data/` directory, preserving the original dataset folder structure.

The training script expects to locate the cleaned CSV file within the `data/` directory.

**Note:** The `data/` directory is excluded from Git version control because the original dataset is distributed separately.

### 5.4 Create the Python Virtual Environment

The project was developed using Python 3.12.

From the project root, create a virtual environment:

```powershell
python -m venv .venv
```

Activate it in Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

### 5.5 Install Dependencies

Install the required packages:

```powershell
python -m pip install -r requirements.txt
```

The dependency versions are pinned to support reproducible execution.

### 5.6 Reproduce Model Training

Run the standalone training script:

```powershell
python train.py
```

The script performs session-aware splitting, preprocessing, model training, evaluation, and model export.

The final model uses the tuned XGBoost configuration selected during the notebook experiments.

### 5.7 Expected Results

A successful execution produces results similar to:

```text
Loaded dataset: (30925, 51)
Training records: 25630
Test records: 5295
Test RMSE: 9.1644 dB
Test MAE:  6.9597 dB
Model saved: .../models/xgboost_signal_model.bin
```

The model artifact is written to:

`models/xgboost_signal_model.bin`

Minor numerical differences may occur across operating systems or dependency environments.

### 5.8 Reproducibility Controls

The following practices support consistent results:

| Control | Implementation |
|---|---|
| Random seed | `random_state=42` |
| Data splitting | `GroupShuffleSplit` by `SessionID` |
| Target leakage prevention | Exclusion of related signal measurement predictors |
| Feature preprocessing | scikit-learn `Pipeline` and `ColumnTransformer` |
| Model configuration | Explicit XGBoost hyperparameters |
| Dependencies | Pinned versions in `requirements.txt` |
| Model export | Serialized trained pipeline |
| Evaluation | Held-out test dataset |

### 5.9 Reproducing the Prediction Service

After training, start the Flask prediction service:

```powershell
python predict.py
```

The service listens on port `9696`.

In another PowerShell terminal, verify the health endpoint:

```powershell
Invoke-RestMethod -Uri http://127.0.0.1:9696/ping
```

Expected response:

```json
{
  "status": "ok"
}
```

The prediction endpoint is documented in Section 6 – Model Deployment.

### 5.10 Repository Files

| File | Purpose |
|---|---|
| `notebook.ipynb` | EDA, experiments, model comparison, and tuning |
| `train.py` | Reproducible model training |
| `predict.py` | Flask prediction API |
| `models/xgboost_signal_model.bin` | Exported trained model |
| `requirements.txt` | Python dependencies |
| `Dockerfile` | Containerized prediction service |
| `.dockerignore` | Docker build exclusions |
| `.gitignore` | Git version-control exclusions |
| `README.md` | Project documentation |



## 6. Model Deployment

### 6.1 Overview

The trained XGBoost regression model is deployed locally as a REST API using Flask and Waitress.

The prediction service is implemented in [`predict.py`](predict.py) and loads the serialized model from:

`models/xgboost_signal_model.bin`

The service accepts geographic, mobility, and network information in JSON format and returns the predicted mobile network signal level in dBm.

The model is loaded when the application starts, so individual prediction requests do not require retraining.

### 6.2 API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/ping` | Health check |
| POST | `/predict` | Predict mobile network signal level |

**Local API address:** `http://127.0.0.1:9696`

### 6.3 Start the Prediction Service

Ensure the Python virtual environment is activated and dependencies are installed.

From the project root directory, run:

```powershell
python predict.py
```

The application starts the prediction service on port `9696`.

Keep the terminal running while testing the API.

### 6.4 Test the Health Endpoint

Open a second PowerShell terminal and run:

```powershell
Invoke-RestMethod -Uri http://127.0.0.1:9696/ping
```

Expected JSON response:

```json
{
  "status": "ok"
}
```

A successful response confirms that the API is running.

### 6.5 Send a Prediction Request

The `/predict` endpoint accepts a JSON object containing seven input features.

Example request:

```json
{
  "Latitude": 3.0678,
  "Longitude": 101.6053,
  "Speed": 15.0,
  "Operatorname": "Operator A",
  "NetworkTech": "5G",
  "Mobility": "BRT",
  "Altitude": 28.0
}
```

Run the following command in PowerShell:

```powershell
$body = @{
    Latitude = 3.0678
    Longitude = 101.6053
    Speed = 15.0
    Operatorname = "Operator A"
    NetworkTech = "5G"
    Mobility = "BRT"
    Altitude = 28.0
} | ConvertTo-Json

Invoke-RestMethod `
    -Uri http://127.0.0.1:9696/predict `
    -Method Post `
    -ContentType "application/json" `
    -Body $body
```

### 6.6 Expected Prediction Response

The API returns a JSON object containing the predicted signal level.

Example response from the tested model:

```json
{
  "predicted_signal_level_dbm": -97.588
}
```

The predicted signal level is approximately **−97.59 dBm** for the supplied input characteristics.

This is a model-generated estimate, not a live network measurement.

### 6.7 Input Validation and Error Handling

The prediction API validates the supplied input before running inference.

The service checks for required features and supported categorical values.

Supported categorical inputs:

| Feature | Accepted values |
|---|---|
| Operatorname | Operator A, Operator B, Operator C |
| NetworkTech | 4G, 5G |
| Mobility | Walk, BRT, Shuttle |

Requests with missing required fields or invalid inputs return an HTTP `400 Bad Request` response with error information.

This helps prevent malformed requests from being passed to the trained model.

### 6.8 Deployment Verification

The prediction service was tested locally using the exported XGBoost model.

The following checks were completed:

- Flask application started successfully.
- `/ping` returned a successful health response.
- `/predict` accepted a valid JSON request.
- The model returned a numerical signal-level prediction.
- Missing required fields produced an HTTP 400 validation error.
- The same prediction was reproduced using the Dockerized service.

The API can therefore be run locally without executing the training notebook or retraining the model.

Docker deployment instructions are provided in Section 8 – Containerization.



## 7. Dependencies and Environment

### 7.1 Development Environment

The project was developed and tested using the following environment:

| Component | Configuration |
|---|---|
| Operating system | Windows |
| Python | 3.12.10 |
| Environment management | Python `venv` |
| Machine learning framework | scikit-learn |
| Gradient boosting framework | XGBoost |
| Prediction API | Flask |
| Production WSGI server | Waitress |
| Containerization | Docker |

The project uses a dedicated Python virtual environment to isolate its dependencies from other Python projects.

### 7.2 Python Dependencies

The required application and training dependencies are specified in [`requirements.txt`](requirements.txt).

```text
flask==3.1.3
waitress==3.0.2
numpy==2.5.4
pandas==3.1.0
scikit-learn==1.9.1
xgboost==3.4.1
```

These packages support data processing, preprocessing, machine learning, model serialization, and API deployment.

The versions are pinned to reduce compatibility differences between environments.

### 7.3 Create a Virtual Environment

Python 3.12 is recommended.

From the project root directory, create a virtual environment:

```powershell
python -m venv .venv
```

Activate it in Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

Verify the Python version:

```powershell
python --version
```

Expected development version:

```text
Python 3.12.10
```

### 7.4 Install Project Dependencies

Install the required packages:

```powershell
python -m pip install -r requirements.txt
```

Verify that the packages are installed:

```powershell
python -m pip check
```

This command checks for incompatible or missing package dependencies.

### 7.5 Run the Project

Once the environment is configured and the dataset has been downloaded, reproduce the final model training:

```powershell
python train.py
```

Start the prediction API:

```powershell
python predict.py
```

The API runs locally at:

`http://127.0.0.1:9696`

### 7.6 Notebook Environment

The experimental workflow is documented in [`notebook.ipynb`](notebook.ipynb).

JupyterLab and visualization libraries such as Matplotlib and Seaborn were used during exploratory analysis and experimentation.

These notebook-specific tools are not included in the minimal application `requirements.txt` because they are not required to execute `train.py` or run the prediction API.

To inspect and execute the notebook, reviewers can install the additional tools:

```powershell
python -m pip install jupyterlab matplotlib seaborn
```

### 7.7 Dependency Management

The project separates its core runtime dependencies from optional notebook tools.

This approach provides:

- A smaller production dependency set.
- Consistent model training and inference environments.
- Easier installation for reviewers.
- Reduced Docker image complexity.
- Clearer separation between experimentation and deployment.

The Docker image uses `python:3.12-slim` and installs dependencies directly from `requirements.txt`.

Containerization instructions are provided in Section 8.



## 8. Containerization

### 8.1 Overview

The prediction service is containerized using Docker to provide a consistent and portable runtime environment.

The Docker image includes:

- Python 3.12
- Required dependencies from `requirements.txt`
- Flask prediction application (`predict.py`)
- Pretrained XGBoost model (`models/xgboost_signal_model.bin`)
- Waitress production WSGI server

The Docker container serves the trained model without requiring Jupyter Notebook, the original training dataset, or model retraining.

### 8.2 Dockerfile

The project includes a [`Dockerfile`](Dockerfile) with the following configuration:

```dockerfile
FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY predict.py .

COPY models/xgboost_signal_model.bin models/xgboost_signal_model.bin

EXPOSE 9696

CMD ["waitress-serve", "--host=0.0.0.0", "--port=9696", "predict:app"]
```

The image uses the lightweight Python 3.12 slim base image.

Waitress serves the Flask application on port `9696` and listens on all container network interfaces.

### 8.3 Build the Docker Image

Docker must be installed and running.

From the project root directory, execute:

```powershell
docker build -t ml-zoomcamp-midterm:v1 .
```

This command builds the image and installs the required dependencies.

The trained model is copied into the image during the build process.

### 8.4 Run the Docker Container

Start the container:

```powershell
docker run --rm -d --name ml-midterm-api -p 9697:9696 ml-zoomcamp-midterm:v1
```

The command maps:

- Host port: `9697`
- Container port: `9696`

The prediction service is accessible at:

`http://127.0.0.1:9697`

The `--rm` option removes the container automatically when it stops.

### 8.5 Verify the Container

Check that the container is running:

```powershell
docker ps
```

Test the health endpoint:

```powershell
Invoke-RestMethod -Uri http://127.0.0.1:9697/ping
```

Expected response:

```json
{
  "status": "ok"
}
```

### 8.6 Test the Containerized Prediction API

Use the following sample request:

```powershell
$body = @{
    Latitude = 3.0678
    Longitude = 101.6053
    Speed = 15.0
    Operatorname = "Operator A"
    NetworkTech = "5G"
    Mobility = "BRT"
    Altitude = 28.0
} | ConvertTo-Json

Invoke-RestMethod `
    -Uri http://127.0.0.1:9697/predict `
    -Method Post `
    -ContentType "application/json" `
    -Body $body
```

Expected response from the tested model:

```json
{
  "predicted_signal_level_dbm": -97.588
}
```

The Dockerized service produced the same prediction as the locally executed Flask application.

### 8.7 Docker Build Optimization

The project includes a [`.dockerignore`](.dockerignore) file to exclude unnecessary files from the Docker build context.

Excluded items include:

- Python virtual environments
- Git metadata
- Jupyter notebooks and checkpoints
- Training datasets
- Python cache files
- Local environment and log files

This reduces the build context size and keeps the container focused on model inference.

### 8.8 Stop the Container

To stop the running container:

```powershell
docker stop ml-midterm-api
```

### 8.9 Containerization Verification

The Docker implementation was tested successfully.

| Test | Result |
|---|---|
| Docker image build | Passed |
| Container startup | Passed |
| Health endpoint `/ping` | Passed |
| Prediction endpoint `/predict` | Passed |
| Local and Docker prediction consistency | Passed |

The containerized prediction service is ready for deployment to a compatible cloud hosting platform.



## 9. Cloud Deployment

### 9.1 Overview

The trained XGBoost regression model is deployed to Render as a Docker-based web service.

The deployment provides a publicly accessible REST API for predicting 4G/5G mobile network signal levels.

**Cloud hosting platform:** Render

**Public service URL:**

https://ml-zoomcamp-2026-midterm.onrender.com

**GitHub repository:**

https://github.com/robanc/ml-zoomcamp-2026-midterm

### 9.2 Deployment Configuration

| Setting | Configuration |
|---|---|
| Cloud platform | Render |
| Service name | `ml-zoomcamp-2026-midterm` |
| Runtime | Docker |
| Instance type | Free |
| Python base image | `python:3.12-slim` |
| Application server | Waitress |
| Container port | 9696 |
| Health check path | `/ping` |
| Auto-deploy | On Commit |
| Initial deployed commit | `007dd6a` |

The service uses the Dockerfile included in the repository and loads the previously trained model artifact.

### 9.3 Deployment Procedure

The deployment was configured using the following steps:

1. Sign in to the Render dashboard.
2. Create a new Web Service.
3. Select the public GitHub repository.
4. Use Docker as the runtime.
5. Select the Free instance type.
6. Keep the Dockerfile path at its default location.
7. Configure `/ping` as the health check path.
8. Enable automatic deployment on commit.
9. Deploy the web service.

The initial deployment completed successfully in approximately 2 minutes and 26 seconds.

### 9.4 Public API Endpoints

| Method | Endpoint | Purpose |
|---|---|---|
| GET | `/ping` | Verify service health |
| POST | `/predict` | Predict mobile network signal level |

**Health endpoint:**

https://ml-zoomcamp-2026-midterm.onrender.com/ping

**Prediction endpoint:**

https://ml-zoomcamp-2026-midterm.onrender.com/predict

### 9.5 Cloud Health Check

The public health endpoint was tested successfully.

Expected response:

```json
{
  "status": "ok"
}
```

### 9.6 Cloud Prediction Test

The deployed API was tested using Windows PowerShell.

**Sample request:**

```powershell
$body = @{
    Latitude = 3.0678
    Longitude = 101.6053
    Speed = 15.0
    Operatorname = "Operator A"
    NetworkTech = "5G"
    Mobility = "BRT"
    Altitude = 28.0
} | ConvertTo-Json

Invoke-RestMethod `
    -Uri "https://ml-zoomcamp-2026-midterm.onrender.com/predict" `
    -Method Post `
    -ContentType "application/json" `
    -Body $body
```

**Actual response:**

```json
{
  "predicted_signal_level_dbm": -97.588
}
```

The cloud prediction matched the results obtained from the locally executed Flask service and Docker container.

### 9.7 Deployment Verification

| Verification | Result |
|---|---|
| Docker image built successfully | Passed |
| Render deployment succeeded | Passed |
| Public health endpoint | Passed |
| Public prediction endpoint | Passed |
| Prediction consistency | Passed |

### 9.8 Free Instance Limitations

The application uses Render's Free instance type.

Free instances can spin down during inactivity. As a result, the first request after an idle period may take 50 seconds or longer while the service restarts.

Subsequent requests are generally faster while the instance remains active.

### 9.9 Deployment Outcome

The project demonstrates a complete machine learning workflow:

**Data exploration → Model training → Hyperparameter tuning → Model export → Flask API → Docker containerization → Cloud deployment**

The trained XGBoost model is accessible through a public REST API, enabling predictions without requiring users to install Python, download the dataset, or retrain the model.
