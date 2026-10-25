# AI Data Handling Project M2 2026
Machine learning and AI projects 2026 course.

# My Project
Build an AI system that identifies early signs of equipment malfunction to reduce batch failures, prevent quality deviations, and support continuous improvement in pharmaceutical production environments.

#1.Project definition
Project Title: Equipment-Failure-20122724
Problem Statement: I would like to build an AI system that identifies early signs of equipment malfunction to reduce batch failures, prevent quality deviations, and support continuous improvement in pharmaceutical production environments.
Intended Users: The system will predict whether equipment is likely to fail or operate abnormally during production. Early detection of equipment issues helps prevent batch deviations, OOS/OOC results, and production downtime, supporting continuous improvement and process optimisation in pharmaceutical plants.
Success Criteria: The model is successful if it achieves >80% recall on failure detection while maintaining precision above 70%.

#2.Data source
The dataset used in this project is the Equipment Failure Prediction Dataset, openly available on Kaggle.
Source URL: https://www.kaggle.com/datasets/geetanjalisikarwar/equipment-failure-prediction-dataset  
Data owner: Geetanjali Sikarwar (dataset contributor on Kaggle)
Licence / Terms of Use: Shared under Kaggle’s open data terms, typically CC0 (Public Domain). It is free to use for academic and research purposes, provided proper attribution is given to the dataset owner.
Collection context: The dataset simulates industrial equipment operation records with labelled outcomes (failure vs. no failure). It was created for machine learning practice in predictive maintenance and reliability modelling. While not pharmaceutical‑specific, the context of equipment reliability is directly transferable to pharmaceutical manufacturing, where equipment failures often lead to batch deviations and quality issues.
Permissions: As an openly published Kaggle dataset, no special permissions are required for academic use.
Privacy considerations: The dataset contains no personal or sensitive information (no employee data, no patient data, no proprietary company details). It is safe to use in terms of privacy.
Bias considerations: The dataset may be biased toward certain types of equipment failures and may not represent the full diversity of industrial or pharmaceutical equipment. Models trained on this dataset may not generalise perfectly to all contexts.
Ethical considerations: Using this dataset for academic purposes is ethical, as it is openly shared. Care should be taken not to misrepresent it as pharmaceutical‑specific data; instead, it should be framed as a proxy dataset that demonstrates how AI can be applied to equipment failure prediction in manufacturing environments, including pharma.
Data limitation: Dataset is simulated, not pharma-specific, may not generalize to GMP environments.


#3.Data description
For statistical analysis and exploratory data analysis (EDA), we use the **processed dataset** stored in GCS:

`gs://equipment-failure-20122724/processed/Equipment-Failure-clean.csv`

This file is derived from the raw Kaggle dataset (`gs://equipment-failure-20122724/raw/Equipment-Failure.csv`) after validation and cleaning. Column names were adjusted to meet BigQuery requirements, and data types were standardized.

Screenshot of files on GCS:


*RAW
<img width="1766" height="617" alt="image" src="https://github.com/user-attachments/assets/3fac4fa8-344b-4515-bb77-1594401bb5d9" />

*PROCESSED
<img width="1767" height="601" alt="image" src="https://github.com/user-attachments/assets/70a15183-3c8c-43cc-b4d0-f6ac7fdd3857" />



### Unit of analysis
Each row represents one equipment operation instance with associated process variables and a binary failure outcome.

- Column types:
  - UDI: integer
  - product_id: string
  - type: categorical (string)
  - air_temperature_: float32
  - process_temperature_k: float32
  - rotational_speed_rpm: float32
  - torque_nm: float32
  - tool_wear_min: float32
  - machine_failure: int8 (0/1)
  - TWF, HDF, PWF, OSF, RNF: int8 (0/1)


Number of records: Approximately 10,000+ rows.

Fields and data types: The dataset contains multiple numeric and categorical fields describing operational parameters (e.g., load, pressure, temperature, speed) and a binary target variable indicating equipment failure (Yes/No).

Target variable: Failure (binary classification: 1 = failure, 0 = no failure).

Time range: The dataset does not specify a real-world time range; it is presented as a static collection of operational records.

Class balance: The dataset includes both failure and non-failure cases. Failures are less frequent than normal operations, which introduces a class imbalance typical of real industrial data.

Missingness: No missing values in key features.


Known limitations:

The dataset is not pharmaceutical-specific; it represents general industrial equipment.

It may not capture the full diversity of equipment types or failure modes found in pharma plants.

The imbalance between failure and non-failure records may bias models toward predicting “no failure”.

The dataset is anonymised/simulated, so while useful for modelling, it may not fully reflect the complexity of regulated GMP environments.



Code snippets from M2 equipment_failure_notebook_Statistical_Reasoning.ipynb Section 2: Dataset has 10,000 rows and 15 columns. Failure rate is ~3%. No missing values in key features.

<img width="842" height="600" alt="image" src="https://github.com/user-attachments/assets/5181e0a6-9e0c-41cb-8883-f411a06a7d28" />


<img width="876" height="335" alt="image" src="https://github.com/user-attachments/assets/baea8f30-ff44-4dc3-b09d-c6fdabe22a5e" />


<img width="891" height="635" alt="image" src="https://github.com/user-attachments/assets/ae4432b1-b531-48c9-a72e-ceaf3918f579" />




# 4.Exploratory data analysis

EDA was performed on the **processed dataset** (`processed/Equipment-Failure-clean.csv`).  
Key findings:

- **Failure distribution**: Failures are rare (~3%), confirming the need for resampling or class-weight adjustments.
- **Air temperature [K]**: Normally distributed between 295–325K.
- **Process temperature [K]**: Slightly skewed, ranges 305–325K.
- **Tool wear [min]**: Strongly skewed, with long tail → scaling required.
- **Rotational speed [rpm]**: Wide range, correlated with torque.
- **Correlation heatmap**: Moderate correlation between torque and rotational speed.


Code snippets from M2 equipment_failure_notebook_Statistical_Reasoning.ipynb section 4:

<img width="896" height="677" alt="image" src="https://github.com/user-attachments/assets/81236b22-be67-48d6-b4e9-ccb7d2a6d9f7" />


<img width="1383" height="1183" alt="image" src="https://github.com/user-attachments/assets/b03d61b0-4f10-4174-9a08-7b7a6ecc7dcd" />


<img width="962" height="851" alt="image" src="https://github.com/user-attachments/assets/c7ffcb08-58d4-4d23-8ea2-df06e6ce5b50" />


The statistics analysis show that: 
Data Quality & CompletenessSample Size (n = 10,000): All 5 features have 10,000 non-null values with 0 missing values. This ensures high statistical power for downstream modeling or inference. 

Air Temperature [K]:
  Central Tendency: Mean 300.005 [K] and Median 300.100 [K] are nearly identical.   
  Spread: Standard deviation is low 2.000 [K] with an IQR of 3.200 [K], spanning from 295.300 [K] to 304.500 [K].  
  Distribution Shape: Skewness is slightly positive 0.114. The data closely resembles a symmetrical, approximately Normal distribution.   

Process Temperature [K]
  Central Tendency: Mean 310.006 [K] and Median 310.100 [K] are virtually equal. On average, process temperature is about 10 [K] higher than air temperature.   
  Spread: Very low dispersion std = 1.484 [K], IQR = 2.300 [K], ranging between 305.700 [K] and 313.800[K].   
  Distribution Shape: Skewness is near zero (0.015), indicating a highly symmetric / bell-shaped distribution tightly controlled around 310[K].  
  
Rotational Speed [rpm]
  Central Tendency: Mean 1,538.776[rpm] is visibly higher than the Median 1,503.000[rpm].   
  Spread: Moderate variability std = 179.284[rpm], IQR = 189.000[rpm], spanning from 1,168.000[rpm] up to a maximum of $2,886.000[rpm].   
  Distribution Shape: Skewness is strongly positive 1.993.   
  Takeaway: This feature is strongly right-skewed (positively skewed). Most operating speeds cluster around 1,500[rpm], but there is a long tail of extreme high-speed operational events (reaching up to 2,886 [rpm]. 
  
Torque [Nm]
  Central Tendency: Mean 39.987[Nm] and Median 40.100 [Nm] are centered around 40[Nm].   
  Spread: Standard deviation is 9.969 [Nm] and IQR is 13.600 [Nm], ranging between 3.800 [Nm] and 76.600[Nm]. 
  Distribution Shape: Skewness is near zero -0.010.   
  Takeaway: Torque is well-balanced and symmetric around 40[Nm]. 
  
Tool Wear [min]
  Central Tendency: Mean 107.951[ min] and Median 108.000[min] sit right near the midpoint of its observed range.   
  Spread: High dispersion std = 63.654[ min], IQR = 109.000[min], stretching from 0[min] to 253[min].   
  Distribution Shape: Skewness is near zero 0.027.   
  Takeaway: Represents a Uniform-like distribution across tool lifecycle times, where tools accumulate wear linearly from 0 up to around 250 minutes before replacement or failure

Summary:
- Failure cases are rare (~3%), confirming class imbalance.
- Air temperature ranges between 295K–325K, normally distributed.
- Tool wear shows a skewed distribution, suggesting scaling or transformation.
- Correlation heatmap indicates torque and rotational speed are moderately correlated.
  
#5.Design/ Architecture
The pipeline follows an end‑to‑end structure from data acquisition to output.  
Since Milestone 2 analysis is based on the **processed cleaned file**, the architecture emphasizes reproducibility while ensuring that validated, BigQuery‑compatible data is used for statistical analysis.

```mermaid
flowchart LR
    A[Kaggle Dataset] --> B[Raw Data in GCS]
    B --> C[Validation: schema, ranges, missing values]
    C --> D[Processed Clean File in GCS]
    D --> E[Transformation: encoding, scaling, splitting]
    E --> F[Train/Dev/Test + CV folds]
    F --> G[Outputs for EDA and ML]
```

Source: Kaggle dataset downloaded and stored as raw/Equipment-Failure.csv.

Validation: Schema checks, uniqueness of UDI, and binary integrity of failure flags.

Processed file: Cleaned dataset (processed/Equipment-Failure-clean.csv) with renamed columns for BigQuery compatibility.

Transformation: Encoding categorical variables, scaling numeric features, and splitting into train/dev/test sets.

Output: Sharded datasets and cross‑validation folds stored under sharded/.

Analysis: All EDA and statistical analysis in Milestone 2 are performed on the processed file.


#6.Implementation

The GCS bucket is organized into three main areas:

- **Raw data**: `gs://equipment-failure-20122724/raw/Equipment-Failure.csv`  
  Source dataset downloaded from Kaggle. All train/dev/test splits and CV folds are generated from this file.

- **Processed data**: `gs://equipment-failure-20122724/processed/Equipment-Failure-clean.csv`  
  Cleaned version of the raw dataset, with column names adjusted for BigQuery requirements. Used for SQL queries and reproducibility checks.

- **Sharded splits**: `gs://equipment-failure-20122724/sharded/`  
  Contains train/dev/test partitions and 8-fold cross-validation datasets, all derived from the raw file.

This separation ensures:
- **Raw file** remains untouched for reproducibility.  
- **Processed file** supports BigQuery ingestion.  
- **Sharded splits** support ML training and evaluation.

In Milestone 1, the ingestion and splitting scripts were implemented   
- Raw data (`raw/Equipment-Failure.csv`) was uploaded to GCS.  
- Train/dev/test splits and CV folds were generated from the raw file and stored under `sharded/`.  

In Milestone 2, we build on this foundation by validating and cleaning the dataset, using the BigQuery‑compatible file (`processed/Equipment-Failure-clean.csv`) and performing statistical analysis and EDA.


#7.Next steps
-Baseline: logistic regression/random forest.
-Optimize preprocessing (scaling, feature engineering, dimensionality reduction).
-Prepare for Milestone 3: baseline model comparison and evaluation plan.



