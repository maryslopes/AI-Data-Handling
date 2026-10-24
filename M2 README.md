# AI Data Handling Project M2 2026
Machine learning and AI projects 2026 course.

# My Project
Build an AI system that identifies early signs of equipment malfunction to reduce batch failures, prevent quality deviations, and support continuous improvement in pharmaceutical production environments.

#1.Project definition
Project Title: Equipment-Failure-20122724
Problem Statement: I would like to build an AI system that identifies early signs of equipment malfunction to reduce batch failures, prevent quality deviations, and support continuous improvement in pharmaceutical production environments.
Intended Users: The system will predict whether equipment is likely to fail or operate abnormally during production. Early detection of equipment issues helps prevent batch deviations, OOS/OOC results, and production downtime, supporting continuous improvement and process optimisation in pharmaceutical plants.

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

#3.Data description
Unit of analysis: Each record represents one equipment operation instance, with associated process variables and a failure outcome label.
- Column types:
  - UDI: integer
  - Product ID: string
  - Type: categorical (string)
  - Air temperature [K]: float32
  - Process temperature [K]: float32
  - Rotational speed [rpm]: float32
  - Torque [Nm]: float32
  - Tool wear [min]: float32
  - Machine failure: int8 (0/1)
  - TWF, HDF, PWF, OSF, RNF: int8 (0/1)

Number of records: Approximately 10,000+ rows.

Fields and data types: The dataset contains multiple numeric and categorical fields describing operational parameters (e.g., load, pressure, temperature, speed) and a binary target variable indicating equipment failure (Yes/No).

Target variable: Failure (binary classification: 1 = failure, 0 = no failure).

Time range: The dataset does not specify a real-world time range; it is presented as a static collection of operational records.

Class balance: The dataset includes both failure and non-failure cases. Failures are less frequent than normal operations, which introduces a class imbalance typical of real industrial data.

Missingness: Some fields contain missing values and inconsistencies, reflecting realistic data quality issues encountered in manufacturing environments.

Known limitations:

The dataset is not pharmaceutical-specific; it represents general industrial equipment.

It may not capture the full diversity of equipment types or failure modes found in pharma plants.

The imbalance between failure and non-failure records may bias models toward predicting “no failure”.

The dataset is anonymised/simulated, so while useful for modelling, it may not fully reflect the complexity of regulated GMP environments.



#4.Exploratory data analysis

Relevant descriptive statistics and visualisations. Explain what each result means for 
cleaning, transformation, feature engineering, splitting or evaluation. 

#5.Design/ Architecture

#6.Implementation

#7.Next steps

