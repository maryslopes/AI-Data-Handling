# AI Data Handling Project M1 2026
Machine learning and AI projects 2026 course.

# My Project
Build an AI system that identifies early signs of equipment malfunction to reduce batch failures, prevent quality deviations, and support continuous improvement in pharmaceutical production environments.

#1.Raw data storage
The raw data was obtained from The Raw data will be Kaggle public database center, as per link below:
Dataset: Equipment Failure Prediction Dataset
https://www.kaggle.com/datasets/geetanjalisikarwar/equipment-failure-prediction-dataset
The original CSV is stored unchanged at gs://equipment-failure-20122724/raw/Equipment-Failure.csv. THe CSV file is storage in a Google Cloud Storage (GCS).
GCS is appropriate because it provides durable, scalable object storage, supports versioning, integrates directly with BigQuery and Colab, and allows programmatic access through gcsfs.


#2.Processed data storage & file formats
The processed data will be storage as CSV file on the following bucket
BUCKET_NAME = "equipment-failure-20122724"
LOCATION = "EU"
Processed datasets will be stored under gs://equipment-failure-20122724/processed.
Train, development, test, and cross-validation files will be stored under gs://equipment-failure-20122724/sharded/cv_folds

#3.Database/object storage decision

The files will be storage on Google Cloud Storage (GCS) in a structured format as followed

<img width="600" height="465" alt="image" src="https://github.com/user-attachments/assets/5c9e40a4-24b7-44e3-bda8-e1f5c4810043" />


 The object storage (Google Cloud Storage) rather than a relational database because:
- Data is tabular, file-oriented, and consumed in batch for ML training.
- GCS supports versioning, lifecycle rules, and easy integration with training environments.
- If low-latency row-level queries are later required, we will export to BigQuery or a managed OLAP store.

#4.Data versioning 
The data versioning will be automatically track in Google Cloud Storage by enabling Object Versioning on my bucket. This creates a full history of every overwrite or deletion, each identified by a unique generation number.
Keep up to 3 recent generations per object; older generations are deleted after 7 days (lifecycle rule).

#5.Data access
The data stored in Google Cloud Storage (GCS) will be accessed programmatically through authenticated sessions in Google Colab. Access is performed using:

GCS paths such as:
gs://equipment-failure-20122724/raw/Equipment-Failure.csv

gcsfs for reading CSV files directly into pandas

Google Cloud SDK authentication (from google.colab import auth; auth.authenticate_user())

BigQuery Python client when loading data into BigQuery

Only authenticated users with the correct IAM permissions (Storage Object Viewer / Storage Object Admin) can access the bucket. No public access is enabled.

 
#6.Data split/validation strategy 

First, 10% of observations are assigned to a final test set and locked until final evaluation. Of the remaining 90%, one ninth is assigned to the development set, producing overall proportions of 80% train, 10% development, and 10% test. Splitting uses random_state=42 and stratification by Machine failure to preserve the minority failure rate. If multiple rows share a machine, batch, or product identifier, all rows from the same group are assigned to one partition to prevent leakage. Eight-fold cross-validation is performed only on the training partition. Imputation, encoding, scaling, feature selection, and resampling are fitted separately within each training fold and then applied to its validation fold. The development set is used for final model selection, and the test set is evaluated once.


#7.Feature description
The description are defined below:

UDI: A unique identifier for each data point, ranging from 1 to 10,000.

Product ID: A unique identifier for each product.

Type: The quality variant of the product, categorized as 'L' (low), 'M' (medium), or 'H' (high).

Air temperature [K]: The ambient air temperature in Kelvin.

Process temperature [K]: The temperature of the manufacturing process in Kelvin.

Rotational speed [rpm]: The rotational speed of the machine's tool in revolutions per minute.

Torque [Nm]: The torque applied by the tool in Newton-meters.

Tool wear [min]: The wear on the tool in minutes of usage.

Machine failure: A binary label indicating whether the machine failed (1) or not (0). This is the primary target variable.

TWF (Tool Wear Failure): A binary flag indicating if the failure was caused by tool wear.

HDF (Heat Dissipation Failure): A binary flag indicating if the failure was caused by heat dissipation issues.

PWF (Power Failure): A binary flag indicating if the failure was caused by a power failure.

OSF (Overstrain Failure): A binary flag indicating if the failure was caused by overstraining the tool.

RNF (Random Failure): A binary flag indicating if the failure was a random, non-specific failure.

The prediction target is Machine failure. UDI is excluded because it is a row identifier. Product ID is retained only for grouping or traceability and is not used as a predictor. TWF, HDF, PWF, OSF, and RNF are excluded from the input feature set because they describe failure modes associated with the target and would introduce target leakage. The model predictors are Type, air temperature, process temperature, rotational speed, torque, and tool wear.

#8. Data types and formats

- Raw file: CSV (utf-8) as downloaded from Kaggle.
- Processed files: Parquet (preferred) and CSV exports for inspection.
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

#9. Reproducibility of data collection
-Kaggle dataset identifier: https://www.kaggle.com/datasets/geetanjalisikarwar/equipment-failure-prediction-dataset/data

-Downloaded using Kaggle API or manual download (see screenshot): 
<img width="1887" height="737" alt="image" src="https://github.com/user-attachments/assets/2f95aa85-7ca2-48fd-b8cc-1f95081532a0" />
-GCP project, bucket region, and destination URI: 
BUCKET_NAME = "equipment-failure-20122724"
LOCATION = "US"
gs://equipment-failure-20122724/raw

-Git notebook
The data can be reproduced by following the commands on Colab as per https://github.com/maryslopes/AI-Data-Handling/blob/main/Equipment_Failure_notebook_gcs.ipynb
The Tasks are as followed:
Cell 1 — Setup & Environment Installation
Cell 2 — Configuration & GCP Authentication
Cell 3 — Create the GCS Bucket
Cell 4 — Download Raw Equipment-Failure Data & Ingest to GCS
Cell 5 — Partition Data: Train / Dev / Test + 8-Fold CV
Cell 6 — Export Shards and Push to sharded/ in GCS
Cell 6.b. Process data to align to BigQuery requirements
Cell 7 — Load Data into GCP SQL Databases
Cell 8 — Verify Bucket Hierarchy & Read a GCS Shard


<img width="1602" height="587" alt="M1 screenshot raw data" src="https://github.com/user-attachments/assets/5cefea73-0745-44b0-a5a8-2a4bae8a295f" />

<img width="1552" height="942" alt="M1 screenshot split" src="https://github.com/user-attachments/assets/b312e97e-8630-4fac-a867-5575a9ec62eb" />

#10. Reproducibility of preprocessing
The following steps describes the ordered transformations:
Cell 1 — Setup & Environment Installation

<img width="1090" height="177" alt="image" src="https://github.com/user-attachments/assets/be328999-5ae7-457d-9d41-be5f444cc16f" />


Cell 2 — Configuration & GCP Authentication

<img width="682" height="675" alt="image" src="https://github.com/user-attachments/assets/cd15a785-6daa-4fe3-9f2d-15bbdfb1fbc6" />


Cell 3 — Create the GCS Bucket

<img width="585" height="320" alt="image" src="https://github.com/user-attachments/assets/325a7a91-8d3f-4462-9527-0b4c668f21d4" />


Cell 4 — Download Raw Equipment-Failure Data & Ingest to GCS

<img width="1107" height="532" alt="image" src="https://github.com/user-attachments/assets/62b7eda8-77e7-4e12-8103-fab9f7b8667d" />


Cell 5 — Partition Data: Train / Dev / Test + 8-Fold CV

<img width="601" height="797" alt="image" src="https://github.com/user-attachments/assets/84013cbc-089b-4d41-868d-05e3422d6605" />


<img width="1431" height="597" alt="image" src="https://github.com/user-attachments/assets/5c78eae7-1539-48be-a55a-00297d304f5d" />


Cell 6 — Export Shards and Push to sharded/ in GCS

<img width="645" height="607" alt="image" src="https://github.com/user-attachments/assets/fa26c91e-5746-4ddf-9adc-4b183e564d9a" />


Cell 6.b. Process data to align to BigQuery requirements

<img width="1427" height="682" alt="image" src="https://github.com/user-attachments/assets/bacf58c3-2930-482d-a90c-241e62d799a6" />

The image below show the generated file:
<img width="1537" height="602" alt="M1 screenshot processed file" src="https://github.com/user-attachments/assets/0b2579ae-df4c-4c62-b067-81a4463fda1b" />

Cell 7 — Load Data into GCP SQL Databases

<img width="1307" height="606" alt="image" src="https://github.com/user-attachments/assets/4dc7fd8b-ff91-43a1-8626-739975aeedab" />


Cell 8 — Verify Bucket Hierarchy & Read a GCS Shard
As per steps described on printscreen below the columns names were convert to format as per BigQuery format.

<img width="1007" height="665" alt="M1 screenshot processed data to adjust column names" src="https://github.com/user-attachments/assets/74182961-38ae-4804-8b78-1e068cfbf9f5" />

<img width="1235" height="332" alt="image" src="https://github.com/user-attachments/assets/0c5dd055-7025-41a0-b8a7-3c218e695f09" />


