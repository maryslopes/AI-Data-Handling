# AI Data Handling Project M1 2026
Machine learning and AI projects 2026 course.

# My Project
Build an AI system that identifies early signs of equipment malfunction to reduce batch failures, prevent quality deviations, and support continuous improvement in pharmaceutical production environments.

#1.Raw data storage
he raw data was obtained from The Raw data will be Kaggle public database center, as per link below:
Dataset: Equipment Failure Prediction Dataset
https://www.kaggle.com/datasets/geetanjalisikarwar/equipment-failure-prediction-dataset

The raw dataset (Equipment Failure, originally from Kaggle) is stored as a CSV file inside a Google Cloud Storage (GCS) bucket:
gs://equipment-failure-20122724/raw/Equipment-Failure.csv
GCS is appropriate because it provides durable, scalable object storage, supports versioning, integrates directly with BigQuery and Colab, and allows programmatic access through gcsfs.

#2.Processed data storage & file formats
The processed data will be storage as CSV file on the following bucket
BUCKET_NAME = "equipment-failure-20122724"
LOCATION = "US"

#3.Database/object storage decision

The files will be storage on Google Cloud Storage (GCS) in a structured format as followed
//<equipment-failure-20122724>/
├── raw/Equipment-Failure.csv                - (raw original dataset obtained from Kaggle)
└── sharded/                       - (create splits and cross-valitation)
    ├── train.csv
    ├── dev.csv
    ├── test.csv
    └── cv_folds/                  - (cross-validation folders for AI training)
        ├── fold_0_train.csv
        ├── fold_0_val.csv
        ├── fold_1_train.csv
        ├── fold_1_val.csv
        ├── fold_2_train.csv
        ├── fold_2_val.csv
        ├── fold_3_train.csv
        ├── fold_3_val.csv
        ├── fold_4_train.csv
        ├── fold_4_val.csv
        ├── fold_5_train.csv
        ├── fold_5_val.csv
        ├── fold_6_train.csv
        ├── fold_6_val.csv
        ├── fold_7_train.csv
        ├── fold_7_val.csv
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
Primary split (train / dev / test)**
- **Test set**: 10% of unique Product ID groups, held out and never used during model selection.
- **Dev set**: 10% of unique Product ID groups, used for hyperparameter tuning.
- **Train set**: remaining 80%.

#7.Feature description
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
The data can be reproduced by following the commands on Colab as per https://github.com/maryslopes/AI-Data-Handling/blob/main/Equipment_Failure_notebook_gcs.ipynb
<img width="1602" height="587" alt="M1 screenshot raw data" src="https://github.com/user-attachments/assets/5cefea73-0745-44b0-a5a8-2a4bae8a295f" />

<img width="1552" height="942" alt="M1 screenshot split" src="https://github.com/user-attachments/assets/b312e97e-8630-4fac-a867-5575a9ec62eb" />

#10. Reproducibility of preprocessing
As per steps described on printscreen below the columns names were convert to format as per BigQuery format.
<img width="1007" height="665" alt="M1 screenshot processed data to adjust column names" src="https://github.com/user-attachments/assets/74182961-38ae-4804-8b78-1e068cfbf9f5" />

The image below show the generated file:
<img width="1537" height="602" alt="M1 screenshot processed file" src="https://github.com/user-attachments/assets/0b2579ae-df4c-4c62-b067-81a4463fda1b" />
