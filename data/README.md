Data Directory
==============

This directory holds the source datasets and any locally downloaded files
needed to run the pipeline. Files here are mounted into the Spark containers
at /opt/spark-data.


Primary Dataset: US Health Insurance Dataset
--------------------------------------------
Source  : https://www.kaggle.com/datasets/teertha/ushealthinsurancedataset
File    : insurance.csv
Format  : CSV, ~1340 rows
License : Public domain (Kaggle)

Columns:
  age      - integer, policyholder age
  sex      - male / female
  bmi      - body mass index, float
  children - number of dependents
  smoker   - yes / no
  region   - northeast / northwest / southeast / southwest
  charges  - medical bill charged by insurer (USD)

Download:
  kaggle datasets download -d teertha/ushealthinsurancedataset
  Place insurance.csv in this folder as:
    data/insurance.csv

This dataset does not include a fraud label. See augmentation_strategy.md
for how fraud labels were derived and justified for this project.


Secondary Dataset: Medicare Claims Synthetic Dataset
----------------------------------------------------
Source  : https://www.cms.gov/Research-Statistics-Data-and-Systems/Downloadable-Public-Use-Files/SynPUFs
File    : DE1_0_2008_Beneficiary_Summary_File_Sample_1.csv (or similar)
Format  : CSV
License : CMS Public Use File

Use in this project:
  - Provides large-scale batch processing volume for Spark justification.
  - Column mapping and integration is documented in augmentation_strategy.md.
  - After downloading, place in:
      data/medicare_claims_sample.csv

Note: For local testing without downloading CMS data, the Kafka producer
generates synthetic streaming events that cover the same schema.


Streaming Dataset
-----------------
Generated synthetically via kafka/producer.py using Python and Faker.
No download required. The producer simulates health insurance claim events
with configurable fraud injection. See kafka/producer.py for field definitions.


Sample File (Included)
----------------------
  data/insurance_sample.csv - 30-row subset of the Kaggle dataset format
    used to verify the feature engineering pipeline locally without
    downloading the full dataset from Kaggle.


Local Data Bootstrap
--------------------
To load insurance.csv into the Spark pipeline for initial batch training
without waiting for Kafka streaming to populate HDFS:

  docker exec -it hdfs-namenode hdfs dfs -mkdir -p /user/health/raw_claims
  docker exec -it spark-master spark-submit \
    --master spark://spark-master:7077 \
    /opt/spark-apps/batch_feature_engineering.py

This reads from HDFS. To seed HDFS with CSV data first, use:
  docker exec -it hdfs-namenode hdfs dfs -put /opt/spark-data/insurance.csv /user/health/source/
Then adapt the batch ETL script to read from /user/health/source/ for the
initial bootstrap run. See docs/architecture.md for the full flow.
