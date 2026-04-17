CREATE USER hive WITH PASSWORD 'hive123';
CREATE DATABASE metastore OWNER hive;
GRANT ALL PRIVILEGES ON DATABASE metastore TO hive;

CREATE USER airflow WITH PASSWORD 'airflow123';
CREATE DATABASE airflow OWNER airflow;
GRANT ALL PRIVILEGES ON DATABASE airflow TO airflow;

\c metastore
GRANT ALL ON SCHEMA public TO hive;

\c airflow
GRANT ALL ON SCHEMA public TO airflow;
