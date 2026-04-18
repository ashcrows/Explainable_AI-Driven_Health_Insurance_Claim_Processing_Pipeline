FROM apache/spark:3.4.4
USER root

# Install Python ML dependencies system-wide so spark-submit subprocesses can see them
RUN pip install --no-cache-dir \
    numpy==1.26.2 \
    pandas==2.1.3 \
    pyarrow==14.0.1 \
    scikit-learn==1.3.2 \
    shap==0.44.0 \
    matplotlib==3.8.2

USER root
