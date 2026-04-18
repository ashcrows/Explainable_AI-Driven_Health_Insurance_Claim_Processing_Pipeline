FROM apache/spark:3.4.4
USER root

# apache/spark:3.4.4 ships Python 3.8 — use versions compatible with it
RUN pip install --no-cache-dir \
    numpy==1.24.4 \
    pandas==2.0.3 \
    pyarrow==12.0.1 \
    scikit-learn==1.3.2 \
    shap==0.43.0 \
    matplotlib==3.7.5

USER root
