FROM apache/spark:3.4.1-scala2.12-java11-python3-r-ubuntu

USER root

RUN apt-get update && apt-get install -y python3-pip

RUN pip3 install --no-cache-dir \
    numpy \
    pandas \
    scikit-learn \
    matplotlib \
    seaborn \
    shap

USER spark