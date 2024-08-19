FROM python:3.11-slim

WORKDIR /opt
RUN mkdir xgboost
RUN mkdir xgboost/models

COPY src/train.py xgboost
COPY src/datasets xgboost/datasets

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

WORKDIR /opt/xgboost

ENTRYPOINT [ "python", "train.py" ]