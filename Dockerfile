FROM python:3.11-slim

WORKDIR /opt
RUN mkdir xgboost

COPY src/train.py xgboost
COPY src/datasets xgboost

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

ENTRYPOINT [ "python", "xgboost/train.py" ]