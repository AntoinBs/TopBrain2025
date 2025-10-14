FROM python:3.11-slim

RUN pip install --no-cache-dir mlflow==3.4.0

WORKDIR /mlflow

RUN mkdir -p /mlflow/mlruns

EXPOSE 5000

CMD ["mlflow","server",\
     "--host","0.0.0.0","--port","5000",\
     "--backend-store-uri","sqlite:///mlflow.db",\
     "--artifacts-destination","file:///mlflow/mlruns"]