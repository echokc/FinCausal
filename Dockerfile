FROM python:3.10-slim
RUN pip install pandas numpy scipy scikit-learn statsmodels --no-cache-dir
WORKDIR /app
