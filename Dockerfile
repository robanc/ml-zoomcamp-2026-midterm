
FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY predict.py .
COPY models/xgboost_signal_model.bin models/xgboost_signal_model.bin

EXPOSE 9696

CMD ["waitress-serve", "--host=0.0.0.0", "--port=9696", "predict:app"]
