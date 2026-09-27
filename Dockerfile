FROM python:3.11-slim
WORKDIR /app
ENV PYTHONUNBUFFERED=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
RUN python -m grpc_tools.protoc -I . --python_out=. --grpc_python_out=. communication/protocol.proto
CMD ["python", "-m", "client.train_cluster", "--coordinator", "coordinator:50050"]
