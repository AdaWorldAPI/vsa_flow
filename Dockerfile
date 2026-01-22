FROM python:3.11-slim

WORKDIR /app

# Copy everything first
COPY . .

# Install
RUN pip install --no-cache-dir -e .

# Run on 8080
EXPOSE 8080

CMD ["uvicorn", "vsa_flow.api.main:app", "--host", "0.0.0.0", "--port", "8080"]
