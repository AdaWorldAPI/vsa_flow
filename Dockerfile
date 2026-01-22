FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY pyproject.toml .
RUN pip install --no-cache-dir .

# Copy source
COPY vsa_flow/ vsa_flow/

# Run on 8080 (Railway standard)
ENV PORT=8080
EXPOSE 8080

CMD ["uvicorn", "vsa_flow.api.main:app", "--host", "0.0.0.0", "--port", "8080"]
