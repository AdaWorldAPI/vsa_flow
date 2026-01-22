FROM python:3.11-slim

WORKDIR /app

# Copy everything first
COPY . .

# Install dependencies
RUN pip install --no-cache-dir -e .

# Run
ENV PORT=8000
EXPOSE 8000

CMD ["uvicorn", "vsa_flow.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
