FROM python:3.9-slim

WORKDIR /app
# Copy only the lean API requirements
COPY requirements-api.txt .

# Increase pip timeout to avoid network hiccups
ENV PIP_DEFAULT_TIMEOUT=100

# Install only those packages
RUN pip install --no-cache-dir --disable-pip-version-check -r requirements-api.txt


COPY . .

EXPOSE 8000
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
