FROM python:3.12-slim


WORKDIR /app

COPY wheelhouse /app/wheelhouse
COPY requirements-api.txt .

RUN pip install --no-index --find-links wheelhouse \
    --disable-pip-version-check -r requirements-api.txt

COPY . .

EXPOSE 8000
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
