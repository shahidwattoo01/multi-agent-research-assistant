FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    RA_MEMORY_DIR=/data/memory

WORKDIR /app
COPY pyproject.toml README.md LICENSE ./
COPY src ./src
RUN pip install ".[api,autogen]" && useradd --create-home app && mkdir -p /data && chown app /data

USER app
VOLUME ["/data"]
EXPOSE 8000
CMD ["uvicorn", "research_assistant.api:app", "--host", "0.0.0.0", "--port", "8000"]
