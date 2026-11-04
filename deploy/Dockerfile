# BikeTools Analyzer server image. Build from the repository root:  docker build -f deploy/Dockerfile -t biketools .
# Python 3.11 is what the tests run on.
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1
WORKDIR /app

COPY requirements.txt requirements-server.txt ./
RUN pip install -r requirements.txt -r requirements-server.txt

COPY app.py ./
COPY src ./src
COPY .streamlit/config.toml ./.streamlit/config.toml

# The app runs without root rights. Provider keys (secrets.toml) are mounted at run time, never baked into the image.
RUN useradd --create-home --uid 1000 biketools && chown -R biketools /app
USER biketools

ENV BIKETOOLS_SERVER_MODE=1
EXPOSE 8501
HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
    CMD python -c "import urllib.request as u; u.urlopen('http://127.0.0.1:8501/_stcore/health', timeout=4)"

# maxUploadSize is in MB. Keep it at or above BIKETOOLS_MAX_UNPACKED_MB only if you accept very large archives.
CMD ["streamlit", "run", "app.py", \
     "--server.address=0.0.0.0", "--server.port=8501", "--server.headless=true", \
     "--server.maxUploadSize=200", "--server.fileWatcherType=none", "--browser.gatherUsageStats=false"]
