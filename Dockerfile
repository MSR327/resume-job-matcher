# Local/self-hosted run of the app. Torch-free: both models run with ONNX Runtime on CPU.
#   docker build -t jobmatch . && docker run -p 8501:8501 -e GROQ_API_KEY=... jobmatch
FROM python:3.12-slim
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends git && rm -rf /var/lib/apt/lists/*
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY streamlit_app.py ./
COPY pages ./pages
COPY src ./src
RUN useradd -m app && chown -R app /app
USER app
EXPOSE 8501
HEALTHCHECK --interval=30s --timeout=5s --start-period=120s \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8501/_stcore/health')"
CMD ["streamlit", "run", "streamlit_app.py", "--server.address=0.0.0.0", "--server.port=8501", "--server.headless=true"]
