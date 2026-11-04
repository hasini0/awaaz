# Works on Hugging Face Spaces (Docker SDK), Render, or any container host.
FROM python:3.11-slim
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cpu -r requirements.txt
COPY . .
# pre-download models into the image so the first request is fast
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('intfloat/multilingual-e5-small')" \
 && python -c "from awaaz.asr import _whisper; _whisper('openai/whisper-small'); _whisper('vasista22/whisper-telugu-small')"
ENV PORT=7860
EXPOSE 7860
CMD streamlit run app.py --server.port $PORT --server.address 0.0.0.0 --server.headless true
