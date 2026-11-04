# Works on Hugging Face Spaces (Docker SDK), Render, or any container host.
FROM python:3.11-slim
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg && rm -rf /var/lib/apt/lists/*

# Hugging Face Spaces run the container as uid 1000, so everything the app writes lives in its home
RUN useradd -m -u 1000 user
ENV HOME=/home/user HF_HOME=/home/user/.cache/huggingface PORT=7860
WORKDIR /home/user/app

COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cpu -r requirements.txt
COPY --chown=user . .
USER user

# bake the models and the demo database into the image, so the first visit is fast
RUN python -c "from awaaz.classify import default_embedder; default_embedder()" \
 && python -c "from awaaz.asr import preload; preload()" \
 && python scripts/seed_db.py

EXPOSE 7860
CMD streamlit run app.py --server.port $PORT --server.address 0.0.0.0 --server.headless true
