FROM python:3.11-slim
 
# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git build-essential libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*
 
# Create non-root user (HF Spaces requirement)
RUN useradd -m -u 1000 user
USER user
ENV PATH="/home/user/.local/bin:$PATH"
 
WORKDIR /app
 
# Copy and install requirements
COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir --upgrade -r requirements.txt
RUN pip install --no-cache-dir git+https://github.com/openai/CLIP.git
 
# Copy all app files
COPY --chown=user . .
 
# HF Spaces must listen on port 7860
EXPOSE 7860
 
# Start the server
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]