FROM python:3.12-slim

# LiteRT's XNNPACK backend links against OpenMP, which the slim image omits.
# Without this, importing the interpreter fails with
# "libgomp.so.1: cannot open shared object file".
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Run as a non-root user. Several hosts (Hugging Face Spaces among them) run
# containers as UID 1000, so create that user and let it own the app directory.
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    PYTHONUNBUFFERED=1

WORKDIR $HOME/app

# Copy requirements alone first: this layer is cached and skipped on rebuilds
# that only touch application code.
COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

COPY --chown=user . .

# Most platforms inject $PORT; 8080 is a sane default for those that don't.
ENV PORT=8080
EXPOSE 8080

# One worker keeps a single copy of the TFLite interpreter in memory. The
# threads absorb concurrent viewers; app.py serialises invoke() with a lock,
# because a TFLite Interpreter is not thread-safe. Shell form, so $PORT expands.
CMD gunicorn -w 1 --threads 4 -b 0.0.0.0:$PORT --timeout 120 app:app
