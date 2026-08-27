FROM python:3.11-slim-bookworm

RUN apt-get update \
 && apt-get install -y --no-install-recommends \
      ffmpeg git libgl1 libglib2.0-0 ca-certificates curl \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# rtmlib depends on plain opencv-python while we standardise on
# opencv-contrib-python (required by mediapipe). Removing the duplicate also
# deletes files shared by both wheels, so contrib is force-restored after.
RUN pip uninstall -y opencv-python || true
RUN pip install --no-cache-dir --force-reinstall --no-deps \
      opencv-contrib-python==4.10.0.84

COPY src/ ./src/
COPY scripts/ ./scripts/
COPY corpus/ ./corpus/
COPY configs/ ./configs/
COPY foot_contact_annotator.html ./foot_contact_annotator.html

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app/src

# Output/persistent dirs exist even without mounts (mirrors toolbox convention).
RUN mkdir -p /work/auto-sfx /cache/auto-sfx/models

ENTRYPOINT ["python", "-m", "auto_sfx"]
