FROM py-audio-box:reference-quality-tests
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_DISABLE_PIP_VERSION_CHECK=1
RUN pip install --no-cache-dir faster-whisper==1.2.1 ctranslate2==4.8.3
WORKDIR /evaluation
ENTRYPOINT []
CMD ["python", "/evaluation/benchmark.py", "--help"]
