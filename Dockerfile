FROM vllm/vllm-openai:latest
WORKDIR /app

# Create a non-root user (1000:1000)
RUN addgroup --gid 1000 appgroup && \
    adduser --uid 1000 --gid 1000 --disabled-password --gecos "" appuser

# Install dependencies
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# Copy project fiels
COPY experiments /app/experiments
COPY common /app/common
COPY src /app/src

# Copy executables
COPY main.py /app/main.py
COPY entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh

# Set premissions and switch to new appuser
RUN chown -R 1000:1000 /app
USER appuser

ENTRYPOINT ["/app/entrypoint.sh"]