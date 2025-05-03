# Use Python 3.10 as the base image since it's well-supported and stable
FROM python:3.10-slim

# Set working directory in the container
WORKDIR /app

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8000

# Copy requirements first for better layer caching
COPY mcp-server/requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code to the container
COPY mcp-server/ .

# Expose the port the app runs on
EXPOSE 8000

# Set up healthcheck
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
  CMD curl -f http://localhost:${PORT}/health || exit 1

# Command to run the application
# Use Gunicorn for production-ready ASGI server
CMD ["python", "server.py"]