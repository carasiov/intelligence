# CUDA 12 + Python base image
FROM nvidia/cuda:12.2.0-runtime-ubuntu22.04

# Avoid interactive prompts
ENV DEBIAN_FRONTEND=noninteractive

# Install Python and essentials
RUN apt-get update && apt-get install -y \
    python3.11 \
    python3.11-venv \
    python3-pip \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Make python3.11 the default
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1 \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1

# Install Poetry
ENV POETRY_HOME=/opt/poetry
ENV POETRY_VIRTUALENVS_IN_PROJECT=true
ENV PATH="$POETRY_HOME/bin:$PATH"
RUN curl -sSL https://install.python-poetry.org | python3 -

# Create non-root user with UID 1000 (common host UID)
ARG USER_UID=1000
ARG USER_GID=1000
RUN groupadd --gid $USER_GID devuser \
    && useradd --uid $USER_UID --gid $USER_GID -m -s /bin/bash devuser

# Set working directory
WORKDIR /workspace

# Copy project files (context is parent directory)
COPY --chown=devuser:devuser pyproject.toml poetry.lock* README.md ./
COPY --chown=devuser:devuser src/ ./src/
COPY --chown=devuser:devuser scripts/ ./scripts/
COPY --chown=devuser:devuser tests/ ./tests/

# Create results directory
RUN mkdir -p results && chown devuser:devuser results

# Install dependencies
RUN poetry install --no-root --no-interaction

# Switch to non-root user
USER devuser

# Verify JAX can see CUDA (will fail gracefully if no GPU during build)
RUN poetry run python -c "import jax; print(f'JAX devices: {jax.devices()}')" || echo "GPU check skipped during build"

CMD ["bash"]
