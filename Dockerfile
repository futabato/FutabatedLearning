FROM pytorch/pytorch:2.0.0-cuda11.7-cudnn8-devel

COPY pyproject.toml poetry.lock /
RUN pip install --upgrade pip && pip install poetry \
    && poetry config virtualenvs.create false \
    && poetry install --no-root

WORKDIR /workspace