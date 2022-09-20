FROM python:3.8 as py38

ENV PYTHONUNBUFFERED=1 \
    POETRY_VERSION=1.1.14

WORKDIR /app

RUN pip install "poetry==$POETRY_VERSION"

COPY feature_draft ./feature_draft

COPY poetry.lock pyproject.toml ./

RUN poetry config virtualenvs.create false \
  && poetry install

COPY tests ./tests

CMD ["pytest", "tests/."]