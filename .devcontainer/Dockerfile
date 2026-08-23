FROM huggingface/transformers-all-latest-gpu:latest

ENV LANG=C.UTF-8
ENV LC_ALL=C.UTF-8

COPY .devcontainer/requirements.txt /tmp/requirements.txt

RUN --mount=type=cache,target=/root/.cache/pip \
    python3 -m pip install --upgrade -r /tmp/requirements.txt

WORKDIR /workspace