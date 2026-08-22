FROM huggingface/transformers-all-latest-gpu:latest

ENV LANG=C.UTF-8
ENV LC_ALL=C.UTF-8

RUN python3 -m pip install --no-cache-dir \
    ipykernel \
    ipywidgets \
    peft==0.20.0

    WORKDIR /workspace