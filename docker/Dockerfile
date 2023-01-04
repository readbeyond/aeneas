FROM python:3.5

RUN apt update && \
    apt install -y ffmpeg espeak && \
    apt install -y libespeak-dev && \
    pip install numpy && \
    pip install aeneas && \

    wget https://espeak.sourceforge.net/data/ru_dict-48.zip && \
    unzip ru_dict-48.zip && \
    cp ru_dict-48 /usr/lib/x86_64-linux-gnu/espeak-data/ru_dict