FROM ubuntu:latest
RUN  apt-get update && \
     apt-get install -y python3 python3-pip && \
     pip install tensorflow gym
ENTRYPOINT ["echo", "hello"]
