# hash:sha256:4556e5c9dac08c2942363e356153e2a18e6a335259ff3d86e0bb8fdcff7886d7
FROM registry.codeocean.allenneuraldynamics.org/codeocean/tensorflow:2.11-python3.10-cuda11.7.0-runtime-ubuntu22.04

ARG DEBIAN_FRONTEND=noninteractive

ARG GIT_ASKPASS
ARG GIT_ACCESS_TOKEN
COPY git-askpass /

COPY postInstall /
RUN /postInstall
