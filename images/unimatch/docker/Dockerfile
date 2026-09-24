# Define workdir folder for all stages
# Must be renewed in the beginning of each stage
ARG WORKSPACE=/workspace
# SERVICE_NAME identifies the python file with the server
# The python file should have the name ${SERVICE_NAME}_service.py
# All boxes share the same proto file: pipeline.proto
ARG SERVICE_NAME=unimatch

# Pin the vendored UniMatch source (autonomousvision/unimatch, MIT)
ARG UNIMATCH_COMMIT=943a99006950ab6c84ea72a72beb147fdd03c19a

# Flow checkpoint baked into the image (one of the box's defaults per task;
# stereo/depth checkpoints can be added by passing parameters.model with a
# file name / path / URL, or by extending this RUN).
ARG FLOW_WEIGHTS=gmflow-scale1-mixdata-train320x576-4c3a6e9a.pth
ARG WEIGHTS_URL=https://s3.eu-central-1.amazonaws.com/avg-projects/unimatch/pretrained

# --------------------------------------
# Builder stage to generate .proto files
# --------------------------------------

FROM python:3.10-slim@sha256:fd76ade0c607f27677bc04be3c60749f400eedc941d9e72967e19a4cedff80c2 AS builder
# Renew build args
ARG WORKSPACE
ARG SERVICE_NAME

# Path for the protos folder to copy
ARG PROTOS_FOLDER_DIR=protos

RUN pip install --upgrade pip && \
    pip install grpcio grpcio-tools==1.74.0 protobuf

COPY ${PROTOS_FOLDER_DIR} ${WORKSPACE}/
WORKDIR ${WORKSPACE}

# Compile the shared proto file
RUN python -m grpc_tools.protoc -I. --python_out=. --grpc_python_out=. pipeline.proto

# -----------------------------
# Stage to generate final image
# -----------------------------

FROM python:3.10-slim@sha256:fd76ade0c607f27677bc04be3c60749f400eedc941d9e72967e19a4cedff80c2
# Renew build args
ARG WORKSPACE
ARG SERVICE_NAME

ARG USER=runner
ARG GROUP=runner-group
ARG SRC=src
ARG UNIMATCH_COMMIT
ARG FLOW_WEIGHTS
ARG WEIGHTS_URL

# Create non-privileged user and workspace
RUN addgroup --system ${GROUP} && \
    adduser --system --no-create-home --ingroup ${GROUP} ${USER} && \
    mkdir ${WORKSPACE} && \
    chown -R ${USER}:${GROUP} ${WORKSPACE}

# Torch/torchvision-style cache inside the (owned) workspace
ENV TORCH_HOME=/workspace/.torch_cache
ENV XDG_CACHE_DIR=/workspace/.xcache

# Install requirements (torch default wheels carry CUDA runtime libs)
COPY requirements.txt .
RUN apt update -y && \
    apt install -y --no-install-recommends pip git ca-certificates && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt && \
    rm requirements.txt

# Vendor the UniMatch source (pinned commit — self-contained build, no
# runtime dependency on the repo's host)
RUN git clone --depth 1 https://github.com/autonomousvision/unimatch.git ${WORKSPACE}/unimatch && \
    cd ${WORKSPACE}/unimatch && \
    ([ "$(git rev-parse HEAD)" = "${UNIMATCH_COMMIT}" ] || \
     (git fetch --depth 1 origin ${UNIMATCH_COMMIT} && git checkout ${UNIMATCH_COMMIT})) && \
    rm -rf .git

# Bake in the default flow checkpoint (~20 MB) so the box works out of the box
RUN mkdir -p ${WORKSPACE}/pretrained && \
    python -c "import urllib.request; urllib.request.urlretrieve('${WEIGHTS_URL}/${FLOW_WEIGHTS}', '${WORKSPACE}/pretrained/${FLOW_WEIGHTS}')" && \
    # runtime (runner) must be able to write further parameters.model downloads here
    chown -R ${USER}:${GROUP} ${WORKSPACE}/pretrained

# COPY .proto file to root to meet ai4eu specifications
COPY --from=builder --chown=${USER}:${GROUP} ${WORKSPACE}/pipeline.proto /

# Copy generated .py files to workspace
COPY --from=builder --chown=${USER}:${GROUP} ${WORKSPACE}/*.py ${WORKSPACE}/

# Copy the service file and the utils to workspace
# (rename service file to only service.py for generic usage)
COPY --chown=${USER}:${GROUP} ${SRC} ${WORKSPACE}
COPY --chown=${USER}:${GROUP} ${SRC}/${SERVICE_NAME}_service.py ${WORKSPACE}/service.py

# Change to non-privileged user
USER ${USER}

# Expose port 8061 according to ai4eu specifications
EXPOSE 8061

WORKDIR ${WORKSPACE}

CMD ["python3", "service.py"]
