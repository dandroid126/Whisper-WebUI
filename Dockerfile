FROM debian:bookworm-slim AS builder

RUN apt-get update && \
    apt-get install -y curl git python3 python3-pip python3-venv && \
    rm -rf /var/lib/apt/lists/* /var/cache/apt/archives/* && \
    mkdir -p /Whisper-WebUI

WORKDIR /Whisper-WebUI

COPY requirements.txt constraints.txt ./

# Upstream moved the wheel index to cu128 for Blackwell (jhj0517/Whisper-WebUI#619), but the cu128
# wheels dropped Pascal. On a GTX 1080 Ti (sm_61) every torch backed feature -- Silero VAD, UVR
# background music separation, diarization -- then dies at runtime with "no kernel image is
# available for execution on the device", while plain transcription keeps working because
# faster-whisper runs on CTranslate2. Rewrite the index rather than editing requirements.txt, so
# this stays a build time choice and the file itself matches upstream.
ARG TORCH_CUDA=cu126
RUN sed -i "/^--extra-index-url/s|/whl/cu[0-9]\+|/whl/${TORCH_CUDA}|" requirements.txt

RUN python3 -m venv venv && \
    . venv/bin/activate && \
    PIP_CONSTRAINT=constraints.txt PIP_BUILD_CONSTRAINT=constraints.txt pip install -U -r requirements.txt && \
    python -c "import torch; assert '+${TORCH_CUDA}' in torch.__version__, 'expected a ${TORCH_CUDA} build, got ' + torch.__version__"



FROM debian:bookworm-slim AS runtime

RUN apt-get update && \
    apt-get install -y curl ffmpeg python3 && \
    rm -rf /var/lib/apt/lists/* /var/cache/apt/archives/*

WORKDIR /Whisper-WebUI

COPY . .
COPY --from=builder /Whisper-WebUI/venv /Whisper-WebUI/venv

VOLUME [ "/Whisper-WebUI/models" ]
VOLUME [ "/Whisper-WebUI/outputs" ]

ENV PATH="/Whisper-WebUI/venv/bin:$PATH"
ENV LD_LIBRARY_PATH=/Whisper-WebUI/venv/lib64/python3.11/site-packages/nvidia/cublas/lib:/Whisper-WebUI/venv/lib64/python3.11/site-packages/nvidia/cudnn/lib

ENTRYPOINT [ "python", "app.py" ]
