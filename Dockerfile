FROM pytorch/pytorch:2.4.1-cuda12.4-cudnn9-devel

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

# Clone and install fast-hadamard-transform
RUN git clone https://github.com/Dao-AILab/fast-hadamard-transform.git /workspace/third-part/fast-hadamard-transform \
    && pip install /workspace/third-part/fast-hadamard-transform

# Clone and install lm-evaluation-harness
RUN git clone https://github.com/EleutherAI/lm-evaluation-harness.git /workspace/third-part/lm-evaluation-harness \
    && pip install -e /workspace/third-part/lm-evaluation-harness

# Clone DartQuant and install requirements
RUN git clone -b hw_checks https://github.com/yanivbl6/DartQuant.git /workspace/DartQuant \
    && pip install -r /workspace/DartQuant/requirement.txt

WORKDIR /workspace/DartQuant

CMD ["/bin/bash", "--login"]
