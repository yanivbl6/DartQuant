FROM pytorch/pytorch:2.4.1-cuda12.4-cudnn9-devel

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    git \
    openssh-client \
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

RUN rm -rf /workspace/DartQuant/data && ln -s /data/users/yanivbl /workspace/DartQuant/data

RUN cd /workspace/DartQuant/ && pip install -r /workspace/DartQuant/requirements.txt

RUN useradd -ms /bin/bash yanivbl \
    && chown -R yanivbl:yanivbl /workspace

USER yanivbl

WORKDIR /workspace/DartQuant

# HF cache on persistent mount (datasets pre-populated via scripts/download_lm_eval_datasets.py)
ENV HF_HOME=/data/data/huggingface
RUN git config --global user.email "yanivblm6@gmail.com"
RUN git config --global user.name "Yaniv Blumenfeld"

CMD ["/bin/bash", "--login"]
