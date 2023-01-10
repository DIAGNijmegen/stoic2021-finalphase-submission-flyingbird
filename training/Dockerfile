# Edit the base image here, e.g., to use
# TENSORFLOW (https://hub.docker.com/r/tensorflow/tensorflow/)
# or a different PYTORCH (https://hub.docker.com/r/pytorch/pytorch/) base image
FROM pytorch/pytorch

#RUN apt-get update
RUN apt-get update && apt-get install -y git
RUN groupadd -r train && useradd -m --no-log-init -r -g train train

RUN mkdir -p /opt/train /input /output \
    && chown train:train /opt/train /input /output

USER train

WORKDIR /opt/train

ENV PATH="/home/train/.local/bin:${PATH}"

RUN python -m pip install --user -U pip


#COPY --chown=train:train requirements.txt /opt/train/
#RUN python -m pip install --user -r requirements.txt
RUN python -m pip install --user monai
RUN python -m pip install --user git+https://github.com/JoHof/lungmask
RUN python -m pip install --user pandas
RUN python -m pip install --user scipy
RUN python -m pip install --user scikit-learn
RUN python -m pip install --user scikit-image
RUN python -m pip install --user simpleitk
RUN python -m pip install --user tqdm
RUN python -m pip install --user evalutils

COPY --chown=train:train configs/ /opt/train/configs/
COPY --chown=train:train data/ /opt/train/data/
COPY --chown=train:train models/ /opt/train/models/
COPY --chown=train:train preparation/ /opt/train/preparation/
COPY --chown=train:train pretrain/ /opt/train/pretrain/
COPY --chown=train:train main.py /opt/train/
COPY --chown=train:train train.py /opt/train/

ENTRYPOINT python -m main $0 $@



