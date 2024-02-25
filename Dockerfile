FROM pytorch/pytorch:1.12.1-cuda11.3-cudnn8-runtime
LABEL maintainer="fname.lname@domain.com"

# install opencv åreqs
RUN apt-get update \
    && apt-get install libsm6 libxext6 libgl1-mesa-glx libglib2.0-0 libxrender1 wget --no-install-recommends -y

# remove cache
RUN apt-get autoremove -y \
    && rm -rf /var/lib/apt/lists/*

# set username inside docker
ARG UNAME=user1
ARG UID=1000

# add user UNAME as a member of the sudoers group
RUN useradd -rm --home-dir "/home/$UNAME" --shell /bin/bash -g root -G sudo -u "$UID" "$UNAME"
# activate user
USER "$UNAME"
WORKDIR "/home/$UNAME"

ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1
ENV PATH="/home/$UNAME/miniconda3/bin:${PATH}"
ARG PATH="/home/$UNAME/miniconda3/bin:${PATH}"

# download and install miniconda
RUN wget \
    https://repo.anaconda.com/miniconda/Miniconda3-py38_23.1.0-1-Linux-x86_64.sh \
    && mkdir "/home/$UNAME/.conda" \
    && bash Miniconda3-py38_23.1.0-1-Linux-x86_64.sh -b \
    && rm -f Miniconda3-py38_23.1.0-1-Linux-x86_64.sh

# copy env yaml file
COPY environment.yaml "/home/$UNAME"

# create conda env
RUN conda init bash \
    && conda install -n base conda-libmamba-solver \
    && conda config --set solver libmamba \
    && conda env create -f environment.yaml

# add conda env activation to bashrc
RUN echo "conda activate control" >> ~/.bashrc

# copy all files from current dir except those in .dockerignore
COPY . "/home/$UNAME/ControlNet"

# change file ownership to docker user
USER root
RUN chown -R "$UNAME" "/home/$UNAME/ControlNet"
USER "$UNAME"

# switch to ControlNet dir
WORKDIR "/home/$UNAME/ControlNet"
CMD ["/bin/bash"]