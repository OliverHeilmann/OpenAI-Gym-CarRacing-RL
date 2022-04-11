FROM bath:2020-gpu
WORKDIR /code

RUN apt-get update
RUN apt-get install ffmpeg libsm6 libxext6 -y
RUN apt install xvfb -y
COPY requirements.txt requirements.txt
RUN pip3 install -r requirements.txt

# Dependencies for glvnd and X11.
RUN apt-get update && apt-get install -y -qq --no-install-recommends \
    libxext6 \
    libx11-6 \
    libglvnd0 \
    libgl1 \
    libglx0 \
    libegl1 \
    freeglut3-dev && rm -rf /var/lib/apt/lists/*

# Env vars for the nvidia-container-runtime.
ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES graphics,utility,compute

# Stop display with gym. 
RUN echo "export DISPLAY=:0"  >> /etc/profile

# Add gym.
RUN pip install --upgrade pip && pip install gym==0.17.2 box2d==2.3.10
