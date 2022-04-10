FROM bath:2020-gpu
WORKDIR /code
COPY requirements.txt requirements.txt
RUN pip3 install -r requirements.txt

