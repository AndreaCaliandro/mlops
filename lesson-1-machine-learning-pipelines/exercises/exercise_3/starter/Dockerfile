FROM python:3.8

RUN apt update && \
    apt install -y python3-pip
RUN pip3 install --upgrade pip

COPY requirements.txt ./requirements.txt
RUN pip3 install -r requirements.txt

CMD ['/bin/bash']
