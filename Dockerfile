FROM python:3.8

RUN apt-get update
COPY requirements.txt .

RUN pip3 install -r requirements.txt

WORKDIR /workdir
COPY main.py style_transfer_nn.py config.json entrypoint.sh ./
COPY templates /workdir/templates

RUN chmod +x /workdir/entrypoint.sh
ENTRYPOINT "/workdir/entrypoint.sh"