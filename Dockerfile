FROM python:3.12-slim

RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /opt/app

COPY requirements.txt /opt/app/requirements.txt

RUN pip install --no-cache-dir -r requirements.txt

COPY entrypoint.sh /opt/app/entrypoint.sh
COPY app.py /opt/app/app.py

RUN chmod +x /opt/app/entrypoint.sh

ENTRYPOINT ["/opt/app/entrypoint.sh"]
