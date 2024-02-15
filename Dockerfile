FROM python:3.11-slim-bullseye

COPY ./requirements.txt .

# RUN apt-get update && apt-get install -y 
RUN pip3 install --upgrade pip

RUN pip3 --no-cache-dir install -r requirements.txt

WORKDIR /src
COPY . /src
# COPY ./server.py .

HEALTHCHECK CMD curl --fail http://localhost:8000/_stcore/health

CMD ["uvicorn", "server:app","--host", "0.0.0.0", "--port", "8000"]