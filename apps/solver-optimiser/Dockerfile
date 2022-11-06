FROM python:3.10.6




RUN apt-get update -y

RUN apt-get install -y coinor-cbc coinor-libcbc-dev

WORKDIR /code

COPY apps/solver-optimiser/ /code/app
COPY src/ /code/src/
COPY requirements.txt /code/
COPY setup.py /code/

# 
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt
RUN pip install -e /code/.


# 
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "80", "--h11-max-incomplete-event-size", "1638400"]

