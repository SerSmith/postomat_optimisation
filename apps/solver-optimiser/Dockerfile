FROM python:3.10.6


ENV db_host 178.170.192.244
ENV db_login root
ENV db_passw Optimists1!
ENV db_port 5432
ENV db_name postgres

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
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "80"]
