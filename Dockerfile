# 
FROM python:3.10.6

#
ENV db_host 178.170.192.244
ENV db_login root
ENV db_passw Optimists1!
ENV db_port 5432
ENV db_name postgres

# 
WORKDIR /code

# 
COPY ./ /code/

# 
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt
RUN pip install /code/.


# 
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "80"]
