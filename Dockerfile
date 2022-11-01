# 
FROM python:3.10.6

# 
WORKDIR /code

# 
COPY ./ /code/

# 
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt
RUN pip install -e /code/.


# 
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "80"]
