#
FROM python:3.10

#
WORKDIR /app

#
COPY ./requirements.txt /app/requirements.txt

#
RUN pip install --no-cache-dir --upgrade -r /app/requirements.txt

#
COPY ./main.py /app/main.py

#
EXPOSE 8000

#
CMD ["fastapi", "run", "app/main.py", "--port", "8000"]