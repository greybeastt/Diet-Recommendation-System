FROM python

COPY ./requirements.txt .

RUN pip install -r requirements.txt

WORKDIR /app

COPY . .

EXPOSE 8080

CMD ["uvicorn","main:app","--host","0.0.0.0","--port","8080"]