FROM python:3.10.2-slim

RUN pip install -U pip 
RUN pip install pipenv

COPY ["Pipfile", "Pipfile.lock", "./"]

RUN pipenv install --system --deploy 

COPY ["app.py","predict.py","text_clf.h5", "./"]

# EXPOSE 9696

ENTRYPOINT ["gunicorn", "--bind=0.0.0.0:9696", "app:app"]

# ENTRYPOINT ["python3", "app.py"]
# CMD [ "python3", "-m" , "flask", "run", "--host=127.0.0.1"]
# RUN python3 predict.py

# docker pull svizor/zoomcamp-model:3.9.12-slim
# docker images
# docker build -t text_clf:v1 .
# docker run -it --rm -p 9696:9696 ad2493560758
# docker rmi <Image_ID>
