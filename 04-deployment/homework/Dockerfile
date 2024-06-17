FROM agrigorev/zoomcamp-model:mlops-2024-3.10.13-slim
RUN pip install pipenv

WORKDIR /app

COPY Pipfile Pipfile.lock /app/
RUN pipenv install

COPY predict_cli.py /app/predict_cli.py

#CMD ["pipenv", "run", "python", "predict_cli.py", "--year", "2023", "--month", "05"]

ENTRYPOINT ["pipenv", "run", "python", "predict_cli.py"]