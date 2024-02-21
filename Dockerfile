
FROM python:3.11-buster
RUN pip install poetry
COPY . .
RUN poetry install
EXPOSE 5000  

# line below assumes that there is a file called entry.py in the root of the project
ENTRYPOINT ["poetry", "run", "python", "-m", "run"]
# docker build -t flask_app:v0 .
# docker run flask_app:v0

