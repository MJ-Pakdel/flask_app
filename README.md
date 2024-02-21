# build the app
docker build -t flask_app .
# run docker image: 
docker run -p 5000:5000 flask_app
# go to your browser
http://localhost:5000 or http://[::1]:5000/