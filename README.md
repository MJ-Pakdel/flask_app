# build the app
docker build -t flask_app .
# run docker image: 
docker run -p 5000:5000 flask_app
# go to your browser
http://localhost:5000 or http://[::1]:5000/

hello1

# to destroy terraform
terraform destroy

# configure docker 
aws ecr get-login-password --region 'us-east-1' | docker login --username AWS --password-stdin 153295639067.dkr.ecr.us-east-1.amazonaws.com


docker tag flask_app:latest 153295639067.dkr.ecr.us-east-1.amazonaws.com/flask_deployment-prod-ecr:latest

docker push 153295639067.dkr.ecr.us-east-1.amazonaws.com/flask_deployment-prod-ecr:latest
