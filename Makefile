
#################### PACKAGE ACTIONS ###################
# git:
# git add .
# git commit -m "$m"
# git push -u origin master

install_requirements:
	@pip install -r requirements.txt

run_api:
	uvicorn stories_generator.api.fast:app --reload

streamlit:
	-@streamlit run ./project_website/app.py

##### Prediction API - - - - - - - - - - - - - - - - - - - - - - - - -
run_api:
    uvicorn fast_api.api:app --reload
##### Docker - - - - - - - - - - - - - - - - - - - - - - - - -
docker_build:
    docker build -t template-image-api .
docker_run:
    docker run -p 8000:8000 --name api template-image-api
##### GCP - - - - - - - - - - - - - - - - - - - - - - - - -
GCP_PROJECT_ID=########
DOCKER_IMAGE_NAME=#####
# https://cloud.google.com/storage/docs/locations#location-mr
GCR_MULTI_REGION=eu.gcr.io
# https://cloud.google.com/compute/docs/regions-zones#available
REGION=eu-west1
build_gcr_image:
    docker build -t $(GCR_MULTI_REGION)/$(GCP_PROJECT_ID)/$(DOCKER_IMAGE_NAME) .
build_gcr_image_m1:
    docker build --platform linux/amd64 -t $(GCR_MULTI_REGION)/$(GCP_PROJECT_ID)/$(DOCKER_IMAGE_NAME) .
run_gcr_image:
    docker run -v=${HOME}/.config/gcloud:/root/.config/gcloud --env PROJECT_ID=$(GCP_PROJECT_ID) \
    --env PORT=8000 -p 8080:8000 $(GCR_MULTI_REGION)/$(GCP_PROJECT_ID)/$(DOCKER_IMAGE_NAME)
push_gcr_image:
    docker push $(GCR_MULTI_REGION)/$(GCP_PROJECT_ID)/$(DOCKER_IMAGE_NAME)
gcr_deploy:
    gcloud run deploy --image $(GCR_MULTI_REGION)/$(GCP_PROJECT_ID)/$(DOCKER_IMAGE_NAME) --platform managed --region $(REGION)
