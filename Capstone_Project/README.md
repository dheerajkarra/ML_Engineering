# Capstone Project

This is my Capstone project for ML Zoomcamp from DataTalks.Club.

## Problem description

The main aim of the project is to containarize and deploy the ML pipeline built
after performing EDA and training of the model. 

The dataset used is  of wine reviews of size 50MB and can be
downloaded from this [link](https://www.kaggle.com/datasets/zynicide/wine-reviews) in kaggle website.  
This project is regarding prediction of wine category based on the reviews 
in the text format. Out of all categories, 10 categories of wine 
reviews are considered in this modelling exercise.

This is a multi-class text classification problem where in the 10 categories
are trained using ML (scikit-learn package) and DL (keras) approaches.

## EDA
Extensive EDA such as top categories of the variables, word clouds for all the data
and for each of the categories and visualizations of various
analysis are done. The EDA done is in notebook6d93df5778_EDA.ipynb.

## Model training
The EDA and the training of the model is in Script.py file. Different 
classification algorithms (both ML & DL based) were tried and 
neural network approach was selected as the best model due to higher accuracy. 

The hyperparameter tuning was done to consider different number of
layers for the neural network algorithm. The best model is saved
as text_clf file in the h5 format so that it can be deployed later.

## Exporting notebook to script
A separate script is saved as Script.py file

## Reproducibility
The dataset (winemag-data_first150k.csv) is in the link provided above and 
can be downloaded from Kaggle.
The model file can be generated using Script.py file. Flask app can be 
started using app.py and the predictions can be obtained using predict.py script.

## Model deployment
The model is deployed using Flask

## Dependency and enviroment management
The virutual environment used is pipenv and the dependencies needed for the run 
are in Pipfile and Pipfile.lock. The command to install the dependencies is below:

pipenv install --system --deploy

## Containerization
Containerization is done using Docker and Dockerfile is provided.

## Cloud deployment
The Flask Web App is to be deployed.

## How to Run

1. A new virtual environment is to be created and the required dependencies 
provided in Pipefile are to be installed in it.

Related commands in Pipenv: 
* pipenv shell
* pipenv install <package_name>
* pipenv install --system --deploy

2. The model which is selected as best model can be obtained by running 
Script.py file or it is provided as .h5 file. Please note that the working 
directory needs to be changed for loading the data in this script.

3. In one terminal, Flask app can be started using app.py and the predictions 
can be obtained using predict.py in another terminal.

4. The predictions can also be obtained using the docker container (
app in one terminal and docker run in another terminal) built using the 
below commands:

*  docker images - To get the list of images
*  docker build -t text_clf:v1 .  - To build the docker image from Dockerfile
with name and tag specified as name:tag (Please note the dot at the end)
*  docker run -it --rm -p 9696:9696 ad2493560758 - To run the docker image 
by specifying the port and the image id of the latest build.
'ad2493560758' is an example image id.
*  docker rmi <Image_ID> - To remove the docker image by specifying the image id.

**NOTE:**

The Screenshots of the run are in Output folder.
