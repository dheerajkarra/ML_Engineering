# Mid Term Project

This is my Mid Term project for ML Zoomcamp from DataTalks.Club

## Problem description

The main aim of the project is to containarize and deploy the ML pipeline built
after performing EDA and training of the model. 

The dataset used is of Credit card Fraud Detection of size 150MB and can be
downloaded from this link in kaggle website [link](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) 
The probability that a transaction is fraud using the features provided is obtained
 using this ML modelling.

This is a imbalanced classification problem in which the distribution of response
variable is skewed in the sense that the number of frauds are very much less
than non-frauds in the dataset.

Different classification algorithms are tried along with Hyperparameter tuning
and Logistic Regression was obtained as the best model. SMOTE algorithm was 
used to deal with the imbalance of response variable. 

## EDA
Extensive EDA such as distributions of the variables, correlation analysis, visualizations of various
analysis are done. Basic steps such as missing values, min-max values are also done.
Outlier analysis (using IQR or other methods) and feature selection 
to select variables among V1 to V2 (using mutual info score from sklearn) 
can be done as next steps.

## Model training
The EDA and the training of the model is in Script.py file. Different 
classification algorithms (both linear & tree based) were tried and 
Logistic Regression was selected as the best model. The hyperparameter tuning was
also done for these alogirthms. The best model is saved
as .bin file in the pickle format so that it can be deployed later.

## Exporting notebook to script
A separate script is saved as Script.py file

## Reproducibility
The dataset is in the link provided above and can be downloaded.
The model file can be generated using Script.py file. Flask app can be 
started using app.py and the predictions can be obtained using predict.py script.

## Model deployment
The model is deployed using Flask

## Dependency and enviroment management
The virutual environment used is pipenv and the dependencies needed for the run 
are in Pipfile. The command to install the dependencies is below:

pipenv install --system --deploy

## Containerization
Containerization is done using Docker and Dockerfile is provided.

**Cloud deployment**
This is to be updated

## How to Run

1. A new virtual environment is to be created and the required dependencies 
provided in Pipefile are to be installed in it.

Related commands in Pipenv: 
* pipenv shell
* pipenv install <package_name>
* pipenv install --system --deploy

2. The model which is selected as best model can be obtained by running 
Script.py file or it is provided as .bin file.

3. In one terminal, Flask app can be started using app.py and the predictions 
can be obtained using predict.py in another terminal.

4. The predictions can also be obtained using the docker container (
app in one terminal and docker run in another terminal) built using the 
below commands:

a) docker images - To get the list of images
b) docker build -t cc_fraud:v1 .  - To build the docker image from Dockerfile
with name and tag specified as name:tag
c) docker run -it --rm -p 9696:9696 a2c3e4d50472 - To run the docker image 
by specifying the port and the image id of the latest build.
'a2c3e4d50472' is an example image id.
d) docker rmi <Image_ID> - To remove the docker image by specifying the image id.

**NOTE:**
The Screenshots of the run are in Output folder.
The plots generated during EDA are in Plots folder.
