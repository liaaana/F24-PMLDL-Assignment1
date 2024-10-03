# Wine Quality Prediction Model Deployment

## Overview

This repository contains the implementation of a machine learning model for predicting wine quality using a Random Forest classifier. The model is deployed as an API using FastAPI and is accompanied by a web application built with Streamlit. The deployment is managed using Docker containers.

## Dataset

- **Load the trained model**: [Wine dataset](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_wine.html#sklearn.datasets.load_wine)
- **Model Selection**: The Random Forest classifier was selected for its superior performance compared to other algorithms, as noted in the dataset documentation [here](https://archive.ics.uci.edu/dataset/109/wine).

## Technologies Used

- **Machine Learning Model:** Random Forest classifier
- **API Framework:** FastAPI
- **Web Application Framework:** Streamlit
- **Containerization:** Docker
- **Container Orchestration:** Docker Compose

## Instructions

1. Clone the repository to your local machine.
2. Navigate to the `code/deployment` directory.
3. Build and run the Docker containers using the following command:
```bash
docker-compose up --build
```