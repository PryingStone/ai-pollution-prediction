# Project Context

Project Name: AirSenseAI

## Goal

Build an AI-powered AQI forecasting system for Karnataka cities.

## Scope

The system should:

* Fetch real-time AQI data
* Fetch weather data
* Clean and process data
* Train a Random Forest model
* Predict AQI for the next 24 hours
* Display AQI trends
* Display pollution risk levels
* Provide health recommendations
* Show results on a Streamlit dashboard

## Cities

Support Karnataka cities that have AQI monitoring stations.

## Data Sources

AQI:

* CPCB historical data
* WAQI real-time data

Weather:

* OpenWeather API

## Machine Learning

Model:

* Random Forest Regressor

Evaluation:

* MAE
* RMSE
* R² Score

## Responsible AI

* No user accounts
* No personal health data
* No medical diagnosis
* Recommendations are advisory only

## Development Rules

Keep architecture simple.

This is a second-semester engineering project.

Avoid:

* Docker
* Kubernetes
* Microservices
* Authentication systems
* Complex cloud infrastructure

Always explain code and file purpose because the project owner is a beginner.


## Secrets

Never hardcode API keys.

Use:
- .env
- .env.example

API keys must never be committed.