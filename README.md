# Smart Agriculture System using Machine Learning

A Flask web app that recommends crops, detects plant diseases from leaf images, and suggests fertilizers.

## Features
- Crop recommendation from soil and weather values
- Plant disease detection from leaf images (deep learning, PlantVillage dataset)
- Fertilizer recommendation
- AI chat assistant (Groq API)

## Tech Stack
Python, Flask, scikit-learn, TensorFlow/Keras, Pandas, Groq API

## Run locally (Windows PowerShell)
    python -m venv .venv
    .\.venv\Scripts\Activate.ps1
    pip install -r requirements.txt
    copy .env.example .env      # then add your GROQ_API_KEY
    python app.py

## Dataset
PlantVillage dataset (not included, download from Kaggle).

## Screenshots
Add screenshots in assets/screenshots/.