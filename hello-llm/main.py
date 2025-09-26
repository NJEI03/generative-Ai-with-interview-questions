from fastapi import FastAPI
from pydantic import BaseModel
from transformers import pipeline  #Easiest way to use

#Creating the app
app = FastAPI()

#Loading the model once the app starts
generator = pipeline('text-generation', model='distilgpt2')