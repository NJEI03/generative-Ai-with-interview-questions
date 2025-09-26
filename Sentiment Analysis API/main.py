from fastapi import FastAPI
from pydantic import BaseModel
from transformers import pipeline

app = FastAPI()
#Load the sentimnt analysis model

classifier = pipeline("sentiment-analysis", model ="distilbert-base-uncased-finetuned-sst-2-english")

class sentimentRequest(BaseModel):
    text: str