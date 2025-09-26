from fastapi import FastAPI
from pydantic import BaseModel
from transformers import pipeline

app = FastAPI()

#Loading the summarization pipeline
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")