from langchain_community.utilities import WikipediaAPIWrapper
from langchain.chains import LLMChain, sequential
from langchain.prompts import PromptTemplate
from langchain_community.llms import huggingface_pipeline
from transformers import pipeline
import re

# Initialize components

wiki = WikipediaAPIWrapper()

# LLM for sumarization and sentiment
llm_pipeline = pipeline(
    "text-generation",
    model = "google/flan-t5-small", # better for instructions than distilgpt2
    max_length=200
)
llm = huggingface_pipeline(pipeline=llm_pipeline)

# Step 2: Wikipedia Research

research_prompt = PromptTemplate(
    input_variable = ["topic"],
    template =  " Research this topic and provide the key facts: {topic}"
)

# Step 2: Summarization
summary_prompt = PromptTemplate(
    input_variables=["research"],
    template="Summarize this in 3 bullet points: {research}"
)

# Step 3: Sentiment Analysis
sentiment_prompt = PromptTemplate(
    inpute_variable=['summary'],
    template="Analyze the sentiment of this text (positive/negative/neutral):{summary}"
)

# Create the Chain
research_chain = LLMChain(llm=llm, prompt =research_prompt, output_key ="research")

summary_chain = LLMChain(llm=llm, prompt= summary_prompt, output_key="summary")

sentiment_chain = LLMChain(llm=llm, prompt = sentiment_prompt, output_key="sentiment")