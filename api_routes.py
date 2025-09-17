from fastapi import FastAPI
import random

app = FastAPI()

@app.get("/")
def home():
    return {"message": "Hello, FastAPI!"}


@app.get("/ping")
def ping():
    return {"ping": "pong"}


@app.get("/roll")
async def roll():
    return {"result": random.randint(1, 6)}