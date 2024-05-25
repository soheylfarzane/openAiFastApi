from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from openai import OpenAI


app = FastAPI()

class ChatPayload(BaseModel):
    prompt: str = None
    api_key: str = None
    model: str = None
    messages: list = None

@app.post("/chat")
async def chat(payload: ChatPayload):
    if not payload.api_key:
        raise HTTPException(status_code=400, detail="API key is missing")

    if not payload.model:
        payload.model = "gpt-3.5-turbo-0125"

    if not payload.messages:
        payload.messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": payload.prompt},
        ]

    client = OpenAI(api_key=payload.api_key)
    response = client.chat.completions.create(
        model=payload.model,
        messages=payload.messages,
    )

    plain_answer = response.choices[0].message.content
    payload.messages.append({"role": "assistant", "content": plain_answer})

    organized_response = {
        "conversation": [
            {"role": msg["role"], "content": msg["content"]} for msg in payload.messages
        ],
        "answer": plain_answer
    }

    return organized_response
