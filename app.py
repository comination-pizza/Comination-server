import os
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi import Request
from ollama import Client
load_dotenv()
client = Client(host=os.getenv("OLLAMA_HOST", "http://localhost:11434"))
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")  
MODEL_NAME = os.getenv("OLLAMA_MODEL", "llama3.2:3b")

app=FastAPI()
templates = Jinja2Templates(directory="templates")


@app.get("/", response_class=HTMLResponse) 
async def get(request: Request):
    return templates.TemplateResponse(
        "index.html",
        {"request": request, "response_text": None, "user_prompt": ""}
    )

@app.post("/", response_class=HTMLResponse)
async def post(request: Request):
    form = await request.form()
    user_prompt = form.get("prompt", "").strip()
    style = form.get("style", "공손하고 간결하게")
    response_text = None


    if user_prompt:
        messages = [
            {"role": "system", "content": f"사용자를 {style}대하면서 답변해줘"},
            {"role": "user", "content": user_prompt},
        ]
        try:
            res = client.chat(
                model=MODEL_NAME,
                messages=messages,
                options={
                    "temperature": 0.6,
                    "top_p": 0.9,
                    "num_ctx": 2048,
                    "repeat_penalty": 1.1,
                },
            )
            response_text = (res.get("message") or {}).get("content", "")
        except Exception as e:
            response_text = f"[오류] {e}"

    return templates.TemplateResponse(
        "index.html",
        {"request": request, "response_text": response_text, "user_prompt": user_prompt},
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=31337)