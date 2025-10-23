import os
from flask import Flask, render_template, request
from dotenv import load_dotenv
from ollama import Client

load_dotenv()

OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
MODEL_NAME = os.getenv("OLLAMA_MODEL", "llama3.2:3b")
client = Client(host=OLLAMA_HOST)

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def index():
    response_text = None
    user_prompt = ""
    if request.method == "POST":
        user_prompt = request.form.get("prompt", "").strip()
        style = request.form.get("style", "공손하고 간결하게")

        if user_prompt:
            messages = [
                {
                    "role": "system",
                    "content": (
                        f"사용자를 {style}대하면서 답변해줘"
                    ),
                },
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
                        "repeat_penalty": 1.1
                    },
                )
                response_text = (res.get("message") or {}).get("content", "")
            except Exception as e:
                response_text = f"[오류] {e}"

    return render_template(
        "index.html",
        response_text=response_text,
        user_prompt=user_prompt
    )

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=31337, debug=True)