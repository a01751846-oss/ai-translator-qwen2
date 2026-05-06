from flask import Flask, request, render_template_string
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

app = Flask(__name__)

# =========================
# Cargar modelo Qwen2-0.5B
# =========================
model_name = "Qwen/Qwen2-0.5B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float32,
    device_map="auto"
)

# =========================
# Función de traducción
# =========================
def translate_text(text, target_lang="Spanish"):
    prompt = f"Translate the following text to {target_lang}:\n{text}"

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    outputs = model.generate(
        **inputs,
        max_new_tokens=60,
        temperature=0.7
    )

    result = tokenizer.decode(outputs[0], skip_special_tokens=True)

    return result.replace(prompt, "").strip()


# =========================
# HTML + CSS embebido
# =========================
HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>AI Translator - Qwen2</title>
    <style>
        body {
            font-family: Arial;
            background: #0f172a;
            color: white;
            display: flex;
            justify-content: center;
            align-items: center;
            height: 100vh;
        }

        .container {
            background: #1e293b;
            padding: 30px;
            border-radius: 15px;
            width: 420px;
            text-align: center;
            box-shadow: 0 10px 25px rgba(0,0,0,0.3);
        }

        textarea {
            width: 100%;
            height: 100px;
            border-radius: 10px;
            padding: 10px;
            border: none;
            margin-bottom: 15px;
        }

        button {
            background: #38bdf8;
            border: none;
            padding: 10px 20px;
            border-radius: 10px;
            cursor: pointer;
            font-weight: bold;
        }

        button:hover {
            background: #0ea5e9;
        }

        .result {
            margin-top: 20px;
            background: #334155;
            padding: 10px;
            border-radius: 10px;
        }

        h1 {
            margin-bottom: 5px;
        }

        p {
            opacity: 0.8;
        }
    </style>
</head>

<body>
<div class="container">
    <h1>🌍 AI Translator</h1>
    <p>Powered by Qwen2-0.5B (local)</p>

    <form method="POST">
        <textarea name="text" placeholder="Write text to translate...">{{ original_text }}</textarea>
        <br>
        <button type="submit">Translate</button>
    </form>

    {% if translation %}
    <div class="result">
        <h3>Translation:</h3>
        <p>{{ translation }}</p>
    </div>
    {% endif %}
</div>
</body>
</html>
"""

# =========================
# Ruta principal
# =========================
@app.route("/", methods=["GET", "POST"])
def index():
    translation = ""
    original_text = ""

    if request.method == "POST":
        original_text = request.form["text"]
        translation = translate_text(original_text)

    return render_template_string(
        HTML_TEMPLATE,
        translation=translation,
        original_text=original_text
    )


# =========================
# Run app
# =========================
if __name__ == "__main__":
    app.run(debug=True)