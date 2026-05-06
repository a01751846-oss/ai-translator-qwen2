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
    torch_dtype=torch.float32
).to("cpu")


# =========================
# Función de traducción (mejorada)
# =========================
def translate_text(text):
    prompt = f"English: {text}\nSpanish:"

    inputs = tokenizer(prompt, return_tensors="pt").to("cpu")

    outputs = model.generate(
        **inputs,
        max_new_tokens=12,        # evita que se alargue
        temperature=0.0,          # cero creatividad
        do_sample=False,          # determinístico
        top_p=1.0,
        repetition_penalty=1.2,
        eos_token_id=tokenizer.eos_token_id
    )

    result = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # limpieza fuerte
    if "Spanish:" in result:
        translated = result.split("Spanish:")[-1]
    else:
        translated = result

    translated = translated.strip()
    translated = translated.split("\n")[0]

    # quitar cosas raras adicionales
    if "English:" in translated:
        translated = translated.split("English:")[0]

    return translated


# =========================
# HTML
# =========================
HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>AI Translator</title>
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

        .result {
            margin-top: 20px;
            background: #334155;
            padding: 10px;
            border-radius: 10px;
        }
    </style>
</head>

<body>
<div class="container">
    <h1>🌍 AI Translator</h1>
    <p>English → Spanish (Qwen)</p>

    <form method="POST">
        <textarea name="text" placeholder="Write in English...">{{ original_text }}</textarea>
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
# Ruta
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
# Run
# =========================
if __name__ == "__main__":
    app.run(debug=True)