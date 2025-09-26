# app.py

from flask import Flask, request, render_template
from sentence_transformers.cross_encoder import CrossEncoder

app = Flask(__name__)

# Load the model once when the application starts
print("Loading Cross-Encoder model...")
model = CrossEncoder('cross-encoder/stsb-roberta-large')
print("Model loaded successfully.")

@app.route("/", methods=["GET", "POST"])
def index():
    """
    Handles both displaying the form (GET) and processing the
    form submission (POST).
    """
    score = None
    text1 = ""
    text2 = ""
    
    if request.method == "POST":
        text1 = request.form.get("text1", "").strip()
        text2 = request.form.get("text2", "").strip()
        
        if text1 and text2:
            raw_score = model.predict((text1, text2), show_progress_bar=False)
            score = f"{float(raw_score):.4f}"
            
    # This will look for a file named "index.html" inside a "templates" folder.
    return render_template("index.html", similarity_score=score, text1=text1, text2=text2)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)