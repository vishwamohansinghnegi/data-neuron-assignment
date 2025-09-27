# app.py

from flask import Flask, request, render_template
from sentence_transformers import SentenceTransformer, util

app = Flask(__name__)

# Load the model once when the application starts
print("Loading all-MiniLM-L6-v2 model...")
model = SentenceTransformer('all-MiniLM-L6-v2')
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
            # --- KEY CHANGE: Prediction logic for Sentence-Transformers ---
            # 1. Encode both texts into embeddings
            embeddings = model.encode([text1, text2], convert_to_tensor=True)
            # 2. Compute cosine similarity
            cosine_score = util.cos_sim(embeddings[0], embeddings[1])
            score = f"{cosine_score.item():.4f}"
            
    return render_template("index.html", similarity_score=score, text1=text1, text2=text2)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)