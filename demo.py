# run_similarity_simple.py

from sentence_transformers.cross_encoder import CrossEncoder

# --- 1. Load the model once when the script starts ---
print("Loading the Cross-Encoder model...")
model = CrossEncoder('cross-encoder/stsb-roberta-large')
print("Model loaded successfully.\n")


def get_similarity_score(text1: str, text2: str) -> float:
    """
    Takes a single pair of texts and returns their similarity score.
    This is a simple, single-purpose function.

    Args:
        text1 (str): The first piece of text.
        text2 (str): The second piece of text.

    Returns:
        float: The similarity score.
    """
    # The model's predict method calculates the score for the given pair.
    score = model.predict((text1, text2), show_progress_bar=False)
    return float(score)


if __name__ == '__main__':
    # --- 2. DEFINE INPUT DATA ---
    input_data = [
        {
            "text1": "The International Space Station is in low Earth orbit.",
            "text2": "Orbiting the Earth is a large satellite called the ISS."
        },
        {
            "text1": "The team celebrated their victory.",
            "text2": "The team was disappointed by their loss."
        },
        {
            "text1": "Baking a perfect sourdough loaf requires a patient technique.",
            "text2": "The new policy will focus on improving public transportation."
        }
    ]

    # --- 3. Process each pair by calling the function ---
    print("--- Processing Pairs ---")
    for i, pair in enumerate(input_data):
        score = get_similarity_score(pair["text1"], pair["text2"])
        print(f"Pair {i+1} Score: {score:.4f}")