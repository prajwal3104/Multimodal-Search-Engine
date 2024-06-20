from flask import Flask, request, jsonify
from src.models.model_inference import generate_image_embeddings, find_matches
from src.utils.config import Config
from src.utils.logger import get_logger

app = Flask(__name__)
logger = get_logger(__name__)

config = Config()

# Load image embeddings
image_embeddings = generate_image_embeddings(
    config.image_paths, config.model_dir, config.batch_size
)

@app.route("/search", methods=["POST"])
def search():
    """API endpoint for image search."""

    try:
        data = request.get_json()
        queries = data.get("queries")
        if not queries:
            return jsonify({"error": "Missing queries"}), 400

        results = find_matches(image_embeddings, queries, k=config.k)

        return jsonify({"results": results}), 200

    except Exception as e:
        logger.error(f"Error during search: {e}")
        return jsonify({"error": "Internal server error"}), 500

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000) 