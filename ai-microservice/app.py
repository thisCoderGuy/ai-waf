from flask import Flask, request, jsonify
import logging


app = Flask(__name__)


# Configure logging to stdout/stderr which Docker captures
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


@app.route('/classify', methods=['POST'])
def classify():
    data = request.get_json()
    if not data:
        return jsonify({"error": "No JSON data provided"}), 400


    method = data.get('method', 'GET')
    path = data.get('path', '/')
    query = data.get('query', '')
    body = data.get('body', '')


    logging.info(f"Received request for classification: Method={method}, Path={path}, Query={query}, Body={body[:100]}...") # Log first 100 chars of body


    verdict = "benign"
    score = 0 # Default score


    # --- Dummy AI Logic (Replace with your actual model) ---
    # Example: Simple keyword-based detection for demonstration
    if "union select" in body.lower() or "union select" in query.lower():
        verdict = "malicious"
        score = 0.95
        logging.warning("Dummy AI: SQLi keyword detected!")
    elif "<script>" in body.lower() or "<script>" in query.lower() or "<script>" in path.lower():
        verdict = "malicious"
        score = 0.85
        logging.warning("Dummy AI: XSS keyword detected!")
    elif "etc/passwd" in path.lower() or "etc/passwd" in query.lower():
        verdict = "malicious"
        score = 0.75
        logging.warning("Dummy AI: Path Traversal keyword detected!")
    else:
        logging.info("Dummy AI: No obvious malicious keywords detected.")


    return jsonify({
        "verdict": verdict,
        "score": score
    })


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
