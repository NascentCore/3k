from flask import Flask, request, jsonify
import requests
from transformers import AutoTokenizer, AutoModel
import json
from pymilvus import connections, Collection

import config

app = Flask(__name__)
app.config.from_object(config.Config)

# Load model and tokenizer for embeddings
tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")

# Load ID to Text Mapping
with open(config.Config.ID_TEXT_FILE_NAME, 'r') as f:
    id_text_map = json.load(f)

# Connect to Milvus
connections.connect("default", host=app.config['MILVUS_HOST'], port=app.config['MILVUS_PORT'])

collection = Collection(name=app.config['MILVUS_COLLECTION_NAME'])

@app.route("/complete", methods=["POST"])
def complete():
    data = request.json
    message = data.get("message", "")
    rag = data.get("rag", False)  # Get the 'rag' parameter, default to False if not provided

    if rag:
        # Embed the message to create a query vector
        inputs = tokenizer(message, return_tensors="pt", truncation=True, max_length=512)
        outputs = model(**inputs)
        query_vector = outputs.last_hidden_state.mean(dim=1).detach().numpy().flatten().tolist()

        # Search in Milvus for similar embeddings
        search_params = {"metric_type": "L2", "params": {"nprobe": 10}}
        top_k = 3  # Number of similar texts to retrieve
        results = collection.search([query_vector], "text_vector", search_params, top_k)

        # Construct a new prompt from the search results and the original message
        similar_texts = [id_text_map.get(str(hit.id), "Related text not found") for hit in results[0]]
        prompt = f"Based on: {'; '.join(similar_texts)}.\nMy question is: {message}"
    else:
        # If 'rag' is False, use the original message as the prompt
        prompt = message

    # Construct the payload for OpenChat
    openchat_payload = {
        "model": "openchat_3.5",
        "messages": [{"role": "user", "content": prompt}]
    }

    # Invoke the OpenChat API with the constructed payload
    response = requests.post(app.config['OPENCHAT_URL'], json=openchat_payload)

    if response.status_code == 200:
        resp_data = response.json()
        return jsonify(resp_data)  # Return the OpenChat response
    else:
        return jsonify({"error": "Failed to get response from OpenChat", "status_code": response.status_code})

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=5000)
