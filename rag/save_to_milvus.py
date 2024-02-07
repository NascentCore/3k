import argparse
import json
from transformers import AutoTokenizer, AutoModel
from pymilvus import connections, FieldSchema, CollectionSchema, DataType, Collection
import config

# Set up argument parser
parser = argparse.ArgumentParser(description="Load texts from a JSON file and process them.")
parser.add_argument("json_file", help="Path to the JSON file containing texts.")
parser.add_argument("output_file", help="Path to the output file where the ID to text mapping will be saved.")
args = parser.parse_args()

print("Reading texts from JSON file...")
# Read texts from the JSON file
with open(args.json_file, 'r') as f:
    texts = json.load(f)
print(f"Loaded {len(texts)} texts from the file.")

# Load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
print("Model and tokenizer loaded successfully.")

# Convert texts to vectors
vectors = []
for i, text in enumerate(texts, start=1):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    outputs = model(**inputs)
    embeddings = outputs.last_hidden_state.mean(dim=1).detach().numpy()
    vectors.append(embeddings[0])
print("All texts have been converted to vectors.")

# Connect to Milvus
connections.connect("default", host=config.Config.MILVUS_HOST, port=config.Config.MILVUS_PORT)
print("Connected to Milvus successfully.")

# Define fields for the collection
fields = [
    FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
    FieldSchema(name="text_vector", dtype=DataType.FLOAT_VECTOR, dim=384)  # Dimension based on the model
]

# Create a collection schema
schema = CollectionSchema(fields, description="Text Collection")

# Create a collection in Milvus
collection_name = config.Config.MILVUS_COLLECTION_NAME
collection = Collection(name=collection_name, schema=schema)
print(f"Collection '{collection_name}' created in Milvus.")

# Insert vectors into the collection
mr = collection.insert([vectors])
ids = mr.primary_keys
print("Vectors inserted successfully. IDs obtained.")

# Create a mapping of IDs to texts
id_text_map = {str(id): text for id, text in zip(ids, texts)}

# Save the mapping to the specified output file
with open(args.output_file, 'w') as f:
    json.dump(id_text_map, f, indent=4)
print(f"ID to text mapping saved to '{args.output_file}'.")

index_params = {"index_type": "IVF_FLAT", "metric_type": "L2", "params": {"nlist": 128}}
collection.create_index(field_name="text_vector", index_params=index_params)

print("Process completed.")
