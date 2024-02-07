from pymilvus import connections, Collection, utility, FieldSchema, CollectionSchema, DataType
import config

def recreate_collection(collection_name):
    # Connect to Milvus
    connections.connect("default", host=config.Config.MILVUS_HOST, port=config.Config.MILVUS_PORT)
    print("Connected to Milvus successfully.")

    # Check if the collection exists
    if utility.has_collection(collection_name):
        # Drop the collection
        collection = Collection(name=collection_name)
        collection.drop()
        print(f"Collection '{collection_name}' has been dropped.")

    # Recreate the collection with the same schema
    fields = [
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
        FieldSchema(name="text_vector", dtype=DataType.FLOAT_VECTOR, dim=384)  # Dimension based on the model
    ]
    schema = CollectionSchema(fields, description="Text Collection")
    Collection(name=collection_name, schema=schema)
    print(f"Collection '{collection_name}' has been recreated.")

    # Disconnect from Milvus
    connections.disconnect(alias="default")

if __name__ == "__main__":
    recreate_collection(config.Config.MILVUS_COLLECTION_NAME)
