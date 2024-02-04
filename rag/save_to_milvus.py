from transformers import AutoTokenizer, AutoModel
from pymilvus import connections, FieldSchema, CollectionSchema, DataType, Collection
import json
import config

# Load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")

# Your texts
texts = [
    '''MANCHESTER, N.H. (AP) — Former President Donald Trump easily won New Hampshire’s primary on Tuesday, seizing command of the race for the Republican nomination and making a November rematch against President Joe Biden feel all the more inevitable.
The result was a setback for former U.N. Ambassador Nikki Haley, who finished second despite investing significant time and financial resources in a state famous for its independent streak. She’s the last major challenger after Florida Gov. Ron DeSantis ended his presidential bid over the weekend, allowing her to campaign as the sole alternative to Trump.
Trump’s allies ramped up pressure on Haley to leave the race before the polls had closed, but Haley vowed after the results were announced to continue her campaign. Speaking to supporters, she intensified her criticism of the former president, questioning his mental acuity and pitching herself as a unifying candidate who would usher in generational change.
''',
    '''MANCHESTER, New Hampshire, Jan 23 (Reuters) - Donald Trump cruised to victory in New Hampshire's Republican presidential contest on Tuesday, marching closer to a November rematch with Democratic President Joe Biden even as his only remaining rival, former U.N. Ambassador Nikki Haley, vowed to soldier on.
"This race is far from over," she told supporters at a post-election party in Concord, challenging Trump to debate her. "I'm a fighter. And I'm scrappy. And now we're the last one standing next to Donald Trump."
''',
    '''MANCHESTER, N.H. — Former President Donald Trump has won the New Hampshire primary, according to The Associated Press, a victory that puts him on a clear path to securing the Republican nomination.
Trump dominated in last week's Iowa caucuses and has now won the first-in-the-nation primary as well.
In New Hampshire, he withstood an aggressive challenge from his former United Nations ambassador, Nikki Haley, who is the final major Republican candidate standing in Trump's way after other hopefuls dropped out.
''',
    '''Donald Trump won New Hampshire's Republican presidential primary on Tuesday, besting his only top-tier rival in the GOP race, Nikki Haley. It comes on the heels of Trump's record-breaking victory in Iowa last week. The former president kept up his calls for Haley to drop out of the race on Tuesday night, but Haley told supporters, “This race is far from over” and vowed to press ahead.
Donald Trump: He appears to be on his way to the GOP nomination, but Tuesday's contest signaled some problems for Trump as a candidate in November's general election.
Nikki Haley: With her path to victory narrowing, the former South Carolina governor maintained that she is the anti-chaos candidate who can beat President Biden in the November election.
Democrats: Biden, who wasn't on Tuesday's primary ballot but still won thanks to a write-in campaign, is now gearing up for a rematch with Trump this fall.
The next big contest: South Carolina's GOP presidential primary on Feb. 24 gets the spotlight next on the election calendar with early state polling showing Trump leading Haley, despite her close ties to the state.
'''
]

# Convert texts to vectors
vectors = []
for text in texts:
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    outputs = model(**inputs)
    embeddings = outputs.last_hidden_state.mean(dim=1).detach().numpy()
    vectors.append(embeddings[0])

# Connect to Milvus
connections.connect("default", host=config.Config.MILVUS_HOST, port=config.Config.MILVUS_PORT)

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

# Insert vectors into the collection
mr = collection.insert([vectors])
ids = mr.primary_keys

# Create a mapping of IDs to texts
id_text_map = {str(id): text for id, text in zip(ids, texts)}

# Save the mapping to a file
map_file = config.Config.ID_TEXT_FILE_NAME
with open(map_file, 'w') as f:
    json.dump(id_text_map, f)

index_params = {"index_type": "IVF_FLAT", "metric_type": "L2", "params": {"nlist": 128}}
collection.create_index(field_name="text_vector", index_params=index_params)
