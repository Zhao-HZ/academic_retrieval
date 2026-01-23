from pymilvus import MilvusClient

client = MilvusClient(
    uri="http://localhost:19530",
    token="root:Milvus"
)

client.use_database('academic_retrieval_db')

client.drop_collection(collection_name=client.list_collections()[0])
print(client.list_collections())

# query = "graph neural-network"
# results = client.search(
#     collection_name='academic_retrieval_collection',
#     data=[query],
#     anns_field="sparse_vector",
#     limit=5,
#     output_fields=['content', 'metadata']
# )

# sparse_results = results[0]
# for i, result in enumerate(sparse_results):
#     print(f"{i+1}. Score: {result['distance']:.4f}, Content: {result['entity']['content']}")
