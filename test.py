from pymilvus import MilvusClient

client = MilvusClient(
    uri="http://localhost:19530",
    token="root:Milvus"
)

# client.use_database('academic_retrieval_db')
# print(client.list_collections())

client.drop_collection(collection_name='academic_retrieval_db')

# res = client.drop_collection(
#     collection_name="review"
# )

# res = client.drop_database(
#     db_name='academic_retrieval_db'
# )
