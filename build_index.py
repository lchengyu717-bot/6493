from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings, StorageContext
from llama_index.embeddings.ollama import OllamaEmbedding

Settings.embed_model = OllamaEmbedding(model_name="nomic-embed-text")
Settings.chunk_size = 256
Settings.chunk_overlap = 25

print("正在读取 PDF 并建立索引...")
documents = SimpleDirectoryReader("D:/Desktop/courses/6493/lecture").load_data()
index = VectorStoreIndex.from_documents(documents)


save_path = r"D:\Desktop\courses\6493\project\storage"
index.storage_context.persist(persist_dir=save_path)
print("索引已成功保存到 ./storage 文件夹！")