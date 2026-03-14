from llama_index.llms.ollama import Ollama
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings, StorageContext, load_index_from_storage
import time

Settings.embed_model = OllamaEmbedding(model_name="nomic-embed-text")

llm_mistral = Ollama(model="mistral", request_timeout=180.0)
llm_qwen = Ollama(model="qwen2.5:1.5b", request_timeout=180.0)


print("正在从硬盘加载索引...")
storage_context = StorageContext.from_defaults(persist_dir=r"D:\Desktop\courses\6493\project\storage")
index = load_index_from_storage(storage_context)


test_question = "帮我总结老师词嵌入这一章节主要讲的内容"

print(f"\n测试问题: {test_question}\n")
print("=" * 60)


print("正在使用 Mistral-7B 结合 PDF 生成回答...")
start_time = time.time()

query_engine_mistral = index.as_query_engine(llm=llm_mistral)
response_mistral = query_engine_mistral.query(test_question)

end_time = time.time()
print(f"[Mistral 回答] (耗时: {end_time - start_time:.2f} 秒):")
print(response_mistral.response)
print("-" * 60)

print("正在使用 Qwen2.5-1.5B 结合 PDF 生成回答...")
start_time = time.time()

query_engine_qwen = index.as_query_engine(llm=llm_qwen)
response_qwen = query_engine_qwen.query(test_question)

end_time = time.time()
print(f"[Qwen 回答] (耗时: {end_time - start_time:.2f} 秒):")
print(response_qwen.response)
print("=" * 60)

print("正在使用纯 Mistral-7B 生成回答 (无 RAG)...")
start_time = time.time()
pure_response_mistral = llm_mistral.complete(test_question)

end_time = time.time()
print(f"[纯 Mistral 回答] (耗时: {end_time - start_time:.2f} 秒):")
print(pure_response_mistral.text)
print("=" * 60)

print("正在使用纯 Qwen2.5-1.5B 生成回答 (无 RAG)...")
start_time = time.time()
pure_response_qwen = llm_qwen.complete(test_question)

end_time = time.time()
print(f"[纯 Qwen2.5-1.5B 回答] (耗时: {end_time - start_time:.2f} 秒):")
print(pure_response_qwen.text)
print("=" * 60)