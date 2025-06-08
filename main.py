from langchain_community.document_loaders import PyPDFLoader
from langchain_community.chat_models import ChatOpenAI
from langchain_community.llms import SparkLLM
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from openai import OpenAI
import numpy as np
import faiss  # pip install faiss-cpu 或 faiss-gpu

def get_embedding(texts, model="text-embedding-ada-002"):
    client = OpenAI(
        base_url="https://api.starteam.wang/v1",
        # base_url="https://api.chatanywhere.tech/v1",
        api_key="sk-lgO5N1o5LkB8kZ12Fc9071B9B2C0429cBdAe35Ae53351c66"
        # api_key="sk-MW7MscGqMlYiMP0vT0kMEYl2jRouOWjwwUyCXe7NBEeXBke4"
    )

    # 处理单个文本的情况
    if isinstance(texts, str):
        texts = [texts]

    # 过滤非字符串内容
    texts = [str(t) for t in texts if isinstance(t, (str, int, float))]  # 允许数字转换为字符串

    # 分批处理以避免超过 API 限制（假设每次最多处理100个）
    embeddings = []
    batch_size = 100
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        response = client.embeddings.create(
            input=batch,
            model=model
        )
        embeddings += [data.embedding for data in response.data]

    return np.array(embeddings)
###加载文件
loader = PyPDFLoader("data/中国科学技术大学研究生学籍管理实施细则.pdf")
pages = loader.load()

###文本切分
text_splitter = RecursiveCharacterTextSplitter(chunk_size = 300,chunk_overlap = 50,)

docs = text_splitter.split_documents(pages)

# 1. 准备文档块文本列表
texts = [doc.page_content for doc in docs]

# 2. 计算所有块的 embedding
embeddings = get_embedding(texts).astype('float32')  # 强制转为 float32

# 3. 建立 FAISS 索引（这里用内积 + L2 归一化当作余弦相似度）
dim = embeddings.shape[1]
faiss.normalize_L2(embeddings)
index = faiss.IndexFlatIP(dim)
index.add(embeddings)

# …

def retrieve_top_k(query, k=5):
    # 4.1 得到查询向量并转换类型、归一化
    q_emb = get_embedding(query).astype('float32')
    faiss.normalize_L2(q_emb)
    # 4.2 在索引里搜索
    distances, indices = index.search(q_emb, k)
    # 4.3 返回结果
    return [
        {"text": texts[idx], "score": float(score)}
        for score, idx in zip(distances[0], indices[0])
    ]



llm = ChatOpenAI(
            base_url="https://api.starteam.wang/v1",
            api_key="sk-pIwBotFHGSfIZbCL2f0b064c653b4f46Ac36261fF0653594",
            model="gpt-4o"
        )
augmented_prompt = """Using the contexts below, answer the query.

contexts:
{source_knowledge}

query: {query}"""

prompt = PromptTemplate(template=augmented_prompt, input_variables=["source_knowledge" ,"query"])
llm_chain = LLMChain(prompt=prompt, llm=llm  , llm_kwargs = {"temperature":0, "max_tokens":1024})

query = "研究生超过期限多久未注册，会被退学？"
top_k = retrieve_top_k(query, k=3)
for i, item in enumerate(top_k, 1):
    print(f"Rank {i}, score={item['score']:.4f}\n{item['text']}\n")

# 6. 把 top_k 拼成 prompt，调用 LLMChain
source_knowledge = "\n---\n".join([item["text"] for item in top_k])
response = llm_chain.invoke({
    "source_knowledge": source_knowledge,
    "query": query
})
print("LLM 回答：", response)
