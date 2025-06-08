import pandas as pd
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from openai import OpenAI
import numpy as np
import faiss  # pip install faiss-cpu 或 faiss-gpu

def get_embedding(texts, model="text-embedding-ada-002"):
    client = OpenAI(
        base_url="https://api.starteam.wang/v1",
        api_key="sk-lgO5N1o5LkB8kZ12Fc9071B9B2C0429cBdAe35Ae53351c66"
    )
    if isinstance(texts, str):
        texts = [texts]
    texts = [str(t) for t in texts if isinstance(t, (str, int, float))]
    embeddings = []
    batch_size = 100
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        resp = client.embeddings.create(input=batch, model=model)
        embeddings += [d.embedding for d in resp.data]
    return np.array(embeddings)

# 1. 加载并切分 PDF 文档
loader = PyPDFLoader("data/中国科学技术大学研究生学籍管理实施细则.pdf")
pages = loader.load()
splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=50)
docs = splitter.split_documents(pages)
texts = [d.page_content for d in docs]

# 2. 建立向量索引
embs = get_embedding(texts).astype('float32')
faiss.normalize_L2(embs)
dim = embs.shape[1]
index = faiss.IndexFlatIP(dim)
index.add(embs)

def retrieve_top_k(query, k=5):
    q_emb = get_embedding(query).astype('float32')
    faiss.normalize_L2(q_emb)
    D, I = index.search(q_emb, k)
    return [texts[i] for i in I[0]]

# 3. 准备 QA 验证模型
llm = ChatOpenAI(
    base_url="https://api.starteam.wang/v1",
    api_key="sk-pIwBotFHGSfIZbCL2f0b064c653b4f46Ac36261fF0653594",
    model="gpt-4o"
)
# 用于检索答案的 RAG Chain
answer_prompt = PromptTemplate(
    template="""Using the contexts below, answer the query.

contexts:
{source_knowledge}

query: {query}""",
    input_variables=["source_knowledge", "query"]
)
answer_chain = LLMChain(prompt=answer_prompt, llm=llm,
                        llm_kwargs={"temperature":0, "max_tokens":1024})

# 用于打分（0/1）的验证 Chain
verify_prompt = PromptTemplate(
    template="""请判断下面的模型回答是否与正确答案一致，仅返回“0”或“1”：
问题：{question}
模型回答：{model_answer}
正确答案：{gold_answer}""",
    input_variables=["question", "model_answer", "gold_answer"]
)
verify_chain = LLMChain(prompt=verify_prompt, llm=llm,
                        llm_kwargs={"temperature":0, "max_tokens":4})

# 4. 读取问答对 Excel，并依次执行
df = pd.read_excel("data/问答对.xlsx")
scores = []
model_answers = []

for idx, row in df.iterrows():
    q = str(row.iloc[0])
    gold = str(row.iloc[1])
    # 检索上下文、生成模型答案
    ctxs = retrieve_top_k(q, k=5)
    source_knowledge = "\n---\n".join(ctxs)
    out = answer_chain.invoke({"source_knowledge": source_knowledge, "query": q})
    model_ans = out.get("text", "").strip()
    model_answers.append(model_ans)
    # 验证打分
    v = verify_chain.invoke({
        "question": q,
        "model_answer": model_ans,
        "gold_answer": gold
    })
    score = v.get("text", "0").strip()
    scores.append(score)

# 5. 保存结果到 CSV
df["model_answer"] = model_answers
df["score"] = scores
df.to_csv("data/问答对_评估结果.csv", index=False, encoding="utf-8-sig")
print("Evaluation complete. Output saved to data/问答对_评估结果.csv")