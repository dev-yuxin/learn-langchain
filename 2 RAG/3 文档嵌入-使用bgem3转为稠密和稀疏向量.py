import numpy as np
import torch
from FlagEmbedding import BGEM3FlagModel
from FlagEmbedding.inference.embedder.encoder_only.m3 import M3Embedder
from langchain_community.document_loaders import UnstructuredMarkdownLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document


def load_md(file_path, mode="single"):
    loader = UnstructuredMarkdownLoader(
        file_path=file_path,
        mode=mode
    )
    docs = loader.load()
    return docs


def split_docs(docs, chunk_size=512, chunk_overlap=64):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=[
            "\n# ",  # 一级标题（注意前面加 \n，避免匹配到行内 #）
            "\n\n",  # 段落（中英文通用）
            "\n",  # 行
            "。",  # 中文句号
            "！",  # 中文感叹号
            "？",  # 中文问号
            "；",  # 中文分号
            ". ",  # 英文句号 + 空格
            "! ",  # 英文感叹号 + 空格
            "? ",  # 英文问号 + 空格
            "; ",  # 英文分号 + 空格
            "，",  # 中文逗号
            ", ",  # 英文逗号 + 空格
            " ",  # 空格
            "",  # 字符兜底
        ],
    )
    splitted_docs = splitter.split_documents(docs)
    return splitted_docs


def load_embedding_model():
    """使用bge-m3作为嵌入模型"""
    model = BGEM3FlagModel(
        model_name_or_path="models/bge-m3",
    )
    if torch.cuda.is_available():
        model.model = model.model.to("cuda:0")
    print(f"加载bge-m3模型成功，运行设备{next(model.model.parameters()).device}")
    return model


def docs_to_vectors(model: M3Embedder, docs: list[Document], batch_size=32):
    result = model.encode(
        sentences=[doc.page_content for doc in docs],
        return_dense=True,
        return_sparse=True,
        batch_size=batch_size
    )
    dense_vectors = result["dense_vecs"]
    sparse_vectors = result["lexical_weights"]
    print("向量化完成")
    return [{
        # 稠密向量数据类型必须是 float32
        "dense_vector": np.asarray(dense_vector, dtype=np.float32),
        # 稀疏向量键转为整形
        "sparse_vector": {int(k): v for k, v in sparse_vector.items()},
        "metadata": doc.metadata,
        "text": doc.page_content
    } for dense_vector, sparse_vector, doc in zip(dense_vectors, sparse_vectors, docs)]


if __name__ == '__main__':
    file_path = r"docs/中国科学院国家天文台2023年部门预算/full.md"
    docs = load_md(file_path)
    chunks = split_docs(docs)
    model = load_embedding_model()
    result = docs_to_vectors(model, chunks)
    print(result[0]["dense_vector"])
    print(result[0]["sparse_vector"])
    print(result[0]["text"])
    print(result[0]["metadata"])
