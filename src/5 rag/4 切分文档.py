from langchain_core.documents.base import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


# 切分文档
def split_docs(docs: list[Document]):
    # 递归字符切分器
    # 对每个 doc 按分隔符优先级递归切分，直到每块不超过最大长度
    splitter = RecursiveCharacterTextSplitter(
        # 分隔符
        separators=[
            "\r\n\r\n",
            "\n\n",
            "\r\n",
            "\n",
            "。",
            "！",
            "？",
            "；",
            "：",
            "，",
            "、",
            ". ",
            "! ",
            "? ",
            "; ",
            ": ",
            ", ",
            ".",
            "!",
            "?",
            ";",
            ":",
            ",",
            " ",
            "",
        ],
        # 每块最大长度
        chunk_size=1024,
        # 相邻块重叠长度，保持语义连贯
        chunk_overlap=256,
        # 按字符数计算
        length_function=len,
    )
    chunks = splitter.split_documents(docs)
    print(f"切分为{len(chunks)}块")
    return chunks
