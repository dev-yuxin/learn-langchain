from langchain_community.document_loaders import UnstructuredMarkdownLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter


def load_md(file_path, mode="single"):
    loader = UnstructuredMarkdownLoader(
        file_path=file_path,
        mode=mode
    )
    docs = loader.load()
    return docs


def split_docs(docs, chunk_size=512, chunk_overlap=64):
    # 创建递归字符切分器
    splitter = RecursiveCharacterTextSplitter(
        # 每个切片最大字符数
        chunk_size=chunk_size,
        # 每个切片重叠字符数
        chunk_overlap=chunk_overlap,
        # 先按一级标题切分，如果长度符合，则结束
        # 长度超出，再按\n\n切分，如果长度符合，则结束，
        # 长度溢出，继续按下一个切
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


if __name__ == '__main__':
    file_path = r"docs/中国科学院国家天文台2023年部门预算/full.md"
    docs = load_md(file_path)
    splitted_docs = split_docs(docs)

