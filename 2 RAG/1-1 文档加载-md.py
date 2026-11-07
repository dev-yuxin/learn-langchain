from langchain_community.document_loaders import UnstructuredMarkdownLoader


def load_md(file_path, mode="single"):
    loader = UnstructuredMarkdownLoader(
        file_path=file_path,
        # mode:
        # single 切分成一个Document对象，一般用single，然后自定义切分规则
        # elements 按元素切分为多个Document对象
        mode=mode
    )
    # load()/aload() 一次性读取
    # lazy_load()/alazy_load() 懒加载
    docs = loader.load()
    return docs


if __name__ == '__main__':
    docs = load_md("docs/投机解码.md")
    print(len(docs))
    print(docs[0].page_content)
    print(docs[0].metadata)
