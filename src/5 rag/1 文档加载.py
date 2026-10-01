from langchain_unstructured.document_loaders import UnstructuredLoader


# 加载文档
def load_doc(file_path: str, mode="single"):
    try:
        loader = UnstructuredLoader(
            file_path=file_path,
            # mode 取值 "single" / "elements"
            # single: 返回列表里只有一整个文档。
            # elements: markdown 标题、段落、列表等都视为元素。word 一行视为一个元素。返回多个文档组成的列表。
            mode=mode,
        )
        docs = loader.load()
        print(f"加载{file_path}成功")
        return docs
    except Exception as e:  # noqa: BLE001
        print(e)
