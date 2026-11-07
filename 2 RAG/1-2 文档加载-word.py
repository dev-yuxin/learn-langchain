from langchain_community.document_loaders import UnstructuredWordDocumentLoader


def load_word(file_path, mode="single"):
    loader = UnstructuredWordDocumentLoader(
        file_path=file_path,
        mode=mode
    )
    docs = loader.load()
    return docs


if __name__ == '__main__':
    docs = load_word("docs/中华人民共和国民法典.docx")
    print(len(docs))
    print(docs[0].page_content)
    print(docs[0].metadata)
