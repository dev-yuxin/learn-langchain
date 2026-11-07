```bash
# 自己根据 cuda 版本修改 pytorch 版本
# 手动在 `2 RAG/models` 目录（没有自己创建）下载模型
curl.exe -O https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.8.0/en_core_web_sm-3.8.0-py3-none-any.whl
# git 需安装 lfs
git clone https://www.modelscope.cn/BAAI/bge-m3.git 
uv sync
```

