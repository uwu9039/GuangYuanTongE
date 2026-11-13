# 魔搭创空间（Docker 类型）：平台把 7860 端口作为对外入口。
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    TZ=Asia/Shanghai

WORKDIR /app

COPY requirements.txt .
RUN pip install --retries 3 -i https://mirrors.aliyun.com/pypi/simple/ \
        --trusted-host mirrors.aliyun.com -r requirements.txt

COPY . .

# 向量库按字节分片（每片 65 MB），避开平台 ~100MB 单文件校验；构建时拼回原文件。
# 拼完校验字节数，不对就报警（缺片/半截片会造成向量库加载失败）。
# 分片不在仓库里时**不打断构建**：缺它只是「原文语义检索」这条腿空着，问答照常。
RUN if [ -f sources/vector/vectors.npy.part1 ]; then \
        cat sources/vector/vectors.npy.part* > sources/vector/vectors.npy \
        && rm -f sources/vector/vectors.npy.part* \
        && ls -l sources/vector/vectors.npy \
        && python -c "import numpy as np; v=np.load('sources/vector/vectors.npy', mmap_mode='r'); print('vectors', v.shape, v.dtype)"; \
    else \
        echo "!! 提示：找不到 sources/vector/vectors.npy.part*，原文语义检索这条腿关着（问答不受影响）"; \
    fi

# 部署时还原正文树（幂等，只补缺）。线上问答不读它，还原出来只为容器里能跑
# tools/build_kb_db.py 与 tools/check_quotes.py。
# 不让构建失败：这批正文对运行零收益，为它把站点顶下线不值当。失败时顺手把归档的大小与
# 头 16 字节打进日志——是半截 gzip 还是被换成了 LFS 指针，一眼就能分清。
RUN python tools/extract_sources.py || (echo "!! 警告：正文归档不可用，云端重建检索库前请重传 sources-archive.tar.gz"; ls -l sources-archive.tar.gz; head -c 16 sources-archive.tar.gz | od -An -c)

# 检索库是派生品，仓库里不带（398 MB）。镜像构建时现建：没有它，四路融合里
# BM25 与名目两路是空的，只剩规则＋向量。约 20 秒、+398 MB 镜像。
# 不因它把站点顶下线：建失败就照旧用规则＋向量答。
RUN python tools/build_kb_db.py || echo "!! 警告：kb.sqlite 构建失败，线上只用规则＋向量"

# 检索索引是派生品（约 90 MB），仓库里不带：构建时用条目库现建，约 50 秒。
# 建不出来也不把站点顶下线——app.py 会退回旧的规则打分（慢、但能答）。
RUN python -m retrieval.build || echo "!! 警告：检索索引构建失败，线上退回旧规则打分"

# dense 通道（百炼 embedding 语义重排）的条目侧向量。
# 正常情况下 `retrieval/artifacts/dense/` 随仓库走（创空间那份带，约 105 MB），这步直接跳过；
# 换一份不带该目录的仓库重建时，如果构建期能拿到 DASHSCOPE_API_KEY 就现建（约 5–8 分钟、
# 约 3 元），拿不到就只提示——运行时 Key 仍然可以只用来做查询侧 embedding，不会报错。
RUN if [ -f retrieval/artifacts/dense/manifest.json ]; then \
        echo "dense 条目向量随仓库带，跳过构建"; \
        ls -l retrieval/artifacts/dense/; \
    elif [ -n "$DASHSCOPE_API_KEY" ]; then \
        python -m tools.build_dense || echo "!! 警告：dense 条目向量构建失败，dense 通道关着"; \
    else \
        echo "!! 提示：没有随包的 dense 条目向量，构建期也没有 DASHSCOPE_API_KEY，dense 通道关着（字面检索不受影响）"; \
    fi

EXPOSE 7860

CMD ["python", "-m", "uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860", "--workers", "1"]
