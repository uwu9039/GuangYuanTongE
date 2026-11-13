---
license: Apache License 2.0
tags:
- campus
- qa
- rag
---

# 校园办事向导 · 中国传媒大学

把中国传媒大学的办事流程整理成可检索、可核对的问答：校园卡、学籍证明、选课考试、转专业、奖助学金、宿舍后勤、校园地图，另有一个单学年体测成绩估算器（`/tice.html`）。

- 条目库 **39,274 条**（verified 39,239 / source-thin 32 / partial 3），事实字段 **42,414 个**全部带逐字引文，`check_quotes` 失败 0
- 章节索引 **531 份 / 11,886 节 / 22,117 条落节**（章节点加分 ② 默认开，云端可用）
- 每条事实带原文引文与出处，回答里可直接展开核对
- 表格与阅读报告另存可检索数据表 35333 格、成 8395 条记录（收费公示、开支标准、书目目录、日程等），`/api/search` 与问答都会带出命中的原文行
- 事实只写官方原文，不替学校加码；原文没有的就答「没有找到」
- 检索：自建倒排索引 + 学习排序器 + 规则化拒答（见 [`retrieval/`](retrieval/README.md)），
  763 条人工问法 **hit@5 100%**、库外问题 **100% 拒答**、单次查询 **6 ms**；
  详细的算法、评测与失败分析见 [`RETRIEVAL_REPORT.md`](RETRIEVAL_REPORT.md)
- 体测计算器：七项填空，按《国家学生体质健康标准》算单项分与学年总分（标准分＋附加分），纯前端运行，填写内容只留在本机浏览器，不上传

## 隐私

本站不采集、不存储个人信息，不提供登录代理，也不接受任何写入操作。

校园卡余额、消费流水、课表这类个人数据，`/field` 界面的「个人数据」面板只给两个官方网址入口（校园卡服务大厅、畅课），页面本身不读取、不转发任何数据，查询由本人在官方系统登录后完成。`/desk` 界面另保留一种本地浏览器扩展的方式，需要本人自行安装，数据同样不出浏览器。

公开的办公电话与邮箱来自学校官网，属正常公开信息。

## 运行方式

Docker 类型创空间，容器内 `uvicorn app:app --host 0.0.0.0 --port 7860`。

三个界面共用同一份数据：`/field` 办事向导（手机端与微信内嵌浏览器推荐，`/` 自动跳到这里）、`/desk` 双栏、`/classic` 旧版。

可选环境变量：

| 变量 | 作用 |
|---|---|
| `DEEPSEEK_API_KEY` | 拍照/文字核对与模型裁决；不填也能问答，只是少了这一层 |

## 目录

| 路径 | 内容 |
|---|---|
| `app.py` | FastAPI 服务；`rank()` 已接到新检索器，`_rank_legacy()` 是索引缺失时的兜底 |
| `retrieval/` | 检索算法（索引 / 特征 / 排序器 / 拒答 / 答案装配），见其 README |
| `_eval/` | 评测集与评测脚本（hit@k、拒答率、答案完整率、阈值扫描） |
| `items.json.gz` | 条目库（压缩后约 6.1 MB，39222 条） |
| `table-rows.json.gz` | 可检索数据表 35333 格（收费、书目、日程等表格内容） |
| `doc-report.json.gz` | 每份正文文件的条目数与覆盖率台账，离线建库要读 |
| `file-registry.json.gz` | `sources/` 下每个文件的处置结论台账，离线建库要读 |
| `sources-archive.tar.gz` | 原文正文归档（网站不读，供逐字核对与离线建库） |
| `presets.json` | 预置问法 764 条 |
| `web-next/` | 前端（无构建步骤） |
| `sources/` | 全部原文正文与来源索引、校园 POI、向量索引、16 份转专业计划 PDF |
| `tools/` | 抽取、建库、逐字校验脚本（云端要重建检索库时才用） |

## 检索索引

问答检索走 `retrieval/`：倒排索引（`G:` 字符二元组 + `P:` 领域词，5 个字段 BM25F）→
47 维特征 → 线性排序器（用条目库自带的 3.6 万条别名当训练数据）→ 拒答层 → 答案装配。
另有可选的 **dense 通道**（百炼 `qwen3.7-text-embedding`），把查询编码后与 26 647 条
条目向量比余弦，能把「说法不一样」的问题捞回来（手写改写集 hit@5 87% → 98.7%）。

索引是派生数据（倒排 88 MB），**不进仓库**，构建时现建：

```bash
python -m retrieval.build                                   # 倒排索引，约 12 秒
python -m tools.train_ranker --rounds 1 --max-queries 12000 # 排序器，约 1.5 分钟
# 可选 dense（需要 .env 里的 DASHSCOPE_API_KEY）
python -m tools.build_dense --workers 1                     # 约 16 分钟
python -m tools.train_dense_ranker --max-queries 3000       # 约 2 分钟
```

镜像里 `Dockerfile` 只跑倒排索引那一步；排序器权重（几 KB）跟着仓库走。

**dense 通道的条目向量**（`retrieval/artifacts/dense/`，约 105 MB）是唯一跟着仓库走的
派生件：它靠调 embedding 花钱建（约 6M token），云空间里没法自己跑命令，所以在创空间
那份仓库里随包发布（Git 仓库把 `.gitignore` 里的 `retrieval/artifacts/dense/` 开了例外）。
配好 `DASHSCOPE_API_KEY` 即自动启用；想显式关掉就设 `RETRIEVAL_DENSE=0`。
换一份不带这个目录的仓库重建时，`Dockerfile` 会在构建期有 Key 的情况下现建，没有就只提示。
其余索引建不出来时 `app.py` 自动退回旧的规则打分，站点不会挂。评测与复现命令见
[`RETRIEVAL_REPORT.md`](RETRIEVAL_REPORT.md)。

## 全文检索库

`kb.sqlite` 不随仓库发（398 MB），镜像构建时由 `Dockerfile` 现建；本地也能手建（四个 gz 齐了就行）：

```bash
python tools/build_kb_db.py           # 约 21 秒，建出 398 MB
python tools/build_kb_db.py --verify  # 与条目库、台账、sources/ 逐字段对上
```

建出来的表：doc 2184 / item 39222 / quote 42275 / table_row 35333 / registry 3737，另有 coverage 视图。`--verify` 会核对逐字引文、位图缺口与重复认领，并跑 8 个二字探针（中文二元 gram，FTS5 自带的 trigram 对 2 字查询恒 0 命中，所以自建了 gram 列）。

问答路由已由 `retrieval/` 接管；`kb.sqlite` 现在只在 `/api/search` 原文检索与模型复核的
补充腿里用。库缺失时这些腿自动跳过，问答不受影响。

## 来源

学校官网、学生工作部、教务处、信息化处、图书馆、学信网、教育部等公开页面，以及「问道中传」知识库中可公开引用的原文。条目内的引文均可在对应原文中逐字核对。
