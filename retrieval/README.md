# 检索算法（重写版）

把「校园办事向导」的检索从 900 行手写 if/else + 字符哈希向量，换成
**倒排索引 + 学习排序 + 规则化拒答**，外加一条可选的 **dense（云 embedding）通道**。
除 dense 通道外仍然只有 `numpy` 一个依赖。

```
问句 ─► 规范化/拆套话 ─► 双通道 BM25F 召回 ─┬─► 56 维特征 ─► 线性排序器 ─► 排序后规则 ─► 拒答层 ─► 答案装配
        text.py          index.py          │   features.py   ranker.py    engine.py    gate.py   compose.py
                                            └─► dense 召回 ────┘
                                            dense.py（百炼 qwen3.7-text-embedding，可选）
```

「排序后规则」是两档确定性调整（`engine._adjust`）：**标题落位**（冠军标题一个实词都不含
时按实词覆盖重排）、**hub 让位**（汇总/常见问题类碎片让位给具体条目）、**章条锚定**
（问句点了「第X条」时按 `sections.py` 把那一节抬上来）。它们只改相对次序与加分，
不改候选集、不改答案形态。

## 为什么这么拆

| 原实现 | 新实现 |
|---|---|
| `score_item()` 每条问句对 3.9 万条逐条跑正则，`blocked_keys()` 再全库扫一遍 | 预建倒排索引，一次查询只碰命中的 postings |
| 阈值 `MATCH_THRESHOLD=55` 人工拍死，散落十几处 `min(sc, 54)` 补丁 | 排序器输出概率 + 拒答层按「个人数据 / 别的学校 / 越界话题 / 语料覆盖 / 太笼统」分档判定 |
| 「向量」是字符 1–3 gram 哈希投影，只做字面兜底 | 领域词（P:）+ 字符二元组（G:）两路词条，5 个字段分别打 BM25F；要语义就开 dense |
| 4.9 秒 / 查询 | 字面 4.5 ms；开 dense 约 250–300 ms（云 embedding 一次） |

## 文件

| 文件 | 作用 |
|---|---|
| `text.py` | 规范化、拆问句套话/情态词、词表最长匹配、字符二元组、标题主干、主题串（剥框架词）、实词、hub 判定 |
| `index.py` | 倒排索引（`.npy` 目录）、BM25F、覆盖度累加、精确串表 |
| `features.py` | 候选对 56 维特征（两通道 × 5 字段、字面覆盖、种类先验、交互项、同名条目、主题/实词覆盖、hub、dense 分） |
| `ranker.py` | numpy 逻辑回归（牛顿法），权重存 JSON |
| `sections.py` | 章条锚定：问句点了「第X条/章」时，按「这一节讲什么」给该节条目加分 |
| `gate.py` | 拒答：个人数据 / 别的学校 / 越界话题 / 校情统计 / 问面—字段核对 / 语料覆盖 / 太笼统 / 无足够接近条目 |
| `compose.py` | 把命中条目装配成条目化答复（步骤/材料/费用/地点/时间/时限/注意事项/联系方式/出处） |
| `engine.py` | 串起来：`Engine.load(dense=False)` → `Engine.search()`；含排序后规则 `_adjust()` |
| `dense.py` | 百炼 embedding 客户端 + 向量缓存 + 向量检索（可选通道） |
| `build.py` | 建倒排索引与词表：`python -m retrieval.build` |

## 快速开始

```bash
python -m retrieval.build          # 建倒排索引（约 12 秒，产物 88 MB）
python -m tools.train_ranker       # 字面排序器（别名 1.2 万条，1～2 轮，约 1.5 分钟/轮）

# 可选：dense 通道（需要 .env 里的 DASHSCOPE_API_KEY）
python -m tools.build_dense --workers 1          # 编码 26 647 条可答条目，约 16 分钟
python -m tools.train_dense_ranker --max-queries 3000   # 带 dense 特征的排序器，约 2 分钟

python - <<'PY'
from retrieval import Engine
eng = Engine.load(dense=True)                    # 不传 dense 就是纯字面
res = eng.search("饭卡丢了怎么办", k=5, with_answer=True)
print(res.hits[0].title, res.hits[0].prob)       # 校园卡补卡 0.95
print(res.answer.text)                           # 完整答复
print(eng.search("清华校园卡怎么补办").refused)    # True
PY
```

## 两条通道的关系

* **字面通道**永远在，4–6 ms，不依赖网络；dense 挂了 `Engine.search` 会退回字面权重
  （两套权重共用同一套特征位，dense 关闭时那两维恒为 0），不会混合出没验证过的分数。
* **dense 通道**把查询编码一次（约 250–300 ms），在 26 647 条可答条目里取 top-120，
  并入候选池；排序器用 `dense_score` / `dense_rank_recip` 两维决定它值多少。
* 入口开关：`Engine.load(dense=True)`，或环境变量 `RETRIEVAL_DENSE=1`（`app.py` 读它）。

## 评价

```bash
python -m _eval.build_sets                 # 生成自建评测集
python -m _eval.run_eval --engine new      # 新检索器（presets / paraphrase / negatives）
python -m _eval.eval_answers               # 答案字段完整率
python -m _eval.bank_eval --dense          # 题库_20261113（433 题）带 dense 那一跑
python -m _eval.bank_compare               # 两跑对照
python -m _eval.refusal_check --dense      # 拒答回归
```

## 设计约束（改之前先读）

1. **除 dense 外不引入外部依赖**。目标机是 ModelScope 创空间的 Docker，
   `requirements.txt` 只有 fastapi / uvicorn / numpy。dense 是可选件：没有
   `DASHSCOPE_API_KEY` 就只跑字面通道，服务不受影响。
2. **训练集不碰评测集**。`tools/train_ranker.py` 会把 presets / 手写改写 / 题库里的问题
   从训练数据里剔掉，避免背题。
3. **同名条目以人工版为规范答案**。库里同一个事项常有 hand / auto / rule / doc 多份，
   训练时把非人工件的正例改指向它的人工孪生件（`hand_twin` 特征负责在推理时识别）。
4. **别名只信人工条目**。自动抽取条目的 `alias` 多半是标题碎片，一放开就会让
   「图书馆」这种泛词把段落条目顶上来。
5. **拒答是规则，不是模型**。一两百个负例训不稳，规则可审计、能说清为什么拒；
   词表（别的学校 / 越界话题 / 校园实词）都在 `gate.py` 里，改词不改逻辑。
