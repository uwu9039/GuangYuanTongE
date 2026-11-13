"""检索算法（重写版）。

对外 API::

    from retrieval import Engine
    eng = Engine.load()                      # 读 retrieval/artifacts/
    res = eng.search("饭卡丢了怎么办", k=5)
    res.hits; res.refused; res.best.item; res.answer

模块分工：

    text.py     规范化 / 问句拆套话 / 词表最长匹配 / 字符二元组
    index.py    倒排索引 + BM25F（G: 二元组通道 + P: 领域词通道）
    features.py 候选对特征（字面覆盖、通道分、条目种类先验……）
    ranker.py   numpy 逻辑回归排序器（离线训练，权重 JSON）
    gate.py     拒答判定（校准过的概率 + 覆盖度兜底）
    compose.py  把命中的条目装配成「全面」的答案（步骤 / 材料 / 费用 / 地点 / … / 出处）
    engine.py   把上面几层串起来
"""

__all__ = ["Engine", "Result", "Hit", "Answer"]


def __getattr__(name: str):  # 延迟导入：只建索引时不必装载 engine
    if name in __all__:
        from . import engine
        return getattr(engine, name)
    raise AttributeError(name)
