"""追问上下文 v2：硬独立信号 → 语义距离 → 受约束改写 → 检索仲裁。

这个模块**不 import app**，所有外部能力（条目表、地点表、排序器、embedding、模型）
都由调用方注入，单测可以完全离线跑。判定顺序（每一层都能单独关掉）：

    闸门 0  硬独立信号（纯规则，0 模型调用）
        R0 原句自己点了锚点（「转专业要带什么」）→ 直接算追问，后面的独立性检查全跳过
        R1 自足闲聊（现在几点 / 今天星期几 / 天气）→ 新话题
        R2 句中出现**别的**具体事项/地点/部门（词典命中，df≤HUB_DF，且不是通用词、
           属性词、锚点自己的词）→ 新话题
        R3 句首就是一个具体主体词（图书馆开放时间 / 体测标准是多少）→ 新话题
        R4 不是追问形状（无指代、无属性词、无「呢/吗/么/啥」短句）→ 新话题
    闸门 1  语义距离（1 次 embedding，带缓存）
        sim = cos(embed(新句), embed(锚点条目))
        sim < FAR（0.20）→ 新话题；sim ≥ NEAR（0.55）→ 中间带都不用进
    闸门 2  受约束改写（1 次模型调用，只给中间带；三条硬约束任一不过就丢弃改写）
    仲裁    A=rank(原句) / B=rank(补全句)，按 A 强弱与 B 是否落回锚点决定（0 模型调用）

对外入口 `resolve()` 返回 `Decision`（label / query / reason / 各层证据）。
`CTX_V2=0` 时 app 侧根本不进这里，行为与改动前一致。
"""
from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

ROOT = Path(__file__).resolve().parents[1]

# ---------------------------------------------------------------- 词表

# 通用词 / 问句框架词 / 通用动词：出现在句子里不能证明「另有主体」。
# 这张表是闸门 0 的地基：`_shares_topic` 的旧实现用二字组，「申请」这种通用动词
# 都能把「怎么申请换宿舍」和「转专业」判成沾边——那类词一律不进证据。
GENERIC = {
    "怎么", "什么", "为什么", "如何", "哪里", "哪些", "哪个", "多少", "请问",
    "学校", "学生", "同学", "老师", "本院", "我校", "学院",
    "申请", "办理", "办事", "手续", "流程", "程序", "步骤", "材料", "证件",
    "时间", "规定", "办法", "情况", "相关", "现在", "目前", "是否", "没有",
    "这个", "那个", "一个", "就是", "可以", "需要", "必须", "应该", "能够", "能不",
    "我要", "我想", "能不能", "可以吗", "行不行", "有没有", "有没", "的话",
    "不能", "不会", "不行", "可否", "能否", "是否", "还是", "或者", "以及", "需要",
    "时候", "多久", "几天", "多长时间", "什么时候", "几号", "几点", "哪天",
    "交钱", "费用", "收费", "免费", "钱", "价格", "预约", "排队",
    "周末", "工作日", "节假日", "上班时间", "办公时间", "开放时间",
    "电话", "联系人", "地址", "地点", "位置", "在哪", "哪里办",
    "条件", "资格", "要求", "注意事项", "注意", "事项", "问题", "内容", "信息",
    "办理时间", "受理", "服务", "工作", "部门", "单位", "中心", "办公室",
    "要带", "准备", "准备什么", "提交", "提供", "领取",
    "办好", "办完", "办下来", "弄好", "下来", "能办好", "能办", "办成",
    "谢谢", "麻烦", "帮我", "看一下", "告诉我", "问一下", "咨询",
}

# 指代词开头 = 明确的续问信号
ANAPHORA_HEAD = ("那", "那么", "这", "这个", "这些", "那个", "那些", "它", "该", "还",
                 "也", "然后", "另外", "还有", "再", "上面", "刚才", "你说的", "你刚说")
SUFFIX_PARTICLES = ("呢", "吗", "么", "啥", "呀", "啊")

# 属性词：追问在问「上一件事」的某个字段。hours 单独出现时也能是独立问题
# （「现在几点了」「周末去看电影吗」），必须带办理/开放语境才算数。
ATTR_PATTERNS: dict[str, str] = {
    "material": r"材料|带什么|要带|准备什么|带哪|证件|原件|复印件|什么东西|照片|表格|填表|签字|盖章",
    "duration": r"多久|几天|多长时间|多少天|几个工作日|多久能|要多久|多少时间|什么时候能|截止|期限|有效期",
    "fee": r"要交钱|交钱|收钱|收费|费用|多少钱|免费|贵|押金|花多少",
    "contact": r"找谁|联系谁|谁负责|谁管|谁办|咨询电话|哪个部门|去哪办|在哪办|在哪里办|什么地方办",
    "hours": r"几点|上班|办公时间|周末|节假日|工作日|双休|开门|关门|开放时间|全天",
    "procedure": r"怎么|如何|流程|步骤|怎么办|咋办|怎样才能|怎么弄|线上|线下|网上|代办|代领|加急|找人办|别人办",
    "requirement": r"条件|资格|要求|需要什么|能不能|可以吗|行不行|可不可以|够不够|符合|名额|限制",
    "reservation": r"预约|排队|取号|要不要约",
}
HOURS_CONTEXT = re.compile(r"办|上班|办公|开门|关门|开放|服务|受理|工作|营")
SELF_CONTAINED = re.compile(
    r"几点钟?了|现在几|今天.{0,3}(星期|周几|几号)|明天.{0,3}(星期|周几|几号|天气)|"
    r"昨天.{0,3}(星期|周几)|天气|笑话|你叫什么|你是谁|心情|吃火锅|看电影|写一首|"
    r"^你好|^您好|^在吗|^早上好|^晚安")

# 属性词集合（闸门 0 不把属性词当「另有主体」的证据）
ATTR_WORDS = {
    "材料", "证件", "原件", "复印件", "带什么", "要带", "准备什么",
    "多久", "几天", "多长时间", "多少天", "几个工作日", "什么时候",
    "交钱", "费用", "收费", "免费", "多少钱", "押金",
    "找谁", "联系谁", "谁负责", "咨询电话", "哪个部门", "去哪办",
    "几点", "上班时间", "办公时间", "周末", "节假日", "工作日", "开放时间",
    "流程", "步骤", "条件", "资格", "要求", "预约", "排队", "取号",
    "照片", "表格", "填表", "签字", "盖章", "截止", "截止时间", "期限", "有效期",
    "线上", "线下", "网上", "代办", "代领", "加急", "名额", "限制", "表",
}

# 词典规模上限：df 超过它的词太泛（图书馆 451 / 转专业 325 仍算具体）
HUB_DF = 220
# 闸门 1 端点（S2 在 dev 上标定；dev 追问句对锚点的余弦最小 0.212、无关句最大 0.565，
# 两个类别的 raw 余弦几乎完全重叠，所以 FAR 只做「极远」兜底，不做主判据——见报告）
FAR = 0.20
NEAR = 0.55
# 仲裁强弱阈值（rank() 是 55~99 分制，MATCH_THRESHOLD=55）
A_STRONG = 85
A_WEAK = 55
# 「同族」放宽：原句自己答不动（A 的 top1 不是锚点族）、B 的 top1 与锚点同族时，
# B 分到这条线就判追问。65 在 dev 200 条集标定（误接 0 / 接上 100%），holdout 不动。
FAMILY_MIN = 65


def norm(s: str) -> str:
    return "".join(ch for ch in str(s or "").lower() if ch.isalnum())


def base_title(t: str) -> str:
    """标题去掉括号/斜杠后面那些解释性尾巴：「换宿舍 / 调宿」→「换宿舍」。"""
    for sep in ("（", "(", " / ", "/", "：", ":", "—", "－"):
        t = str(t or "").split(sep)[0]
    return t.strip()


# 「学生如何请假」这类问句式的标题：补全串里带着它，检索反而捞不到这件事本身
_QUESTION_HEAD = re.compile(r"^(?:请问|想问|想咨询|咨询一下)?(?:学生|新生|本科生|研究生|在校生|毕业生|教职工)?"
                            r"(?:要如何|该如何|如何|怎么|怎样|咋)")
_QUESTION_TAIL = re.compile(r"(吗|呢|呀|啊|？|\?|。|！|!)$")
_QUESTION_ACTION = re.compile(r"^(?:申请|办理|操作|进行)(?=[\u4e00-\u9fff]{2,})")


def is_question_title(title: str) -> bool:
    """标题是不是问句式的（「学生如何请假」「转专业难吗」）。

    只有这种标题才需要换成本事名去检索；「校外访问（VPN）」括号里是简称，
    去掉括号反而把检索带偏，不能动。
    """
    t = str(title or "").strip()
    if not t:
        return False
    if _QUESTION_HEAD.match(t) or _QUESTION_TAIL.search(t):
        return True
    return bool(re.search(r"如何|怎么|怎样|咋办|吗[？?]?$|呢[？?]?$", t))


def topic_name(title: str) -> str:
    """标题剥掉「学生/如何/怎么/吗/呢/？/括号」这些问法成分，剩下来的才是主题名。

    「学生如何请假」→「请假」，「怎么申请换宿舍」→「换宿舍」。
    本来就是陈述式标题的不动（「换宿舍 / 调宿」原样返回），长度 <2 时退回原标题。
    """
    t = str(title or "")
    # 括号里的解释性尾巴先去（「请假（病假/事假、审批与销假）」→「请假」）
    t = re.sub(r"（[^）]*）|\([^)]*\)", "", t).strip()
    t2 = _QUESTION_HEAD.sub("", t)
    t2 = _QUESTION_ACTION.sub("", t2)
    t2 = _QUESTION_TAIL.sub("", t2.strip())
    t2 = t2.strip(" ，,、")
    if len(t2) >= 2:
        return t2
    t2 = t.strip()
    return t2 if len(t2) >= 2 else str(title or "")


def item_terms(it: dict[str, Any]) -> set[str]:
    out = {norm(base_title(it.get("title") or ""))}
    out |= {norm(a) for a in (it.get("aliases") or []) if isinstance(a, str)}
    out |= {norm(k) for k in (it.get("keywords") or []) if isinstance(k, str)}
    return {t for t in out if 2 <= len(t) <= 12}


# ---------------------------------------------------------------- 词典


class Lexicon:
    """条目名/别名/关键词 + 地点 + 部门构成的词表，带 df（覆盖多少个条目）。

    df 是最有用的那根轴：`宿舍`(190)、`体测`(20)、`超市`(10) 是具体主体；
    `材料`(960)、`图书馆`(451) 这类泛词靠 HUB_DF 与 GENERIC 一起滤掉。
    """

    __slots__ = ("df", "places", "depts")

    def __init__(self, items: Sequence[dict[str, Any]], places: Iterable[str] = (),
                 depts: Iterable[str] = ()):
        df: dict[str, int] = {}
        for it in items:
            if it.get("archive_only"):
                continue
            for t in item_terms(it):
                df[t] = df.get(t, 0) + 1
        self.df = df
        self.places = {norm(p) for p in places if len(norm(p)) >= 2}
        self.depts = {norm(d) for d in depts if len(norm(d)) >= 2}

    def hits(self, qn: str, maxlen: int = 12) -> dict[str, int]:
        """句子里出现的全部词表词（term → df），地点/部门算 df=1。"""
        out: dict[str, int] = {}
        n = len(qn)
        for i in range(n):
            for j in range(i + 2, min(n, i + maxlen) + 1):
                s = qn[i:j]
                if s in self.df:
                    out[s] = self.df[s]
                elif s in self.places or s in self.depts:
                    out[s] = 1
        return out

    def opens_with(self, qn: str, maxlen: int = 8) -> str:
        """句首的最长词表词（「图书馆开放时间」→「图书馆」）。"""
        best = ""
        for j in range(2, min(len(qn), maxlen) + 1):
            s = qn[:j]
            if s in self.df or s in self.places or s in self.depts:
                best = s
        return best


def load_depts(path: Path | None = None) -> list[str]:
    p = path or (ROOT / "departments.json")
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001  没有这个文件不该挡住主流程
        return []
    out: list[str] = []
    if isinstance(data, dict):
        for v in data.values():
            if isinstance(v, str):
                out.append(v)
            elif isinstance(v, dict):
                out.extend(str(x) for x in (v.get("name"), v.get("short")) if x)
            elif isinstance(v, list):
                out.extend(str(x) for x in v if isinstance(x, str))
    elif isinstance(data, list):
        for v in data:
            if isinstance(v, dict):
                out.extend(str(x) for x in (v.get("name"), v.get("short")) if x)
            elif isinstance(v, str):
                out.append(v)
    return out


# ---------------------------------------------------------------- 形状判断


def attr_kinds(qn: str) -> list[str]:
    """这句在问上一件事的哪些属性（材料/时长/费用/联系人/时间/流程/条件/预约）。"""
    out = []
    for kind, pat in ATTR_PATTERNS.items():
        if re.search(pat, qn):
            if kind == "hours" and not HOURS_CONTEXT.search(qn):
                continue
            out.append(kind)
    return out


def anaphora(qn: str) -> bool:
    return qn.startswith(ANAPHORA_HEAD)


def followup_shape(qn: str, *, attrs: list[str] | None = None) -> tuple[bool, str]:
    """追问形状成立不成立：谁/什么/多久/材料这类「离了上文不成句」的形状。"""
    attrs = attr_kinds(qn) if attrs is None else attrs
    if anaphora(qn):
        return True, "anaphora"
    if attrs:
        return True, "attr:" + ",".join(attrs)
    if qn.endswith(SUFFIX_PARTICLES) and len(qn) <= 6:
        return True, "short-particle"
    return False, ""


def cosine(a: Any, b: Any) -> float:
    try:
        na = math.sqrt(sum(float(x) * float(x) for x in a))
        nb = math.sqrt(sum(float(x) * float(x) for x in b))
        if not na or not nb:
            return 0.0
        return sum(float(x) * float(y) for x, y in zip(a, b)) / (na * nb)
    except Exception:  # noqa: BLE001
        return 0.0


def residue(term: str) -> str:
    """剥掉通用词/属性词之后还剩什么：`线上办理`→空，`换宿舍`→`换宿舍`。

    判断「词表里命中的这个词是不是**另有主体**」时用它——直接看命中词本身，
    `线上办理`、`需要预约`、`截止时间` 这些全是属性说法，会被当成新主体误杀真追问。
    """
    r = term
    for w in sorted(GENERIC | ATTR_WORDS, key=len, reverse=True):
        if w and w in r:
            r = r.replace(w, "")
    return r


def _substrings(s: str, maxlen: int = 8) -> set[str]:
    out: set[str] = set()
    n = len(s)
    for i in range(n):
        for j in range(i + 2, min(n, i + maxlen) + 1):
            out.add(s[i:j])
    return out


# ---------------------------------------------------------------- 决策对象


@dataclass
class Decision:
    label: str                 # "followup" | "new"
    query: str                 # 检索串
    reason: str
    gate: int = 0
    sim: float = 0.0
    attrs: list[str] = field(default_factory=list)
    a_top: str = ""
    a_score: int = 0
    b_top: str = ""
    b_score: int = 0
    rewrite: str = ""          # 被闸门 2 接受、真正用于检索的改写句
    gate2: str = ""            # 闸门 2 的结论：ok / fail_substring / fail_new_words / ...
    foreign: list[str] = field(default_factory=list)

    @property
    def linked(self) -> bool:
        return self.label == "followup"

    def as_dict(self) -> dict[str, Any]:
        return {"label": self.label, "query": self.query, "reason": self.reason,
                "gate": self.gate, "sim": round(self.sim, 4), "attrs": self.attrs,
                "a_top": self.a_top, "a_score": self.a_score,
                "b_top": self.b_top, "b_score": self.b_score,
                "rewrite": self.rewrite, "gate2": self.gate2,
                "foreign": self.foreign[:6]}


# ---------------------------------------------------------------- 解析器


class Resolver:
    """注入式解析器。所有依赖都是 callable / 数据，单测不用起服务、不用联网。

    rank_fn(q) -> list[(item_id, score)]，按分数降序（app.rank 就是）。
    embed_fn(text) -> vector（可选；没有就退化成纯规则 + 名词级重合）。
    anchor_vec_fn(item_id) -> vector（可选；默认用 embed_fn 编码锚点标题）。
    llm_fn(prompt) -> dict（可选；闸门 2 用，没给就跳过闸门 2）。
    """

    def __init__(self, items: Sequence[dict[str, Any]], *,
                 rank_fn: Callable[[str], list[tuple[str, int]]],
                 lexicon: Lexicon | None = None, places: Iterable[str] = (),
                 depts: Iterable[str] = (),
                 embed_fn: Callable[[str], Any] | None = None,
                 anchor_vec_fn: Callable[[str], Any] | None = None,
                 llm_fn: Callable[[str], dict[str, Any]] | None = None,
                 use_gate1: bool = True, use_gate2: bool = True,
                 gate2_veto: bool = False,
                 far: float = FAR, near: float = NEAR,
                 a_strong: int = A_STRONG, a_weak: int = A_WEAK) -> None:
        self.items = {str(it.get("id")): it for it in items}
        self.rank = rank_fn
        self.lex = lexicon or Lexicon(items, places, depts)
        self.embed = embed_fn
        self.anchor_vec = anchor_vec_fn
        self.llm = llm_fn
        self.use_gate1 = use_gate1
        self.use_gate2 = use_gate2
        self.gate2_veto = gate2_veto
        self.far = far
        self.near = near
        self.a_strong = a_strong
        self.a_weak = a_weak
        self._vec_cache: dict[str, Any] = {}
        self.calls = {"gate1": 0, "gate2": 0}

    # -------------------------------------------------------- 小工具

    def _anchor_terms(self, aid: str) -> set[str]:
        it = self.items.get(aid) or {}
        t = item_terms(it)
        for s in str(it.get("title") or "").replace("/", " ").replace("（", " ").split():
            n = norm(s)
            if len(n) >= 2:
                t.add(n)
        # 问句式标题（「学生如何请假」）的主题名「请假」也要算锚点词，
        # 否则「请假（病假/事假…）」那条永远判不成同族。
        if is_question_title(str(it.get("title") or "")):
            tn = norm(topic_name(str(it.get("title") or "")))
            if len(tn) >= 2:
                t.add(tn)
        return t

    def _same_topic(self, item_id: str, aid: str, a_terms: set[str]) -> bool:
        if item_id == aid:
            return True
        other = self.items.get(item_id) or {}
        ot = item_terms(other)
        core = {x for x in a_terms if x not in GENERIC and len(x) >= 2}
        if core & ot:
            return True
        # 「校外访问（VPN）」的锚点基名「校外访问」出现在「校外 VPN 使用说明」的标题里：
        # 条目名对不上、主题是同一件事，靠 3 字以上的子串认出来
        hay = norm(str(other.get("title") or "")) + "".join(norm(a) for a in (other.get("aliases") or []))
        return any(len(x) >= 3 and x in hay for x in core)

    def _vec(self, text: str) -> Any:
        if not self.embed:
            return None
        if text in self._vec_cache:
            return self._vec_cache[text]
        v = self.embed(text)
        if len(self._vec_cache) > 512:
            self._vec_cache.clear()
        self._vec_cache[text] = v
        return v

    def sim_to_anchor(self, q: str, aid: str) -> float:
        if not self.embed:
            return 0.0
        if self.anchor_vec:
            av = self.anchor_vec(aid)
        else:
            av = self._vec(str((self.items.get(aid) or {}).get("title") or ""))
        qv = self._vec(q)
        if av is None or qv is None:
            return 0.0
        return cosine(qv, av)

    # -------------------------------------------------------- 闸门 0

    def gate0(self, q: str, qn: str, aid: str, a_terms: set[str]) -> Decision | None:
        """返回 None 表示「没被硬信号拦下」，可以进下一层。"""
        attrs = attr_kinds(qn)
        if qn and any(t and t in qn for t in a_terms):
            # R0：原句自己点了锚点 → 是在接着说这件事，独立性检查全部跳过
            return None
        if SELF_CONTAINED.search(qn):
            return Decision("new", q, "gate0.self_contained", 0, attrs=attrs)
        hits = self.lex.hits(qn)
        foreign = []
        for term, df in hits.items():
            if term in a_terms or any(term in a for a in a_terms if len(a) >= 2):
                continue                     # 锚点自己的词 / 「转专业」里的「专业」
            if len(residue(term)) < 2:
                continue                     # 只是属性说法（线上办理 / 需要预约 / 截止时间）
            if df <= HUB_DF:
                foreign.append(term)
        # 长的优先：命中「换宿舍」就不用再报「宿舍」
        foreign = [t for t in sorted(foreign, key=len, reverse=True)
                   if not any(t != o and t in o for o in foreign)]
        if foreign:
            return Decision("new", q, "gate0.foreign_item", 0, attrs=attrs, foreign=foreign)
        head = self.lex.opens_with(qn)
        if head and len(residue(head)) >= 2 and head not in a_terms:
            return Decision("new", q, "gate0.opens_with_subject", 0, attrs=attrs, foreign=[head])
        ok, _why = followup_shape(qn, attrs=attrs)
        if not ok:
            return Decision("new", q, "gate0.not_followup_shape", 0, attrs=attrs)
        return None

    # -------------------------------------------------------- 闸门 2

    def gate2(self, q: str, anchor: dict[str, Any], a_terms: set[str],
              sim: float) -> tuple[bool, str, str]:
        """受约束改写：只回答「这句能不能补成独立问句」，并给出改写句。

        三条硬约束任一不过 → 丢弃改写（返回 need=False）。
        """
        if not self.llm:
            return False, "", "no_llm"
        title = str(anchor.get("title") or "")
        prompt = (
            "你在判断一句话是不是在追问上一轮问过的事项。\n"
            f"上一轮问的是：{title}\n"
            f"这句话是：{q}\n\n"
            "只回答 JSON：{\"need\": true/false, \"query\": \"补全后的独立问句\"}。\n"
            "need=true 仅当这句话必须结合上一轮才能听懂（缺主语，或只说「那」「还」「几天」这类）。\n"
            "query 只能在这句话前面补上上一轮的事项名，不要替换、不要增删原有字词，"
            "不要引入上一轮没说过的名词。听不懂就 need=false，query 留空。"
        )
        try:
            data = self.llm(prompt) or {}
        except Exception:  # noqa: BLE001
            return False, "", "llm_error"
        self.calls["gate2"] += 1
        if not isinstance(data, dict) or not data.get("need"):
            return False, str((data or {}).get("query") or "")[:60], "llm_no"
        rw = str(data.get("query") or "").strip()
        if not rw:
            return False, "", "empty_rewrite"
        # 约束 1：原句必须是改写句的子串（只允许在开头补锚点 / 换掉开头指代词）
        stripped = re.sub(r"^(那|那么|这个|那个|它|还)", "", q) or q
        if q not in rw and stripped not in rw:
            return False, rw, "fail_substring"
        # 约束 2：新增的词必须出现在锚点主题词集合里
        rest = norm(rw)
        for chunk in sorted({q, stripped}, key=len, reverse=True):
            rest = rest.replace(norm(chunk), " ", 1)
        for t in sorted(a_terms | ATTR_WORDS | GENERIC, key=len, reverse=True):
            rest = rest.replace(t, " ")
        if re.search(r"[一-鿿a-z0-9]{2,}", rest):
            return False, rw, "fail_new_words"
        # 约束 3：改写句到锚点的相似度不许下降
        if self.embed:
            s2 = self.sim_to_anchor(rw, str(anchor.get("id") or ""))
            if s2 + 1e-6 < sim:
                return False, rw, "fail_sim_drop"
        return True, rw, "ok"

    # -------------------------------------------------------- 仲裁

    def arbitrate(self, q: str, completed: str, aid: str, a_terms: set[str]) -> Decision:
        rows_a = self.rank(q) or []
        rows_b = self.rank(completed) or []
        a_top, a_sc = (str(rows_a[0][0]), int(rows_a[0][1])) if rows_a else ("", 0)
        b_top, b_sc = (str(rows_b[0][0]), int(rows_b[0][1])) if rows_b else ("", 0)
        d = Decision("new", q, "arb.weak", 2, a_top=a_top, a_score=a_sc,
                     b_top=b_top, b_score=b_sc)
        a_same = bool(a_top) and (a_top == aid or self._same_topic(a_top, aid, a_terms))
        b_ok = bool(b_top) and (
            (b_top == aid and b_sc >= self.a_weak)
            or (b_sc >= self.a_strong and self._same_topic(b_top, aid, a_terms)))
        if a_sc >= self.a_strong:
            if not a_same:
                d.reason = "arb.a_strong_other_topic"
                return d
            d.label, d.query, d.reason = "followup", completed, "arb.a_strong_anchor"
            return d
        if b_ok:
            d.label, d.query = "followup", completed
            d.reason = "arb.a_weak_b_anchor" if a_sc < self.a_weak else "arb.a_mid_b_anchor"
            return d
        # 同族放宽：原句自己落到别的事上（A 的 top1 不在锚点族），B 却稳稳落回锚点这一族。
        # 「学生如何请假，那要带什么材料」A=研究生新生报到 73、B=请假 75，原来两条都不满足。
        b_same = bool(b_top) and self._same_topic(b_top, aid, a_terms)
        if (not a_same) and b_same and b_sc >= FAMILY_MIN:
            d.label, d.query = "followup", completed
            d.reason = "arb.family"
            return d
        d.reason = "arb.no_b_anchor"
        return d

    # -------------------------------------------------------- 入口

    def decide(self, q: str, anchor: dict[str, Any] | str | None, *,
               prev_q: str = "") -> "Decision | None":
        """接上了返回 Decision，没接上返回 None（app 侧只关心这一个问题）。"""
        d = self.resolve(q, anchor, prev_q=prev_q)
        return d if d.linked else None

    def anchor_from_history(self, prev_questions: Sequence[str], text: str,
                            *, min_score: int = 80) -> str | None:
        """服务端锚点重建：前端刷新后 state.item 丢了，但历史里还留着上一问。

        只有上一问单轮检索很确定（≥min_score）才拿它当锚点——宁可没有锚点，
        也别拿一句搜不准的话当上一条事项。
        """
        for q in reversed(list(prev_questions)):
            q = str(q or "").strip()
            if not q or q == str(text or "").strip():
                continue
            rows = self.rank(q) or []
            if rows and int(rows[0][1]) >= min_score:
                return str(rows[0][0])
        return None

    def resolve(self, q: str, anchor: dict[str, Any] | str | None, *,
                prev_q: str = "") -> Decision:
        if isinstance(anchor, str):
            anchor = self.items.get(anchor) or {}
        if not anchor or not anchor.get("id"):
            return Decision("new", q, "no_anchor", 0)
        aid = str(anchor["id"])
        qn = norm(q)
        a_terms = self._anchor_terms(aid)
        d0 = self.gate0(q, qn, aid, a_terms)
        if d0 is not None:
            return d0
        sim = 0.0
        if self.use_gate1 and self.embed:
            self.calls["gate1"] += 1
            sim = self.sim_to_anchor(q, aid)
            if sim < self.far:
                return Decision("new", q, "gate1.far", 1, sim, attrs=attr_kinds(qn))
        # 问句式标题（「学生如何请假」）当锚点时，补全串用主题名（「请假，那要带什么材料」）：
        # 原标题里的「学生/如何」会把检索带偏，锚点自己反而掉到第 3 名。
        head = str(anchor.get("title") or "")
        tn = topic_name(head) if is_question_title(head) else head
        completed = f"{tn if tn and tn != head else head}，{q}"
        rewrite, why = "", ""
        if self.use_gate2 and self.llm and sim < self.near:
            need, rw, why = self.gate2(q, anchor, a_terms, sim)
            if need:
                rewrite, completed = rw, rw
            elif self.gate2_veto:
                d = Decision("new", q, "gate2.veto:" + why, 2, sim,
                             attrs=attr_kinds(qn), gate2=why)
                return d
        d = self.arbitrate(q, completed, aid, a_terms)
        d.gate = 2 if why else 1
        d.sim = sim
        d.rewrite = rewrite
        d.gate2 = why
        d.attrs = attr_kinds(qn)
        return d


# ---------------------------------------------------------------- 退化路径


def noun_overlap(text: str, anchor_terms: set[str], lex: Lexicon) -> int:
    """没有 embedding 时的替代信号：句子里有几个实词与锚点主题词重合。"""
    qn = norm(text)
    hits = lex.hits(qn)
    return sum(1 for t in hits if t in anchor_terms or any(t in a for a in anchor_terms))


# ---------------------------------------------------------------- app 侧工厂


def build_resolver(items: Sequence[dict[str, Any]], *, rank_fn, places=(),
                   depts=(), engine=None, llm_fn=None, use_gate1: bool = True,
                   use_gate2: bool = True, gate2_veto: bool = False) -> Resolver:
    """给 app.py 用的一行工厂：embedding 直接复用检索引擎的查询向量与条目向量。

    取不到 dense 就自动退化（embed_fn=None），不抛异常、不影响主流程。
    """
    embed_fn = anchor_vec_fn = None
    if engine is not None:
        dense = getattr(engine, "dense", None)
        if dense is not None:
            def embed_fn(text: str):                      # noqa: F811
                try:
                    return engine._query_vec(text)
                except Exception:  # noqa: BLE001
                    return None

            def anchor_vec_fn(item_id: str):               # noqa: F811
                try:
                    j = dense.id_of.get(item_id)
                    return None if j is None else dense.mat[j]
                except Exception:  # noqa: BLE001
                    return None
    lex = Lexicon(items, places, depts)
    return Resolver(items, rank_fn=rank_fn, lexicon=lex, embed_fn=embed_fn,
                    anchor_vec_fn=anchor_vec_fn, llm_fn=llm_fn,
                    use_gate1=use_gate1, use_gate2=use_gate2,
                    gate2_veto=gate2_veto)
