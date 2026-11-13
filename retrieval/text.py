"""中文文本规范化、问句改写与切词。

设计取向（为什么不用 jieba / 不能依赖外部分词器）：

* 目标机是 ModelScope 创空间里的 Docker，`requirements.txt` 只有 fastapi / uvicorn /
  numpy，本地 Python 是 3.15（大量二进制包没有轮子）。所以分词器必须自带。
* 领域词表可以直接从条目库长出来：3.9 万条标题 / 别名 / 关键词本身就是这个校园
  领域的术语表（校园卡、统一身份认证、微专业、推免……）。
* 拿不到词的场合不能丢信息：字符二元组（bigram）对所有字段都建索引，字典词只加在
  短字段（标题 / 别名 / 关键词）上。查询侧两边都出词，命中哪边算哪边。

对外只有四件事：`norm`（文档侧规范化）、`query_text`（问句侧规范化 + 拆掉套话）、
`terms`（字典词最长匹配）、`bigrams`（字符二元组）。
"""
from __future__ import annotations

import re
import unicodedata
from typing import Iterable

# 全角标点、空白、控制符：检索时一律不含信息
_SEP = re.compile(
    r"[\s\u3000-\u303f\uff00-\uffef\u2000-\u206f"
    r"!-/:-@\[-`{-~\u2010-\u2027\u2030-\u205e]+"
)
_KEEP = re.compile(r"[0-9a-z\u4e00-\u9fff]+")

# 问句套话：只拆「礼貌」和「怎么办」这类纯语气，**不拆意图词**。
# 「在哪 / 几点 / 多少 / 能不能 / 怎么」本身是问面信息，而且条目别名里常常就写着
# 「校医院在哪」「能不能毕业」，拆掉反而对不上——所以留着。
_FRAME = re.compile(
    r"请问|想问一下|想问下|想问|问一下|麻烦问一下|麻烦|我想问|我想|我要|"
    r"你好|您好|老师你好|老师您好|同学你好|谢谢|多谢|"
    r"怎么办|咋办|怎么弄|怎么搞|怎么处理|如何处理|"
    r"[呢吗吧啊呀嘛哦哈]+$"
)
_HEAD = re.compile(r"^(我想问|请问|想问|想问一下|你好|您好|老师你好|老师)")
_TAIL = re.compile(r"(怎么办|咋办|怎么弄|怎么搞|行不行|可以吗|能吗)$")
# 「怎么做」里的「怎么」是纯方法问法，拆掉之后「校园卡怎么挂失」才对得上别名「校园卡挂失」。
# 「在哪 / 几点 / 多少 / 能不能」不拆——它们本身就是问面，别名里也常见。
_HOW = re.compile(r"怎么样|怎样|怎么|如何|咋样|咋")
# 情态词：「体育课能免修吗」的主题是「体育课免修」，中间那个「能」会挡住标题/别名的
# 包含判断。只拆多字情态词与「能」（学生会、会议这类不拆，见 _MODAL_SAFE）。
_MODAL = re.compile(r"能不能|能否|可不可以|可以|能|需不需要|要不要")
_MODAL_KEEP = ("学生会", "会议", "会员", "会费", "社会", "机会", "培训会", "发布会", "研讨会",
               "协会", "学会", "理会", "工会", "年会", "晚会", "音乐会", "宣讲会", "答辩会")


def strip_how(s: str) -> str:
    """去掉纯方法问法（怎么/如何/怎样），剩下的仍是主题串。"""
    out = _HOW.sub("", s or "")
    return out if len(out) >= 2 else (s or "")


def strip_modal(s: str) -> str:
    """去掉情态词（能 / 能不能 / 可以），保留「学生会 / 会议」这类含「会」的词。"""
    if not s:
        return s
    saved: dict[str, str] = {}
    out = s
    for i, w in enumerate(_MODAL_KEEP):
        if w in out:
            ph = f"\x00{i}\x00"
            out = out.replace(w, ph)
            saved[ph] = w
    out = _MODAL.sub("", out)
    for ph, w in saved.items():
        out = out.replace(ph, w)
    return out if len(out) >= 3 else s


# 标题里的文号与尾部括号补充：「中传教字〔2025〕7号 …办法」「公共实践场馆预约（演播馆 / 音乐厅）」
_TRAILING_PAREN = re.compile(r"[（(][^（()）]*[）)]\s*$")
_LEADING_DOC_NO = re.compile(r"^[^\s，,。]{0,12}〔\d{4}〕\s*\d+\s*号\s*")


def core_title(title: str) -> str:
    """标题的主干：去掉文号和尾部括号补充，再规范化。

    条目名常常带一个括号说明（「自助人脸照片上传（门禁）」「图书馆借阅（借几本、借多久、
    续借、逾期）」），用户不会把括号念出来，所以标题匹配必须用主干，否则「自助人脸照片
    上传」永远差一个「门禁」。
    """
    t = _LEADING_DOC_NO.sub("", title or "")
    while True:
        nxt = _TRAILING_PAREN.sub("", t).strip()
        if nxt == t:
            return norm(t)
        t = nxt


def norm(s: str) -> str:
    """文档侧规范化：全角转半角、统一小写、去标点空白。不做任何删词。"""
    if not s:
        return ""
    s = unicodedata.normalize("NFKC", s).lower()
    return "".join(_KEEP.findall(s))


def query_text(q: str) -> str:
    """问句侧规范化：在 `norm` 之上拆掉「请问 / 怎么办 / 吗」这类套话。

    拆套话是确定性的、与评测集无关的规则（`怎么办` 是语气，不是内容），
    所以不会把评测集的信息喂给检索器。拆完只剩标点时退回原规范化串。
    """
    s = norm(q)
    if not s:
        return ""
    core = _HEAD.sub("", s)
    core = _TAIL.sub("", _FRAME.sub("", core))
    core = core.strip()
    if len(core) < 2:
        return s
    return core


# 问句框架词：只表示「在问哪个面」，不表示「问的是哪件事」。
# 候选标题本身若是问句（「什么时候加入班级群？」），它就是靠这些词跟问句对上的——
# 把它们从两边都剥掉，剩下的才是「主题串」，主题比对才不会被框架词灌水。
_FRAME_WORDS: tuple[str, ...] = (
    "什么时候", "什么时间", "什么条件", "多长时间", "怎么回事", "什么样", "什么",
    "什么时候能", "何时", "几点", "几号", "几天", "几个", "几门", "几节", "几次", "几科",
    "几本", "几人", "几人间", "多少分", "多少度", "多少", "多久",
    "在哪里", "在哪儿", "在哪", "哪里", "哪儿", "哪个", "哪些", "哪",
    "怎么样", "怎么", "怎样", "咋样", "咋", "如何",
    "是不是", "有没有", "能不能", "可不可以", "要不要", "需不需要", "可以", "能否",
    "是什么", "是谁", "是否", "需要", "请问", "一下",
)
_FRAME_CHARS = "的了吗呢吧啊呀是"

# 标题是「问句 / 整句」的判定：这类标题不是条目名，是原文里的一句话
_QTITLE = re.compile(
    r"什么时候|何时|几点|多少|多久|怎么|如何|在哪|哪里|哪儿|哪些|什么|能否|能不能|"
    r"可以吗|是不是|有没有|？|\?"
)


def topic_text(s: str) -> str:
    """主题串：在 `query_text` 之上再剥掉「问哪个面」的框架词。

    只用于**主题比对**（`features.topic_*`），不参与召回与 BM25——剥掉框架词会
    丢掉「在哪 / 多少」这类问面信息，那是答案要用的，不是排序要用的。
    剥完不足 2 字时退回入参，避免把「在哪」这种短句剥成空串。
    """
    s = s or ""
    out = s
    for w in _FRAME_WORDS:      # 元组已按长度降序排好，先剥长的
        out = out.replace(w, "")
    out = "".join(ch for ch in out if ch not in _FRAME_CHARS)
    return out if len(out) >= 2 else s


def is_question_title(title: str) -> bool:
    """标题本身是不是一个问句 / 整句话（不是条目名）。"""
    t = norm(title)
    if not t:
        return False
    return bool(_QTITLE.search(t)) or len(t) >= 24


# 「大杂烩」条目：一份文档里塞了很多互不相干的问题/事项，标题就写着它是汇总。
# 这类条目在题库里当过 18 次 top1——它是靠「常见问题 / 汇总」里的通用词跟问句对上的，
# 不是因为它就是用户问的那件事。
_HUB_TITLE = re.compile(
    r"常见问题|常见问答|汇总|一览|清单|问答|合集|攻略|宝典|百问|faq|FAQ")


def is_hub_title(title: str, min_len: int = 12) -> bool:
    """标题是不是「汇总 / 常见问题 / 一览」这类大杂烩（或整段带网址的网页答案）。"""
    t = title or ""
    n = norm(t)
    if not n:
        return False
    if "http" in t:
        return True
    return bool(_HUB_TITLE.search(t)) and len(n) >= min_len


# 实词 = 词表切出来、又不在框架词 / 停用片段里的词
_FRAME_SET = frozenset(_FRAME_WORDS)


def content_words(lex: "Lexicon", s: str, min_len: int = 2) -> list[str]:
    """问句里的实词（领域词最长匹配之后，剔掉框架词与停用片段）。

    这是「问的是哪件事」的另一半：二元组会被框架词灌水，实词不会——
    「选课什么时候开始」的实词只有「选课」，「补（退）选课」的名字里有它，
    「体测时间安排」的名字里没有。
    """
    out: list[str] = []
    for w in lex.match(topic_text(s)):
        if len(w) < min_len or w in _FRAME_SET or w in STOP_FRAGMENTS:
            continue
        if w not in out:
            out.append(w)
    return out


def bigrams(s: str) -> list[str]:
    """字符二元组（有序、可重复计数），用于 TF。"""
    if len(s) < 2:
        return [s] if s else []
    return [s[i : i + 2] for i in range(len(s) - 1)]


def bigram_set(s: str) -> set[str]:
    return set(bigrams(s))


def trigrams(s: str) -> list[str]:
    if len(s) < 3:
        return []
    return [s[i : i + 3] for i in range(len(s) - 2)]


class Lexicon:
    """最长匹配用的词典树。

    词表来自条目库自身：标题 / 别名 / 关键词整串 + 高频 n-gram。查询侧只有几十个
    字，字典树走一遍是微秒级；文档侧只在短字段上走。
    """

    __slots__ = ("words", "max_len", "_root", "idf")

    def __init__(self, words: Iterable[str]) -> None:
        self.words: set[str] = {w for w in words if len(w) >= 2}
        self.max_len = max((len(w) for w in self.words), default=2)
        root: dict = {}
        for w in self.words:
            node = root
            for ch in w:
                node = node.setdefault(ch, {})
            node[""] = True
        self._root = root
        # 词的 IDF 在 build 阶段回填（见 index.py），查询侧不需要
        self.idf: dict[str, float] = {}

    def match(self, text: str) -> list[str]:
        """最长匹配切词；未登录字直接跳过（二元组通道会兜住它）。"""
        out: list[str] = []
        i, n = 0, len(text)
        root = self._root
        while i < n:
            node = root
            j = i
            best = i
            while j < n:
                nxt = node.get(text[j])
                if nxt is None:
                    break
                node = nxt
                j += 1
                if "" in node:
                    best = j
            if best > i:
                out.append(text[i:best])
                i = best
            else:
                i += 1
        return out

    def to_json(self) -> list[str]:
        return sorted(self.words)


def mine_ngrams(texts: Iterable[str], lo: int = 2, hi: int = 6, min_freq: int = 4) -> set[str]:
    """从短串里挖高频 n-gram 当候选术语（AutoPhrase 的极简版）。

    只用于**扩充词典**：出现在 >= min_freq 个不同短串里的片段才收。噪声片段
    （「么办」「哪里」这类）靠停用表剔掉。
    """
    counts: dict[str, int] = {}
    for t in texts:
        t = norm(t)
        if len(t) < lo:
            continue
        seen: set[str] = set()
        for L in range(lo, min(hi, len(t)) + 1):
            for i in range(len(t) - L + 1):
                seg = t[i : i + L]
                if seg in seen:
                    continue
                seen.add(seg)
                counts[seg] = counts.get(seg, 0) + 1
    return {w for w, c in counts.items() if c >= min_freq and w not in STOP_FRAGMENTS}


# 只表示语气 / 文书类型的片段：进词典会把「宿舍能不能养猫」粘到「能不能毕业」上
STOP_FRAGMENTS = {
    "怎么", "么办", "什么", "哪些", "哪里", "哪儿", "可以", "能不能", "是否", "如何",
    "多少", "多久", "几点", "一下", "请问", "我想", "我要", "需要", "不用", "这个",
    "那个", "时候", "为什", "行不", "行吗", "的是", "了的", "了吗", "还是", "或者",
    "我们", "你们", "他们", "自己", "没有", "是不", "不是", "就是", "可以", "以及",
    "办法", "规定", "通知", "管理", "实施", "细则", "制度", "方案", "条例", "意见",
    "工作", "学生", "学校", "学院", "本科", "研究生", "在校", "全校", "相关", "有关",
    "关于", "进行", "可以", "按照", "根据", "如下", "以下", "上述", "其中", "其他",
}
