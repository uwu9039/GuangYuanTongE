"""拒答层：判断「库里到底有没有这条问题的答案」。

按「先排除明确不归我们管，再看有没有命中」的顺序判，每一档都有明确理由：

1. **个人数据**（我的余额 / 我的绩点 / 我的课表）——知识库不存个人数据，答不了，
   给官方查询入口。
2. **别的学校**（清华 / 北大 / 北师大的办事流程）——库里只有中国传媒大学的资料。
3. **库外领域**（天气 / 股票 / 写代码 / 翻译 / 减肥 / 驾校）——用「语料覆盖」兜底：
   问句里有多少 CJK 字在 3.9 万条的语料里根本没出现过，最长连续未覆盖片段有多长；
   再加一张「校园助手的职责边界」意图词表。
4. **没有足够接近的条目**——冠军候选的字面覆盖与排序分都过低。

规则是确定性的、可审计的；阈值在 `_eval/gate_probe.py` 里用开发集标定。
不训练模型：拒答只有一两百个样本，规则比小模型更稳、更好解释。
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

GATE_FEATURES: tuple[str, ...] = (
    "top_prob", "margin", "cov_short", "exact", "phrase_hit", "cov_corpus",
    "oov_ratio", "longest_oov", "n_above", "log_len", "kind_hand", "cov_body",
)

# ---------------------------------------------------------------- 规则词表

# 1) 个人数据：问的是「我这个人的数据」，不是「怎么办」。
#
#    2026-11 版把它写成三张结构化表，覆盖「我的X」的变体而不是具体问句：
#      · 数据项名词（我的余额 / 我的准考证号 / 我的学籍状态 / 我的门禁记录 …）；
#      · 第一人称 + 量词动词（我期末考了几门 / 我这个月花了多少钱）；
#      · 第一人称 + 时间 + 事项（我什么时候能毕业）。
#    「我宿舍电费超了会不会断电」问的是政策不是数据：名词分支要求「我的」或贴身 2 字，
#    「宿舍电费」隔了 2 字以上，不会误伤。
_PERSONAL_NOUN: tuple[str, ...] = (
    "余额", "还剩", "剩余", "欠费", "欠款", "罚款", "罚金", "滞纳金",
    "流水", "消费记录", "交易记录", "账单", "明细",
    "绩点", "gpa", "平均分", "成绩", "分数", "排名", "名次",
    "课表", "课程表", "学号", "工号",
    "借阅记录", "借书记录", "借了", "借书",
    "进度", "结果", "记录", "到期",
    "准考证", "考号", "身份证号", "手机号", "电话号码", "联系方式",
    "密码", "口令", "验证码",
    "学籍状态", "学籍信息", "学籍档案",
    "门禁记录", "打卡记录", "考勤记录",
)
# 弱数据项：单独出现时可能是在问「政策」（「我的宿舍电费超了会不会断电」），
# 只有后面紧跟一个问值词才算问数据。
_PERSONAL_NOUN_WEAK: tuple[str, ...] = (
    "电费", "水费", "流量", "网费", "体测成绩", "四六级成绩", "消费金额",
)
_VALUE_WORD = re.compile(r"多少|几度|几块|几元|几分|余额|还剩|剩余|欠|是什么|是几|查|"
                         r"什么时候|何时|哪天|几号")
# 贴身数据项：不带「的」也算（「我余额还有多少」「我还剩几块钱」）。
_PERSONAL_TIGHT: tuple[str, ...] = (
    "余额", "还剩", "剩余", "欠费", "欠款", "罚款", "绩点", "排名", "课表", "密码",
)
PERSONAL_MINE = re.compile(
    r"我的[^，。？?]{0,10}?(?:" + "|".join(_PERSONAL_NOUN) + r")"
    r"|我的[^，。？?]{0,8}?(?:" + "|".join(_PERSONAL_NOUN_WEAK) + r")[^，。？?]{0,4}?"
    r"(?:" + _VALUE_WORD.pattern + r")"
    r"|我[^，。？?]{0,2}?(?:" + "|".join(_PERSONAL_TIGHT) + r")"
    r"|我[^，。？?]{0,8}?(?:余额|还剩|剩余|欠费|欠款|罚款|滞纳金)"
)
# 第一人称 + 量词动词：问的是自己那份数据里有几个 / 花了多少。
PERSONAL_QUANT = re.compile(
    r"我[^，。？?]{0,8}?(?:考了|选了|修了|上了|旷了|逃了|挂了|扣了|花了|消费了|刷了|用了|"
    r"交了|充了|借了|欠了|领了)[^，。？?]{0,6}?"
    r"(?:几门|几节|几次|几科|几本|几个|几块|几元|多少钱|多少分|几分|多少)"
)
# 第一人称 + 时间 + 毕业/离校这类「我的进度」问法。
PERSONAL_WHEN = re.compile(
    r"我[^，。？?]{0,8}?(?:什么时候|何时|哪天|几号)[^，。？?]{0,6}?"
    r"(?:能|可以|才能|会)?(?:毕业|拿毕业证|拿证|离校|答辩|入学)"
)
PERSONAL_SELF = re.compile(
    r"我(上次|这学期|这月|本周|今天|现在|之前|上学期|这学年)[^，。？?]{0,6}?"
    r"(多少分|几分|多少|是什么|几点|怎么样)"
    r"|我[^，。？?]{0,8}?(学期|月|年)的?[^，。？?]{0,6}?(课表|成绩|旷了几次|考勤|排名)"
    r"|我[^，。？?]{0,6}?(饭卡|校园卡|一卡通|卡里)[^，。？?]{0,8}?(余额|还剩|还有多少|多少钱)"
)

# 第二轮：把「个人数据」判成**句型**，不是名词表。
#   主语是第一人称，并且（问一个「值」 或 提到一个「属于某个人的数据项」）才算。
# 上一轮的表在「换一批没见过的说法」上只拒了 3/10（学分 / 排第几 / 批下来 / 借的书 /
# 欠费 / 上学期 都漏），所以这里把「数据项」和「问值」分成两列词表，句型只认结构。
PERSONAL_SUBJECT = re.compile(r"^我(?!们|校|院|班|系|室友|同学|的室友)")
# 个人数据项（按「数据项」列，不按问句列）
_PERSONAL_ITEM: tuple[str, ...] = (
    "学分", "已修学分", "成绩", "分数", "绩点", "gpa", "平均分", "排名", "名次", "排第几",
    "余额", "欠费", "欠款", "罚款", "罚金", "滞纳金", "账单", "流水",
    "课表", "课程表", "选课记录", "学号", "工号", "准考证", "考号", "证书", "证书号",
    "借的书", "借阅", "借了", "续借", "到期", "超期", "归还日期",
    "记录", "门禁", "打卡", "考勤", "进度", "结果", "状态", "编号",
    "密码", "验证码", "手机号", "身份证号", "绑的",
    "团员关系", "团关系", "党团关系", "学籍", "档案", "处分",
    "体测成绩", "四级成绩", "六级成绩", "四六级成绩",
    "奖学金", "助学金", "助学贷款", "报销", "退款", "缴费记录",
    "在读证明", "成绩单", "证明开具情况",
)
# 问「值」的说法（不含「能不能 / 可不可以 / 是不是」——那些问的多半是政策）
_PERSONAL_VALUE = re.compile(
    r"多少|几门|几科|几次|几本|几分|第几|几名|还剩|剩多少|还有多少|剩下|够不够|够了没|"
    r"够毕业|批没批|批下来|到账|发下来|到期|超期|"
    r"有没有(?!资格|条件|机会|名额|办法|要求|规定|可能)"
)
# 句型层：第一人称 + 问一个「值」，或「我的 + 数据项」。
# 数据项**必须带「的」**：「我上学期申请了助学贷款，这学期转专业了，贷款会断吗」
# 这种长句里 我 与 助学贷款 隔得很远，靠名词表会把它当个人数据误拒；
# 「我借的书什么时候到期 / 我申请的那个奖学金批下来没有」都能被「问值」这一支盖住。
PERSONAL_SENTENCE = re.compile(
    r"我的[^，。？?]{0,10}?(?:" + "|".join(_PERSONAL_ITEM) + r")"
    r"|^(?:我)[^，。？?]{0,12}?(?:" + _PERSONAL_VALUE.pattern + r")"
)
# 「能不能 / 可不可以」问的是政策（「我能不能申请助学贷款」），不是我的那份数据。
POLICY_Q = re.compile(r"能不能|可不可以|能否|是不是可以|是不是能|可以吗|可以么|能办|能申请|能领|能退|能补|能报")
_PERSONAL_RULES = (PERSONAL_MINE, PERSONAL_QUANT, PERSONAL_WHEN, PERSONAL_SELF,
                   PERSONAL_SENTENCE)
# 明确在问「怎么查 / 怎么办 / 出问题了怎么处理」的，属于可答（教方法，不是给数据）。
HOWTO = re.compile(
    r"怎么查|在哪查|如何查|哪里查|在哪看|在哪儿看|哪里看|到哪看|怎么看到|查询入口|"
    r"怎么查询|能不能查|"
    r"怎么开|怎么打印|在哪打印|怎么申请|如何申请|怎么办理|如何办理|在哪办|去哪里办|"
    r"要带什么|需要什么材料|什么流程|流程是什么|怎么改|怎么补|怎么弄|怎么办|咋办|"
    r"怎么操作|如何操作|"
    r"怎么更新|如何更新|怎么预约|在哪预约|怎么领|去哪领|到哪领|怎么还|怎么退|"
    # 异常/处置类：问的是「这种事怎么办」，不是「我的值是多少」
    r"被[^，。？?]{0,6}(注册|绑定|占用|盗|锁|冻结)|不对|错误|异常|多扣|少扣|"
    r"解绑|申诉|注销|更改|修改|更换|挂失|补办|处理"
)

# 2) 别的学校：库里只有中国传媒大学
#    这一档**优先于其他一切规则**：问别校的事，哪怕句子里有「宿舍 / 校园卡」也是别校的。
CUC = ("中国传媒大学", "中传", "传媒大学", "cuc")
OTHER_SCHOOLS = (
    "清华", "北大", "北京大学", "北师大", "北京师范大学", "中国人民大学", "人大",
    "浙江大学", "浙大", "复旦大学", "复旦", "上海交通大学", "上交", "北京航空航天大学",
    "北航", "中央美术学院", "央美", "北京电影学院", "北电", "武汉大学", "武大",
    "中山大学", "南京大学", "南大", "成都理工", "山东大学", "天津大学", "厦门大学",
    "吉林大学", "华中科技", "西安交大", "同济大学", "同济", "南开大学", "南开",
    "东南大学", "湖南大学", "四川大学", "重庆大学", "兰州大学", "郑州大学", "苏州大学",
    "华东师范", "东北大学", "西北工业大学", "北京邮电", "北邮", "北京理工", "北理工",
    "中国农业大学", "中央财经", "对外经贸", "北京外国语", "北外", "中国政法",
    "中国科学院", "社科院", "哈佛", "剑桥", "牛津", "斯坦福", "麻省理工",
    # 第二轮补齐的常见缩写（「中科院的研究生宿舍」落到「中科院分区」就是这里漏了）
    "中科院", "中科大", "国科大", "中国科学院大学", "上财", "央财",
    "北交大", "北京交通", "北科大", "北京科技", "华科", "华中科大", "西交",
    "哈工大", "哈工程", "南理工", "电子科大", "西电", "西安电子",
    "中国民航", "外交学院", "国际关系学院", "中央民族", "中央音乐学院",
    "中央戏剧", "中戏", "上海戏剧", "上戏", "北京舞蹈", "中国音乐学院",
    "北京工业", "首都师范", "首都医科", "首都经贸", "北京联合", "北京工商",
    "华东政法", "西南政法", "中南财经", "上海财经", "中央财经大学", "东北财经",
    "南京师范", "华中师范", "华南师范", "陕西师范", "西南大学", "暨南大学",
    "华南理工", "大连理工", "重庆邮电", "北京协和", "协和医学院",
)

# 3) 职责边界：这些不属于「校园办事」
# 强信号：出现即判越界，哪怕句子里还有「宿舍/食堂」这类校园词
OUT_OF_SCOPE_STRONG = (
    "天气", "气温", "下雨", "空气质量", "雾霾",
    "股票", "基金", "比特币", "汇率", "理财", "期货",
    "写一篇", "帮我写", "翻译", "编程", "爬虫", "python",
    "减肥", "星座", "运势", "算命", "塔罗",
    "租房", "买房", "找房", "驾校", "驾照", "雅思", "托福", "gre", "考研数学",
    "高数", "微积分", "火锅店", "开店", "记忆力", "推荐一本", "好书推荐", "书单",
    "国外大学",
)
# 弱信号：这些词顺带出现在一个校园问题里（「室友打游戏到半夜，我能换宿舍吗」）不算越界
OUT_OF_SCOPE_WEAK = (
    "电影", "电视剧", "综艺", "小说", "游戏", "王者荣耀", "原神", "抖音", "小红书",
    "美食", "餐厅推荐", "外卖", "美甲", "理发推荐", "算法题",
    # 生活消费 / 出行 / 旅游：问句里没有校园服务实词时按库外处理
    # （「食堂哪家好吃」有「食堂」，仍按校内问题回答）
    "好吃", "好玩", "景点", "旅游", "游玩", "火锅", "酒店", "宾馆", "民宿",
    "机票", "飞机票", "火车票", "高铁票", "演唱会", "周边游",
    # 生活消费 / 娱乐 / 自学：问句里没有校园服务实词时按库外处理
    # （「食堂晚上吃什么」有「食堂」，仍按校内问题回答）
    "吃什么", "吃啥", "喝什么", "去哪吃", "晚上吃", "夜宵", "零食",
    "这首歌", "什么歌", "歌名", "歌词", "追剧", "综艺", "动漫", "漫画",
    "去玩", "玩的", "周末去", "露营", "徒步", "桌游", "剧本杀", "密室", "ktv",
    "卡路里", "热量", "食谱", "菜谱", "烘焙",
    "奶茶", "脱单", "相亲", "恋爱", "护肤", "化妆", "穿搭", "发型",
    "手机推荐", "电脑推荐", "笔记本推荐", "装机", "steam",
)
# 校外交通设施：不是「出现就拒」。库里本来就有「电子学生交通卡申领」「到校交通」
# （北京站/西站/南站的乘车路线），所以这一组要**同时**满足两个条件才算库外：
#   ① 出现城市交通设施词（地铁 / 公交 / 机场 / 火车站 …）；
#   ② 问的是「怎么走 / 几号线 / 在哪坐 / 首末班」这类路线问题，或问的是一个具体站点名。
# 「学生坐地铁有优惠的卡去哪领」只有①没有②，正常作答（落到电子学生交通卡）。
# 校车 / 班车不在此列：库里确实没有时刻表与乘车点，出现即判库外（见 OUT_OF_SCOPE_TRANSIT_SOLID）。
OUT_OF_SCOPE_TRANSIT = (
    "地铁", "公交", "机场", "火车站", "北京南站", "北京站", "北京西站", "大兴机场",
)
OUT_OF_SCOPE_TRANSIT_SOLID = ("校车", "班车", "通勤车")
TRANSIT_ROUTE = re.compile(
    r"怎么走|怎么去|怎么坐|坐几号线|几号线|换乘|地铁站|公交站|首班|末班|"
    r"哪儿坐|在哪坐|哪里坐|线路|路线|怎么换|怎么到"
)
CAMPUS_SOFT = ("学校", "校内", "返校", "到校", "校区", "校门口")
# 校情统计与学校简介：办事库里只有「怎么办」，没有「学校有几个学院 / 毕业率多少」
# 这类数字，也没有校训、校史、邮编、招生代码这些学校层面的名片信息。
SCHOOL_FACT = re.compile(
    r"校训|校史|办学历史|建校(时间|年份|于)|学校的?历史|"
    r"邮编|邮政编码|院校代码|招生代码|学校代码|"
    r"双一流|985|211|全国排第|排名第几|学校排名|"
    r"(毕业|就业|升学|考研|推免|上线|四六级|英语四级)率|"
    r"学校(有|一共|总共)?(几|多少)个?(学院|系|专业|学生|教师|老师|人|校区)|"
    r"(学校|学校本部|中传|本校)有(多少|几)(名|个)?(学生|人|老师|教师)"
)
# 这些词出现时不算「越界」：学生交通卡 / 一卡通本身就是校内事项
OUT_OF_SCOPE_OK = ("交通卡", "一卡通", "校园卡", "公交卡")
# 只要问句里出现了校园服务的实词，就按校内问题处理，不拿「游戏/电影/酒店」这类
# 顺带出现的词去越界拒答（「我室友打游戏到半夜，我能换宿舍吗」是宿舍问题）。
CAMPUS_TERMS = (
    "宿舍", "寝室", "公寓", "校园卡", "饭卡", "卡里", "校园网", "无线网", "wifi", "网络",
    "选课", "课程", "上课", "成绩", "绩点", "gpa", "考试", "补考", "重修", "缓考", "学分",
    "毕业", "学籍", "休学", "复学", "退学", "转专业", "推免", "保研", "奖学金", "助学金",
    "助学贷款", "勤工", "补助", "报销", "医疗", "校医院", "体检", "体测", "图书馆", "借阅",
    "续借", "食堂", "就餐", "快递", "快递站", "超市", "打印", "浴室", "洗澡", "热水",
    "电费", "水费", "报修", "后勤", "门禁", "校门", "入校", "校园", "导员", "辅导员",
    "学院", "教务", "学工", "档案", "户口", "户籍", "身份证", "证明", "成绩单", "在读",
    "报名", "申请", "办理", "流程", "材料", "收费", "学费", "住宿费", "缴费", "上课",
    "教师", "课堂", "教室", "讲座", "活动", "社团", "志愿", "心理", "就业", "实习",
)


@dataclass
class Decision:
    refused: bool
    reason: str
    confidence: float
    category: str = ""


def _item_text(item: dict[str, Any] | None, limit: int = 4000) -> str:
    """条目的全部文本（字段 + 引文），给「问面核对」找证据用。"""
    if not isinstance(item, dict):
        return ""
    out: list[str] = []

    def walk(v: Any) -> None:
        if len("".join(out)) > limit:
            return
        if isinstance(v, str):
            if v:
                out.append(v)
        elif isinstance(v, dict):
            for x in v.values():
                walk(x)
        elif isinstance(v, (list, tuple)):
            for x in v:
                walk(x)
        elif isinstance(v, (int, float)):
            out.append(str(v))

    walk(item)
    return "".join(out)


# 4) 问面—字段核对：问句在问哪一种「值」。库里条目是按办事事项存的，只有
#    「电话 / 联系人 / 多少钱 / 几点 / 在哪」这些面有结构化字段；问句要的面在冠军
#    条目里连字面都没有，就是「库里有相关条目但没有这一条」。
FACET_ASKS: tuple[tuple[str, "re.Pattern[str]", tuple[str, ...], "re.Pattern[str]"], ...] = (
    ("contact", re.compile(r"电话|电话号码|号码是多少|联系方式|热线"),
     ("contact", "phone"),
     re.compile(r"\d{3,4}[-－—]?\d{7,8}|1[3-9]\d{9}|邮箱|电话|联系方式")),
    ("person", re.compile(r"是谁|谁是|叫什么名|姓名|负责人是谁"),
     ("contact",),
     re.compile(r"[\u4e00-\u9fff]{2,4}(老师|主任|院长|书记|部长|处长|负责人|科长)")),
)


_FACET_LABEL = {"contact": "电话", "person": "联系人"}
_NULLISH_FIELD = {"", "无", "none", "null", "—", "-", "n/a"}


def facet_missing(core: str, hit: Any) -> str:
    """冠军条目缺「问面」时返回面名，否则空串。

    `hit` 是引擎返回的 Hit（`hit.item` 是完整条目 dict，含结构化字段与引文）。
    只有**库里确实不存**的面才判拒：电话 / 联系人。这两个面在 3.9 万条办事条目里
    大多是空的，问句要它而冠军那一条没有，就是库里没这条答案。
    """
    item = getattr(hit, "item", None)
    text = _item_text(item)
    for name, asks, fields, ev in FACET_ASKS:
        if not asks.search(core):
            continue
        if isinstance(item, dict) and any(
                str(item.get(f) or "").strip().lower() not in _NULLISH_FIELD for f in fields):
            continue
        if ev.search(text):
            continue
        return name
    return ""


def personal_hit(core: str, raw: str = "") -> str:
    """个人数据查询：命中的那句原话片段（空串表示没命中）。

    `raw` 是未拆套话的原问句：`query_text` 会把「怎么办」这类语气拆掉，也就会把
    「我的借阅记录怎么查」拆成「我的借阅记录查」——那本来是在问方法（可答），
    所以 HOWTO 要连原句一起看。
    """
    if HOWTO.search(core) or (raw and HOWTO.search(raw)):
        return ""
    policy_q = bool(POLICY_Q.search(core))
    for rule in _PERSONAL_RULES:
        m = rule.search(core)
        if not m:
            continue
        if rule is PERSONAL_SENTENCE and policy_q:
            continue        # 「我能不能申请助学贷款」问的是政策，不是数据
        return m.group(0)
    return ""


# 「中科院分区」是图书馆的期刊分区表，不是中国科学院这所学校——同形不同义，别误判
SCHOOL_HOMONYM = re.compile(r"分区|期刊|影响因子|sci|ssci|cscd|核心期刊|jcr")


def other_schools_hit(core: str) -> str:
    """问句点了别的学校（库里没有那个学校的资料）。"""
    low = core.lower()
    if any(k in low for k in CUC):
        return ""
    if "中科院" in low and SCHOOL_HOMONYM.search(low):
        core = low.replace("中科院", "")
        low = core or low
    for s in OTHER_SCHOOLS:
        if s in low:
            return s
    return ""


def out_of_scope_hit(core: str, raw: str = "") -> str:
    """职责边界之外的话题（天气 / 股票 / 写代码 / 减肥 / 驾校……）。

    强信号词（天气、股票、驾照、火锅店……）出现就判越界；弱信号词（游戏、电影……）
    只有在整句里没有任何校园服务实词时才算越界——「室友打游戏到半夜，我能换宿舍吗」
    是宿舍问题，不是游戏问题。

    交通分两档：校车 / 班车出现即判库外（库里没有时刻表）；
    地铁 / 公交 / 机场 必须**同时**在问走法或线路（`TRANSIT_ROUTE`）才算——
    「学生坐地铁有优惠的卡去哪领」问的是卡，不是路线，落到「电子学生交通卡申领」。
    路线判断要走**原问句**：`query_text`/`strip_how` 会把「怎么」拆掉，
    「北京地铁怎么坐」剥完只剩「北京地铁坐」，光看剥过的串就认不出是在问路线。
    """
    low = core.lower()
    if any(k in low for k in OUT_OF_SCOPE_OK):
        return ""
    for w in OUT_OF_SCOPE_STRONG:
        if w in low:
            return w
    for w in OUT_OF_SCOPE_TRANSIT_SOLID:
        if w in low:
            return w
    if any(k in low for k in CAMPUS_TERMS):
        return ""
    for w in OUT_OF_SCOPE_WEAK:
        if w in low:
            return w
    if any(k in low for k in CAMPUS_SOFT):
        return ""
    if TRANSIT_ROUTE.search(core) or (raw and TRANSIT_ROUTE.search(raw)):
        for w in OUT_OF_SCOPE_TRANSIT:
            if w in low:
                return w
    return ""


def _sigmoid(z: float) -> float:
    import math
    return 1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, z))))


def context_features(ix: Any, cache: Any, ctx: Any, top_row: Any,
                     probs: Sequence[float]) -> list[float]:
    """「这条问题库里有没有」的字符级特征（不依赖候选）。"""
    core = ctx.core
    n = max(len(core), 1)
    covered = [False] * len(core)
    oov = total = 0
    for i in range(len(core) - 1):
        total += 1
        ch = core[i:i + 2]
        if ("G:" + ch) in ix.vocab:
            covered[i] = covered[i + 1] = True
        elif _is_cjk(core[i]) or _is_cjk(core[i + 1]):
            oov += 1
    if len(core) == 1:
        covered[0] = ("G:" + core) in ix.vocab
    run = longest = 0
    for c, ok in zip(core, covered):
        run = 0 if ok else run + 1
        if _is_cjk(c):
            longest = max(longest, run)
        else:
            run = 0
    oov_ratio = oov / max(total, 1)
    ps = list(probs) + [0.0, 0.0]
    row = top_row if top_row is not None and len(top_row) else [0.0] * 45
    return [
        float(ps[0]), float(ps[0] - ps[1]), float(row[16]),
        max(float(row[10]), float(row[13]), float(row[14])), float(row[34]),
        1.0 - oov_ratio, oov_ratio, longest / n,
        float(len([p for p in probs if p >= 0.5])), float(__import__("math").log1p(n)),
        float(row[24]), float(row[18]),
    ]


def _is_cjk(ch: str) -> bool:
    return "\u4e00" <= ch <= "\u9fff"


_DEFAULT_POLICY: dict[str, float] = {
    # 阈值只做兜底：主要靠前几档规则。扫描（_eval/gate_sweep.py）显示
    # min_prob 从 0.05 抬到 0.16，正例误拒从 1.3% 涨到 35%，而负例拒答只多 1.5 个点。
    "min_cov_short": 0.03,        # 冠军候选至少覆盖问句这么多字面
    "min_prob_weak_match": 0.15,  # 字面覆盖为零时，排序器要有这么多把握才判「没有条目」
    "min_cov_corpus": 0.45,       # 问句里至少这么多字在语料里出现过
    "max_oov_run": 3,             # 连续几个 CJK 字语料里没有 -> 判为库外说法
    "min_prob": 0.05,
}


class Gate:
    __slots__ = ("policy", "stats")

    def __init__(self, policy: dict[str, float] | None = None,
                 stats: dict[str, Any] | None = None):
        # 权重文件里存的 policy 可能比代码旧（新增的档位没有），所以是「默认值 + 覆盖」
        self.policy = dict(_DEFAULT_POLICY)
        if policy:
            self.policy.update(policy)
        self.stats = stats or {}

    @classmethod
    def default(cls) -> "Gate":
        return cls()

    # ------------------------------------------------------------------ decide

    def decide(self, engine: Any, ctx: Any, hits: Sequence[Any], probs: Sequence[float],
               cand_list: Sequence[int], x: Any) -> Decision:
        if not hits:
            return Decision(True, "没有任何候选条目", 0.0, "empty")
        idx = None
        for i, di in enumerate(cand_list):
            if engine.ix.docs[di].id == hits[0].id:
                idx = i
                break
        top_row = x[idx] if (idx is not None and x is not None and len(x)) else None
        f = context_features(engine.ix, engine.cache, ctx, top_row, probs)
        core = ctx.core
        raw = getattr(ctx, "raw", "") or ""

        # 1) 别的学校**优先于其他一切规则**：问别校的事，哪怕句子里有「宿舍 / 校园卡」
        #    也是别校的，不能拿中传的条目去答。
        school = other_schools_hit(core)
        if school:
            return Decision(True, f"资料库里只有中国传媒大学的内容，没有「{school}」的资料。",
                            1.0, "other_school")
        # 2) 个人数据
        who = personal_hit(core, raw)
        if who:
            return Decision(True, "知识库不保存个人数据（余额 / 成绩 / 课表这类），"
                                  "这类查询要去学校官方系统，我这里查不到。", 1.0, "personal")
        # 3) 职责边界 / 库外领域
        word = out_of_scope_hit(core, raw)
        if word:
            return Decision(True, f"「{word}」不属于校园办事的范围，资料库里没有相关内容。",
                            1.0, "out_of_scope")
        # 3b) 校情统计 / 学校简介：办事库里没有「几个学院 / 毕业率 / 校训 / 邮编」这类
        #     学校层面的名片与数字，条目全是「怎么办」。
        fact = SCHOOL_FACT.search(core)
        if fact:
            return Decision(True, f"资料库是办事指南，没有「{fact.group(0)}」这类校情统计"
                                  "与学校简介信息。", 1.0, "school_fact")
        # 3c) 问面—字段核对：问句要的「面」（电话 / 联系人）在冠军条目里连字面都没有，
        #     就是「库里有相关条目、但没有这一条」。只用**库里确实不存的面**，避免
        #     拿「在哪 / 几点」这类几乎每篇都提的面去误伤。
        miss = facet_missing(core, hits[0])
        if miss:
            return Decision(True, f"资料库里没有这条的{_FACET_LABEL[miss]}信息"
                                  f"（只有相关事项的办事流程，没有具体{_FACET_LABEL[miss]}）。",
                            0.75, "facet_missing")
        # 4) 语料覆盖兜底：**单个**没见过的词不构成拒答理由（「校医院能拔智齿吗」里的
        #    「智齿」不在语料里，但整句是校内问题，而且排序器很确定）。要整句覆盖低
        #    且排序器没把握，或连续 5 个字都没见过，才判库外。
        if f[5] < self.policy["min_cov_corpus"] and f[0] < 0.5:
            return Decision(True, "问句里的说法在资料库里找不到对应概念。", float(f[5]), "off_domain")
        if f[7] * max(len(core), 1) >= 5:
            return Decision(True, "问句里的说法在资料库里找不到对应概念。", float(f[5]), "off_domain")
        # 5) 太笼统：主题串很短、又没有字面命中、排序器也没把握 -> 先反问。
        #    但「一教在哪」这种短句是精确的问位置，不算笼统。
        if (len(core) <= 5 and f[0] < 0.30 and f[3] < 0.5
                and not re.search(r"在哪|哪里|哪儿|位置|几号楼|怎么走|几点|多少|能不能", core)):
            return Decision(True, "这句还太笼统，说说具体是哪一件事？", float(f[0]), "vague")
        # 6) 没有足够接近的条目。
        #    带一个概率下限：字面覆盖为零但排序器明显有把握（≥0.15）时不能拒——
        #    「心情不好想找人聊聊去哪」一个字都不重合，冠军确实是「心理咨询」，属可答。
        if (f[2] < self.policy["min_cov_short"] and f[3] < 0.5 and f[4] < 1
                and f[0] < self.policy["min_prob_weak_match"]):
            return Decision(True, "资料库里没有对应条目。", float(f[2]), "no_match")
        if f[0] < self.policy["min_prob"]:
            return Decision(True, "资料库里没有足够接近的条目。", float(f[0]), "no_match")
        return Decision(False, "", float(f[0]), "ok")

    # ---------------------------------------------------------------------- io

    def save(self, path: Path) -> None:
        Path(path).write_text(json.dumps({"policy": self.policy, "stats": self.stats},
                                         ensure_ascii=False, indent=1), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "Gate":
        d = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(policy=d.get("policy") or None, stats=d.get("stats") or {})
