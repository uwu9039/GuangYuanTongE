/* ============================================================
   广院通 · 校园经验 / web-next/social.js
   独立页：不加载 app.js，也不加载 field.js。
   - 左栏：问答（POST /api/social/ask）
   - 右栏：收录的帖子（GET /api/social/posts）
   - 手机（≤840px）：右栏变抽屉；引用角标点了先开抽屉再定位
   - 外部文字（标题/摘要/作者/结论）一律 textContent，永远不当 HTML 用
   ============================================================ */
(function () {
  "use strict";

  var POSTS_API = "/api/social/posts";
  var ASK_API = "/api/social/ask";
  var MOBILE_Q = "(max-width: 840px)";
  var HISTORY_TURNS = 3;
  var CONTEXT_POSTS = 5;

  /* 主题顺序固定：先这五个，后端多出来的按原顺序追加 */
  var TOPIC_ORDER = ["北苑食堂", "星光", "中传图书馆（大阅城）", "中蓝", "梆子井", "校园日常"];
  var PLATFORM_NAME = { bilibili: "B站", weibo: "微博", xiaohongshu: "小红书", douyin: "抖音", zhihu: "知乎" };

  /* 修正补丁：首页先放 3 条；其余问法放在「更多问法」里按主题展开。 */
  var EXAMPLES = [
    "有人分享过在北苑食堂吃了什么吗？",
    "大家去中传星光都分享过什么？",
    "去大阅城自习的同学，怎么描述馆里的环境？"
  ];
  var FOLLOWUPS = ["那哪篇提过排队？", "还有别的帖子提到过吗？", "这些说法是哪一年的？"];
  var MORE_QUESTIONS = [
    { topic: "北苑食堂", items: ["有人分享过在北苑食堂吃了什么吗？", "北苑食堂有哪些被反复推荐的窗口？"] },
    { topic: "星光", items: ["大家去中传星光都分享过什么？", "有人写过在星光超市买东西的体验吗？"] },
    { topic: "中传图书馆（大阅城）", items: [
      "去大阅城自习的同学，怎么描述馆里的环境？",
      "有没有帖子具体提到自己坐在哪一层？",
      "有人分享过在中传图书馆找座位的经验吗？",
      "第一次去大阅城，同学们提到了哪些值得留意的地方？"
    ] },
    { topic: "中蓝", items: ["住过中蓝的人提到过哪些生活细节？", "中蓝的宿舍生活，住过的人提到过哪些细节？"] },
    { topic: "梆子井", items: ["梆子井的宿舍和周边，住过的人怎么描述？", "住在梆子井的人，怎么描述周边的生活？"] }
  ];
  var TOPIC_PROMPT = {
    "北苑食堂": "有人分享过在北苑食堂吃了什么吗？",
    "星光": "大家去中传星光都分享过什么？",
    "中传图书馆（大阅城）": "去大阅城自习的同学，怎么描述馆里的环境？",
    "中蓝": "住过中蓝的人提到过哪些生活细节？",
    "梆子井": "梆子井的宿舍和周边，住过的人怎么描述？",
    "校园日常": "中传校园日常是什么样的？"
  };

  var el = {};
  var state = {
    allPosts: [],
    posts: [],
    allTotal: 0,
    collectedAt: "",
    platforms: {},
    topics: [],
    topic: "",
    panelMode: "browse",     // browse | answer
    panelTurn: null,
    drawerOpen: false,
    loading: false,
    loadFailed: false,
    loadedOnce: false,
    turns: [],
    lastQuestion: "",
    contextPostIds: [],
    feedbackLoader: null,
    hitTimers: new WeakMap()
  };

  /* ---------------- 小工具 ---------------- */

  function byId(id) { return document.getElementById(id); }

  function mk(tag, cls, text) {
    var n = document.createElement(tag);
    if (cls) n.className = cls;
    if (text !== undefined && text !== null) n.textContent = String(text);
    return n;
  }

  function safeUrl(value) {
    var url = String(value || "").trim();
    return /^https?:\/\//i.test(url) ? url : "";
  }

  function platformName(platform) {
    var key = String(platform || "").toLowerCase();
    if (PLATFORM_NAME[key]) return PLATFORM_NAME[key];
    if (!key) return "来源未知";
    return String(platform);
  }

  function dateOnly(value) {
    var m = /^(\d{4}-\d{2}-\d{2})/.exec(String(value || "").trim());
    return m ? m[1] : "";
  }

  function yearOf(value) {
    var d = dateOnly(value);
    return d ? Number(d.slice(0, 4)) : 0;
  }

  function pad2(n) {
    var v = String(n);
    return v.length < 2 ? "0" + v : v;
  }

  function isMobile() {
    return !!(window.matchMedia && window.matchMedia(MOBILE_Q).matches);
  }

  function say(text) {
    if (el.status) el.status.textContent = text || "";
  }

  function findTurn(key) {
    for (var i = 0; i < state.turns.length; i += 1) {
      if (state.turns[i].key === key) return state.turns[i];
    }
    return null;
  }

  function turnNo(turn) {
    var i = state.turns.indexOf(turn);
    return i < 0 ? state.turns.length : i + 1;
  }

  async function fetchJson(url, options) {
    var res;
    try {
      res = await fetch(url, options);
    } catch (error) {
      throw new Error("连不上本地服务，请确认 app.py 已经在当前端口运行");
    }
    var data = null;
    try {
      data = await res.json();
    } catch (error) {
      data = null;
    }
    if (!res.ok) {
      var msg = (data && (data.error || data.detail || data.message)) || ("请求失败（" + res.status + "）");
      throw new Error(msg);
    }
    if (!data) throw new Error("接口没有返回内容");
    return data;
  }

  /* ---------------- 初始状态 ---------------- */

  function renderEmpty() {
    el.thread.textContent = "";
    var wrap = mk("div", "empty");
    wrap.appendChild(mk("p", "ek", "CAMPUS NOTES / 公开帖子"));

    var h1 = mk("h1");
    h1.appendChild(document.createTextNode("食堂、超市、日常，"));
    h1.appendChild(document.createElement("br"));
    h1.appendChild(mk("em", null, "都有人写过"));
    wrap.appendChild(h1);

    wrap.appendChild(mk("p", null,
      "这里收的是同学发在小红书、B 站的公开帖子。问一句校园里的事，回答会拆成几条能核对的短句，" +
      "每条后面标着出自第几篇；右边那一栏就是这些帖子，可以点开原文。"
    ));

    var chips = mk("div", "chips");
    EXAMPLES.forEach(function (text) {
      var b = mk("button", null, text);
      b.type = "button";
      b.addEventListener("click", function () { submitAsk(text); });
      chips.appendChild(b);
    });
    wrap.appendChild(chips);

    var more = document.createElement("details");
    more.className = "moreq";
    var sum = document.createElement("summary");
    sum.textContent = "更多问法（按主题展开）";
    more.appendChild(sum);
    MORE_QUESTIONS.forEach(function (group) {
      var block = mk("div", "moreq-group");
      block.appendChild(mk("div", "moreq-topic", group.topic));
      var row = mk("div", "moreq-items");
      group.items.forEach(function (text) {
        var b = mk("button", null, text);
        b.type = "button";
        b.addEventListener("click", function () {
          submitAsk(text, { topic_context: group.topic });
        });
        row.appendChild(b);
      });
      block.appendChild(row);
      more.appendChild(block);
    });
    wrap.appendChild(more);

    var exit = mk("p", "eexit");
    exit.appendChild(mk("b", null, "不太适合这里的问题："));
    exit.appendChild(document.createTextNode(
      "补办校园卡、转专业这类正式流程请去"
    ));
    var link = mk("a", null, "办事向导");
    link.href = "/field";
    exit.appendChild(link);
    exit.appendChild(document.createTextNode(
      "查原文和材料。帖子只代表写它的人当时的体验，不是学校口径。"
    ));
    wrap.appendChild(exit);

    el.thread.appendChild(wrap);
  }

  function resetConversation() {
    state.turns = [];
    state.lastQuestion = "";
    state.contextPostIds = [];
    state.loading = false;
    state.panelMode = "browse";
    state.panelTurn = null;
    state.topic = "";
    state.posts = state.allPosts.slice();
    el.presets.textContent = "";
    el.presets.hidden = true;
    el.qhint.textContent = "Enter 发送 · Shift+Enter 换行 · 每条结论都能点开出处";
    renderEmpty();
    renderPanel();
    updateSend();
    el.q.focus({ preventScroll: true });
    say("已重新开始");
  }

  /* ---------------- 提问 ---------------- */

  function setLoading(turn) {
    turn.slot.textContent = "";
    var box = mk("div", "loading");
    box.setAttribute("role", "status");
    var dots = mk("span", "dots");
    dots.setAttribute("aria-hidden", "true");
    for (var i = 0; i < 3; i += 1) dots.appendChild(document.createElement("i"));
    box.appendChild(dots);
    box.appendChild(mk("span", null, "正在查收录的帖子…"));
    turn.slot.appendChild(box);
  }

  async function submitAsk(raw, opts) {
    opts = opts || {};
    var text = String(raw || "").trim();
    if (!text || state.loading) return;

    var turn = opts.reuse || null;
    if (turn) {
      turn.slot.textContent = "";
    } else {
      turn = { question: text, key: "t" + (state.turns.length + 1), answer: null };
      var wrap = mk("div", "ex social-turn");
      var ask = mk("div", "ask");
      ask.appendChild(mk("div", "bubble", text));
      var slot = mk("div");
      wrap.appendChild(ask);
      wrap.appendChild(slot);
      turn.wrap = wrap;
      turn.slot = slot;
      el.thread.appendChild(wrap);
      state.turns.push(turn);
      el.presets.hidden = true;
    }

    var emptyState = el.thread.querySelector(".empty");
    if (emptyState) emptyState.remove();

    /* 提问时的上下文：上一轮的引用帖子 + 上一轮的问题；历史只带最近 3 轮 user 文本 */
    var history = [];
    state.turns.forEach(function (t) {
      if (t === turn) return;
      if (!t.question || !t.answer) return;
      history.push({ role: "user", content: t.question });
    });
    history = history.slice(-HISTORY_TURNS);

    var body = {
      text: text,
      history: history,
      context_post_ids: state.contextPostIds.slice(0, CONTEXT_POSTS),
      context_topic: opts.topic_context || state.lastQuestion || ""
    };
    if (state.topic) body.topic = state.topic;

    state.loading = true;
    updateSend();
    setLoading(turn);
    scrollThread();
    say("正在查收录的帖子");

    try {
      var data = await fetchJson(ASK_API, {
        method: "POST",
        headers: { "Content-Type": "application/json", Accept: "application/json" },
        body: JSON.stringify(body)
      });
      turn.answer = data;
      turn.sources = Array.isArray(data.sources) ? data.sources : [];
      turn.cited = [];
      (Array.isArray(data.claims) ? data.claims : []).forEach(function (claim) {
        (Array.isArray(claim.post_ids) ? claim.post_ids : []).forEach(function (pid) {
          var key = String(pid);
          var known = turn.sources.some(function (p) { return String(p.post_id) === key; });
          if (known && turn.cited.indexOf(key) < 0) turn.cited.push(key);
        });
      });

      renderAnswer(turn);
      state.lastQuestion = text;
      state.contextPostIds = (turn.cited.length ? turn.cited : turn.sources.map(function (p) {
        return String(p.post_id);
      })).slice(0, CONTEXT_POSTS);

      /* 右栏跟着这一轮走；转办事向导时不显示来源 */
      if (!isRedirect(data) && turn.sources.length) setPanelAnswer(turn);

      say(answerVoice(data));
      renderFollowups();
    } catch (error) {
      renderError(turn, error);
      say("这次没查到，可以重试");
    } finally {
      state.loading = false;
      updateSend();
      scrollThread();
    }
  }

  function answerVoice(data) {
    if (isRedirect(data)) return "这类问题转去办事向导";
    var claims = Array.isArray(data.claims) ? data.claims : [];
    if (!claims.length) return "这一轮没有收录到相关内容";
    var sources = Array.isArray(data.sources) ? data.sources : [];
    return "回答好了：" + claims.length + " 条可核对的说法，引用 " + sources.length + " 篇帖子";
  }

  function isRedirect(data) {
    return !!data && (data.status === "official_redirect" || data.mode === "redirect");
  }

  /* ---------------- 回答卡 ---------------- */

  function statusTag(status, mode) {
    var map = {
      answered: ["tag", "已回答"],
      partial: ["tag warn", "只覆盖一部分"],
      insufficient: ["tag warn", "资料不足"],
      official_redirect: ["tag plain", "转办事向导"]
    };
    var pair = map[status] || (mode === "excerpt" ? ["tag warn", "资料摘录"] : ["tag plain", "结果"]);
    return mk("span", pair[0], pair[1]);
  }

  function citeButton(index, pid, turn) {
    var b = mk("button", "cite", "[" + index + "]");
    b.type = "button";
    b.setAttribute("data-cite", pid);
    b.setAttribute("data-turn", turn.key);
    var title = sourceTitle(turn, pid);
    b.setAttribute("aria-label", "跳到来源 [" + index + "]" + (title ? "：" + title : ""));
    b.title = title ? "来源 [" + index + "] · " + title : "来源 [" + index + "]";
    return b;
  }

  function sourceTitle(turn, pid) {
    var list = turn.sources || [];
    for (var i = 0; i < list.length; i += 1) {
      if (String(list[i].post_id) === String(pid)) return String(list[i].title || "");
    }
    return "";
  }

  function renderAnswer(turn) {
    var a = turn.answer || {};
    var mode = String(a.mode || "");
    var status = String(a.status || "");
    var claims = Array.isArray(a.claims) ? a.claims : [];
    var sources = turn.sources || [];

    if (isRedirect(a)) {
      turn.slot.textContent = "";
      turn.slot.appendChild(redirectCard(a));
      mountFeedback(turn.slot, a);
      return;
    }

    var options = a.ambiguous && Array.isArray(a.ambiguous.options) ? a.ambiguous.options : [];
    if (mode === "ambiguous" && options.length) {
      turn.slot.textContent = "";
      turn.slot.appendChild(ambiguousCard(a, options));
      mountFeedback(turn.slot, a);
      return;
    }

    var srcIndex = {};
    sources.forEach(function (post, i) {
      if (post && post.post_id !== undefined) srcIndex[String(post.post_id)] = i + 1;
    });

    var card = mk("div", "card social-card");
    card.setAttribute("data-mode", mode || "unknown");
    card.setAttribute("data-status", status || "unknown");

    var head = mk("div", "cardhead");
    head.appendChild(statusTag(status, mode));
    if (mode === "excerpt") head.appendChild(mk("span", "tag warn", "资料摘录 · 未经模型归纳"));
    head.appendChild(mk("span", "norm", "公开帖子"));
    head.appendChild(mk("span", "conf", sources.length ? "来源 " + sources.length + " 篇" : "没有引用帖子"));
    card.appendChild(head);

    var body = mk("div", "cardbody");

    if (mode === "excerpt") {
      var noteBox = mk("p", "social-excerptNote");
      noteBox.appendChild(mk("b", null, "资料摘录 · 未经模型归纳"));
      noteBox.appendChild(document.createTextNode("　下面是原帖里和问题相关的句子，按原文抄录，没有经过模型改写。"));
      body.appendChild(noteBox);
    }

    if (claims.length) {
      var ul = mk("ul", "claims");
      claims.forEach(function (claim, i) {
        var li = mk("li", "claim");
        li.appendChild(mk("span", "cn", pad2(i + 1)));
        var ct = mk("p", "ct");
        ct.appendChild(document.createTextNode(String(claim.text || "")));
        (Array.isArray(claim.post_ids) ? claim.post_ids : []).forEach(function (pid) {
          var index = srcIndex[String(pid)];
          if (!index) return;                       // 找不到的引用不渲染角标
          ct.appendChild(citeButton(index, String(pid), turn));
        });
        li.appendChild(ct);
        ul.appendChild(li);
      });
      body.appendChild(ul);
    } else {
      body.appendChild(noClaimBlock(a, status, mode));
    }

    if (claims.length && a.note) {
      body.appendChild(mk("p", "social-note", String(a.note)));
    }

    var missing = Array.isArray(a.missing) ? a.missing.filter(Boolean).map(String) : [];
    if (missing.length && claims.length) {
      var miss = mk("p", "social-missing");
      miss.appendChild(mk("b", null, "还缺："));
      miss.appendChild(document.createTextNode(missing.join("；")));
      body.appendChild(miss);
    }

    card.appendChild(body);

    var foot = mk("div", "cardfoot");
    if (sources.length) {
      var look = mk("button", "cf key", "看这轮出处");
      look.type = "button";
      look.appendChild(mk("span", "n", String(sources.length)));
      look.addEventListener("click", function () {
        setPanelAnswer(turn);
        if (isMobile()) openDrawer({ focusFirst: true });
        else el.inspector.querySelector(".pcard") && scrollPanelTop();
      });
      foot.appendChild(look);
    }
    foot.appendChild(mk("span", "cfstamp", "SOCIAL-" + pad2(turnNo(turn))));
    card.appendChild(foot);

    turn.slot.textContent = "";
    turn.slot.appendChild(card);
    mountFeedback(card, a);
  }

  function noClaimBlock(a, status, mode) {
    var box = mk("div", "social-emptyans");
    var emptyish = (mode === "empty" || status === "insufficient" || !(a.claims || []).length);
    if (emptyish) {
      box.appendChild(mk("p", "social-emptyh", "收录的帖子里暂时没有这方面的信息"));
    }
    var note = String(a.note || "").trim();
    if (note && note.indexOf("暂时没有") < 0) {
      box.appendChild(mk("p", "social-note plain", note));
    }
    var sugg = topicSuggestions();
    if (sugg.length) {
      box.appendChild(mk("p", "social-sub", "换一个收得多的主题试试："));
      var row = mk("div", "social-sugg");
      sugg.forEach(function (topic) {
        var b = mk("button", "opt", topic);
        b.type = "button";
        b.addEventListener("click", function () {
          submitAsk(TOPIC_PROMPT[topic] || (topic + "有哪些帖子提到？"));
        });
        row.appendChild(b);
      });
      box.appendChild(row);
    }
    return box;
  }

  function topicSuggestions() {
    var names = topicList().map(function (t) { return t.name; });
    return names.filter(function (name) { return name !== state.topic; }).slice(0, 4);
  }

  function redirectCard(a) {
    var card = mk("div", "notice social-redirect");
    card.setAttribute("data-mode", "redirect");
    card.appendChild(mk("h3", null, "这类问题按学校官方口径办"));
    card.appendChild(mk("p", null, String(a.note || "正式流程请到办事向导查原文和材料清单。")));
    var redirect = a.redirect || {};
    var url = String(redirect.url || "/field").trim();
    if (!/^(\/|https?:\/\/)/i.test(url)) url = "/field";
    var go = mk("a", "gobtn", String(redirect.label || "去办事向导"));
    go.href = url;
    go.appendChild(mk("span", null, "↗"));
    card.appendChild(go);
    return card;
  }

  function ambiguousCard(a, options) {
    var card = mk("div", "notice social-ambiguous");
    card.appendChild(mk("h3", null, "问的是哪一个？"));
    card.appendChild(mk("p", null, String(a.note || "这个说法在校内指两个地方。")));
    var row = mk("div", "opts");
    options.forEach(function (option) {
      var value = String((option && option.value) || "").trim();
      if (!value) return;
      var b = mk("button", "opt", String(option.label || value));
      b.type = "button";
      b.addEventListener("click", function () { submitAsk(value); });
      row.appendChild(b);
    });
    card.appendChild(row);
    return card;
  }

  function renderError(turn, error) {
    turn.answer = null;
    turn.slot.textContent = "";
    var card = mk("div", "notice social-error");
    card.setAttribute("role", "alert");
    card.appendChild(mk("h3", null, "这次没查到"));
    card.appendChild(mk("p", null, (error && error.message) || "本地服务没有回话。"));
    var row = mk("div", "opts");
    var retry = mk("button", "opt", "重试");
    retry.type = "button";
    retry.addEventListener("click", function () { submitAsk(turn.question, { reuse: turn }); });
    row.appendChild(retry);
    card.appendChild(row);
    turn.slot.appendChild(card);
  }

  /* 反馈：主 agent 提供 window.GYTFeedback.mount。没有 answer_id 就不留节点。 */
  function mountFeedback(host, answer) {
    var answerId = answer && answer.answer_id;
    if (!answerId) {
      if (answer && answer.feedback_unavailable) {
        host.appendChild(mk("p", "answer-feedback-off", "反馈暂不可用"));
      }
      return null;
    }
    var box = mk("div", "social-feedback");
    box.setAttribute("data-answer-id", String(answerId));
    host.appendChild(box);
    scheduleFeedback(box);
    return box;
  }

  function tryMountFeedback(box) {
    if (!box || box.childElementCount) return true;
    var fb = window.GYTFeedback;
    if (!fb || typeof fb.mount !== "function") return false;
    var answerId = box.getAttribute("data-answer-id");
    try {
      fb.mount(box, answerId, { channel: "social" });
      return true;
    } catch (error) {
      /* 主 agent 的 mount 也可能只收一个元素（id 就在 data-answer-id 上） */
      try {
        fb.mount(box);
        return true;
      } catch (error2) {
        return false;
      }
    }
  }

  function loadFeedbackScript() {
    /* social.html 已经直接引了 feedback.js：组件在就直接用，不再插第二个脚本 */
    if (window.GYTFeedback) return Promise.resolve(true);
    if (state.feedbackLoader) return state.feedbackLoader;
    state.feedbackLoader = fetch("feedback.js", { method: "HEAD" }).then(function (res) {
      if (!res.ok) return false;
      if (window.GYTFeedback) return true;
      return new Promise(function (resolve) {
        var s = document.createElement("script");
        s.src = "feedback.js";
        s.defer = true;
        s.onload = function () { resolve(true); };
        s.onerror = function () { resolve(false); };
        document.head.appendChild(s);
      });
    }).catch(function () { return false; });
    return state.feedbackLoader;
  }

  function scheduleFeedback(box) {
    if (tryMountFeedback(box)) return;
    loadFeedbackScript().then(function () {
      window.setTimeout(function () { tryMountFeedback(box); }, 0);
    });
    var tries = 0;
    var timer = window.setInterval(function () {
      tries += 1;
      if (tryMountFeedback(box) || tries > 20 || !document.body.contains(box)) window.clearInterval(timer);
    }, 500);
  }

  /* ---------------- 输入区 ---------------- */

  function resizeQ() {
    el.q.style.height = "auto";
    var next = Math.min(150, Math.max(46, el.q.scrollHeight));
    el.q.style.height = next + "px";
  }

  function updateSend() {
    el.send.disabled = !!state.loading;
  }

  function scrollThread() {
    el.scroller.scrollTop = el.scroller.scrollHeight;
  }

  function renderFollowups() {
    if (el.presets.childElementCount) return;
    FOLLOWUPS.forEach(function (text) {
      var b = mk("button", null, text);
      b.type = "button";
      b.addEventListener("click", function () { submitAsk(text); });
      el.presets.appendChild(b);
    });
    el.presets.hidden = false;
    el.qhint.textContent = "可以接着追问 · Enter 发送 · Shift+Enter 换行";
  }

  /* ---------------- 右栏 ---------------- */

  function topicList() {
    var seen = {};
    var list = [];
    TOPIC_ORDER.forEach(function (name) {
      var hit = state.topics.filter(function (t) { return String(t.name) === name; })[0];
      if (hit) { seen[name] = true; list.push({ name: name, count: Number(hit.count) || 0 }); }
    });
    state.topics.forEach(function (t) {
      var name = String(t.name || "");
      if (!name || seen[name]) return;
      seen[name] = true;
      list.push({ name: name, count: Number(t.count) || 0 });
    });
    return list;
  }

  function platformLabel() {
    var keys = Object.keys(state.platforms || {});
    if (!keys.length) return "B站";
    return keys.map(platformName).join("、");
  }

  function panelList() {
    if (state.panelMode === "answer" && state.panelTurn) return state.panelTurn.sources || [];
    return state.posts;
  }

  function renderPanel() {
    var list = panelList();
    if (!state.loadedOnce && !state.loadFailed) {
      el.jnsub.textContent = "正在读取收录的帖子…";
    } else {
      var bits = ["共 " + list.length + " 条" + (state.panelMode === "answer" ? "（本轮引用）" : "")];
      bits.push("采集于 " + (dateOnly(state.collectedAt) || "日期未知"));
      bits.push("来源 " + platformLabel());
      el.jnsub.textContent = bits.join(" · ");
    }

    el.srcbar.hidden = state.panelMode !== "answer";
    if (state.panelMode === "answer" && state.panelTurn) {
      el.srcbarlabel.textContent = "正在看第 " + turnNo(state.panelTurn) + " 轮的 " + list.length + " 条出处";
    }

    renderFilters();
    renderList(list);

    el.inscount.textContent = String(state.allTotal || list.length);
    el.insbtn.setAttribute("aria-label", "打开资料栏，查看收录的 " + (state.allTotal || list.length) + " 条帖子");
    el.insstamp.textContent = "NO. " + pad2(list.length);
    el.legend.textContent = state.panelMode === "answer" ? "这一轮的出处，点标题可看原文" : "标题与摘要按原帖照录";
  }

  function renderFilters() {
    el.filters.textContent = "";
    var items = [{ name: "全部", count: state.allTotal }].concat(topicList());
    items.forEach(function (item) {
      var on = state.panelMode === "browse" &&
        (item.name === "全部" ? !state.topic : state.topic === item.name);
      var b = mk("button", "fchip");
      b.type = "button";
      b.setAttribute("aria-pressed", on ? "true" : "false");
      b.appendChild(document.createTextNode(item.name));
      if (item.count) b.appendChild(mk("span", "n", String(item.count)));
      b.addEventListener("click", function () { chooseTopic(item.name); });
      el.filters.appendChild(b);
    });
  }

  function chooseTopic(name) {
    state.topic = name === "全部" ? "" : name;
    state.panelMode = "browse";
    state.panelTurn = null;
    state.posts = state.topic
      ? state.allPosts.filter(function (post) {
        return (Array.isArray(post.topics) ? post.topics : []).some(function (t) { return String(t) === state.topic; });
      })
      : state.allPosts.slice();
    renderPanel();
    loadPosts(state.topic, { quiet: true });
  }

  function renderList(list) {
    el.insbody.textContent = "";
    el.insbody.scrollTop = 0;
    if (!list.length) {
      el.insbody.appendChild(listEmpty());
      return;
    }
    var wrap = mk("div", "pclist");
    list.forEach(function (post, i) {
      var card = postCard(post, i + 1);
      /* 正在看某一轮的出处时，这几条就是被引用的那几条，左侧加一道细标 */
      if (state.panelMode === "answer") card.classList.add("is-cited");
      wrap.appendChild(card);
    });
    el.insbody.appendChild(wrap);
  }

  function listEmpty() {
    var box = mk("div", "social-insempty");
    if (state.loadFailed) {
      box.appendChild(mk("h3", null, "这次没读到收录的帖子"));
      box.appendChild(mk("p", null, "接口没有回话。确认后端已经起来，或者稍后再试。"));
      var row = mk("div", "opts");
      var retry = mk("button", "opt", "重试");
      retry.type = "button";
      retry.addEventListener("click", function () { loadPosts(state.topic); });
      row.appendChild(retry);
      box.appendChild(row);
      return box;
    }
    if (!state.allPosts.length && !state.loadedOnce) {
      box.appendChild(mk("h3", null, "正在读取收录的帖子…"));
      box.appendChild(mk("p", null, "第一次打开要向后端要一次列表，稍等一下。"));
      return box;
    }
    if (!state.allPosts.length) {
      box.appendChild(mk("h3", null, "收录的帖子还是空的"));
      box.appendChild(mk("p", null, "等采集跑完，这里会按主题列出帖子；现在提问会提示资料不足。"));
      return box;
    }
    box.appendChild(mk("h3", null, "这个主题下还没有帖子"));
    box.appendChild(mk("p", null, "点上面的「全部」，或者直接问别的事。"));
    return box;
  }

  function postCard(post, index) {
    var card = mk("article", "pcard");
    card.setAttribute("data-pid", String(post && post.post_id !== undefined ? post.post_id : ""));
    card.tabIndex = -1;

    var head = mk("div", "pchead");
    head.appendChild(mk("span", "pcno", "[" + index + "]"));
    head.appendChild(mk("span", "pcplat", platformName(post.platform)));
    head.appendChild(mk("span", "pcdate", dateOnly(post.published_at) || "发布日期未知"));
    card.appendChild(head);

    card.appendChild(mk("h3", "pctitle", String(post.title || "").trim() || "（原帖没有标题）"));
    card.appendChild(mk("p", "pcmeta", "作者 " + (String(post.author || "").trim() || "未署名")));

    var summary = String(post.summary || "").trim();
    if (summary) {
      card.appendChild(mk("p", "pcsum", summary));
    } else if (post.title_only) {
      card.appendChild(mk("p", "pcsum pcnone", "这条只采到标题，没有可摘录的正文。"));
    }

    var foot = mk("div", "pcfoot");
    if (post.title_only) {
      var tag = mk("span", "pctag", "仅标题");
      tag.title = "采集时只拿到标题，没有正文摘要";
      foot.appendChild(tag);
    }
    var year = yearOf(post.published_at);
    if (year && year < 2024) {
      var old = mk("span", "pctag old", year + " 年的帖");
      old.title = "发布时间较早：只能说明当时有人这样描述";
      foot.appendChild(old);
    }
    var url = safeUrl(post.url);
    if (url) {
      var link = mk("a", "pclink", "查看原帖 ↗");
      link.href = url;
      link.target = "_blank";
      link.rel = "noreferrer noopener";
      link.setAttribute("aria-label", "在新窗口打开原帖：" + (String(post.title || "").trim() || "未命名帖子"));
      foot.appendChild(link);
    } else {
      foot.appendChild(mk("span", "pcnolink", "原文链接缺失"));
    }
    card.appendChild(foot);
    return card;
  }

  async function loadPosts(topic, opts) {
    opts = opts || {};
    var wanted = topic || "";
    var url = POSTS_API + "?topic=" + encodeURIComponent(wanted) + "&limit=50";
    try {
      var data = await fetchJson(url);
      if (!data || !Array.isArray(data.posts)) throw new Error("接口没有返回帖子");
      state.loadFailed = false;
      state.collectedAt = String(data.collected_at || "");
      state.allTotal = Number(data.total) || data.posts.length;
      state.platforms = data.platforms || {};
      state.topics = Array.isArray(data.topics) ? data.topics : [];
      if (!wanted) state.allPosts = data.posts.slice();
      state.loadedOnce = true;
      if (state.topic === wanted && state.panelMode === "browse") {
        state.posts = data.posts.slice();
        renderPanel();
      } else if (wanted && state.topic === wanted) {
        /* 落在回答模式时只更新计数，不动列表 */
        renderPanel();
      }
      var total = state.allTotal || data.posts.length;
      el.statline.textContent = "收录 " + total + " 条 · 采集于 " + (dateOnly(state.collectedAt) || "日期未知");
    } catch (error) {
      state.loadFailed = true;
      el.statline.textContent = "读取收录失败 · 稍后重试";
      if (!state.allPosts.length) {
        el.jnsub.textContent = "没读到收录的帖子";
        renderList([]);
      }
      if (!opts.quiet) say("读取帖子的接口没有回话");
    }
  }

  function setPanelAnswer(turn) {
    state.panelMode = "answer";
    state.panelTurn = turn;
    renderPanel();
  }

  function setPanelBrowse() {
    state.panelMode = "browse";
    state.panelTurn = null;
    renderPanel();
  }

  /* 引用角标 → 定位来源卡 */
  function focusSource(pid, turnKey) {
    var turn = findTurn(turnKey);
    if (!turn) return;
    if (state.panelMode !== "answer" || state.panelTurn !== turn) setPanelAnswer(turn);

    var card = null;
    var cards = el.insbody.querySelectorAll(".pcard");
    for (var i = 0; i < cards.length; i += 1) {
      if (cards[i].getAttribute("data-pid") === String(pid)) { card = cards[i]; break; }
    }
    if (!card) return;

    if (isMobile()) {
      openDrawer({ focusCard: card });
    } else {
      revealSource(card);
      card.focus({ preventScroll: true });
    }
  }

  function revealSource(card) {
    if (!card) return;
    card.scrollIntoView({ block: "nearest", behavior: "auto" });
    card.classList.remove("is-hit");
    /* 强制重排，再触发一次高亮动画 */
    void card.offsetWidth;
    card.classList.add("is-hit");
    var old = state.hitTimers.get(card);
    if (old) window.clearTimeout(old);
    state.hitTimers.set(card, window.setTimeout(function () { card.classList.remove("is-hit"); }, 2600));
  }

  function scrollPanelTop() {
    el.insbody.scrollTop = 0;
  }

  /* ---------------- 手机抽屉 ---------------- */

  function openDrawer(opts) {
    opts = opts || {};
    if (!isMobile()) return;
    state.drawerOpen = true;
    state.lastFocus = document.activeElement;
    el.inspector.removeAttribute("inert");
    el.inspector.setAttribute("data-open", "true");
    el.inspector.setAttribute("role", "dialog");
    el.inspector.setAttribute("aria-modal", "true");
    el.inspector.setAttribute("aria-labelledby", "institle");
    el.insbtn.setAttribute("aria-expanded", "true");
    el.scrim.hidden = false;
    window.requestAnimationFrame(function () { el.scrim.setAttribute("data-open", "true"); });
    document.body.classList.add("drawer-open");

    if (opts.focusCard) {
      var card = opts.focusCard;
      window.setTimeout(function () {
        revealSource(card);
        card.focus({ preventScroll: true });
      }, 340);
    } else {
      window.setTimeout(function () {
        if (state.drawerOpen) el.insclose.focus({ preventScroll: true });
      }, 60);
    }
    say("资料栏已打开");
  }

  function closeDrawer() {
    if (!state.drawerOpen) return;
    state.drawerOpen = false;
    el.inspector.removeAttribute("data-open");
    el.inspector.removeAttribute("role");
    el.inspector.removeAttribute("aria-modal");
    el.inspector.removeAttribute("aria-labelledby");
    el.inspector.setAttribute("inert", "");
    el.insbtn.setAttribute("aria-expanded", "false");
    el.scrim.setAttribute("data-open", "false");
    document.body.classList.remove("drawer-open");
    window.setTimeout(function () { if (!state.drawerOpen) el.scrim.hidden = true; }, 240);
    /* 关掉抽屉，焦点回到输入区 */
    el.q.focus({ preventScroll: true });
    say("资料栏已关闭，回到输入区");
  }

  function syncViewport() {
    if (!isMobile()) {
      state.drawerOpen = false;
      el.inspector.removeAttribute("inert");
      el.inspector.removeAttribute("data-open");
      el.inspector.removeAttribute("role");
      el.inspector.removeAttribute("aria-modal");
      el.inspector.removeAttribute("aria-labelledby");
      el.insbtn.setAttribute("aria-expanded", "false");
      el.scrim.hidden = true;
      el.scrim.removeAttribute("data-open");
      document.body.classList.remove("drawer-open");
      return;
    }
    if (state.drawerOpen) el.inspector.removeAttribute("inert");
    else el.inspector.setAttribute("inert", "");
  }

  function trapFocus(ev) {
    if (!state.drawerOpen || !isMobile()) return;
    var nodes = el.inspector.querySelectorAll("a[href],button:not([disabled]),[tabindex]:not([tabindex='-1'])");
    if (!nodes.length) return;
    var first = nodes[0];
    var last = nodes[nodes.length - 1];
    if (ev.shiftKey && document.activeElement === first) {
      ev.preventDefault();
      last.focus({ preventScroll: true });
    } else if (!ev.shiftKey && document.activeElement === last) {
      ev.preventDefault();
      first.focus({ preventScroll: true });
    }
  }

  /* ---------------- 绑定 ---------------- */

  function bind() {
    el.composer.addEventListener("submit", function (ev) {
      ev.preventDefault();
      submitAsk(el.q.value);
    });

    el.q.addEventListener("input", function () {
      resizeQ();
      updateSend();
    });

    el.q.addEventListener("keydown", function (ev) {
      if (ev.key !== "Enter" || ev.shiftKey) return;
      if (ev.isComposing || ev.keyCode === 229) return;   /* 中文输入法选词时不发送 */
      ev.preventDefault();
      submitAsk(el.q.value);
    });

    el.clearbtn.addEventListener("click", resetConversation);

    el.thread.addEventListener("click", function (ev) {
      var btn = ev.target.closest ? ev.target.closest("[data-cite]") : null;
      if (!btn) return;
      ev.preventDefault();
      var prev = el.thread.querySelector(".cite[aria-current='true']");
      if (prev) prev.removeAttribute("aria-current");
      btn.setAttribute("aria-current", "true");
      focusSource(btn.getAttribute("data-cite"), btn.getAttribute("data-turn"));
    });

    el.insbtn.addEventListener("click", function () {
      if (state.drawerOpen) closeDrawer();
      else openDrawer({ focusFirst: true });
    });

    el.insclose.addEventListener("click", closeDrawer);
    el.allposts.addEventListener("click", function () {
      setPanelBrowse();
      if (isMobile()) closeDrawer();
    });

    el.scrim.addEventListener("click", closeDrawer);

    document.addEventListener("keydown", function (ev) {
      if (ev.key === "Escape" && state.drawerOpen) {
        ev.preventDefault();
        closeDrawer();
        return;
      }
      if (ev.key === "Tab") trapFocus(ev);
    });

    if (window.matchMedia) {
      var mq = window.matchMedia(MOBILE_Q);
      if (mq.addEventListener) mq.addEventListener("change", syncViewport);
      else if (mq.addListener) mq.addListener(syncViewport);
    }

    window.addEventListener("resize", function () {
      resizeQ();
      if (!isMobile() && state.drawerOpen) syncViewport();
    });
  }

  function init() {
    [
      "scroller", "thread", "composer", "q", "send", "clearbtn", "presets", "qhint",
      "statline", "status", "inspector", "jnsub", "filters", "srcbar", "srcbarlabel",
      "allposts", "insbody", "insstamp", "legend", "insclose", "insbtn", "inscount", "scrim"
    ].forEach(function (id) { el[id] = byId(id); });

    renderEmpty();
    bind();
    syncViewport();
    resizeQ();
    updateSend();
    renderPanel();
    loadPosts("");

    /* /social?q=… 直接带问题进来（和 /field 的分享链接一致）；
       ?drawer=1 供手机宽度截图/演示直接展开资料抽屉。 */
    var params = null;
    try {
      params = new URLSearchParams(window.location.search);
    } catch (error) {
      params = null;
    }
    var initial = (params && params.get("q")) || "";
    if (initial.trim()) {
      el.q.value = initial;
      resizeQ();
      submitAsk(initial);
    }
    if (params && params.get("drawer") === "1") {
      window.setTimeout(function () {
        if (isMobile()) openDrawer({ focusFirst: false });
      }, initial ? 2600 : 900);
    }
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }
}());
