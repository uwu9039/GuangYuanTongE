/* ============================================================
   广院通 · 匿名赞踩组件（feedback.js）
   ------------------------------------------------------------
   普通问答与校园经验问答共用；/field、/desk、/classic、/social
   都通过 window.GYTFeedback.mount(container, answerId, options) 挂载。

   设计约束（改代码前先看一眼）：
   - 零依赖、零计时器；页面切走不会留下任何 setTimeout / setInterval。
   - 不发送姓名、学号、问题或回答正文，只发匿名标记与 vote。
   - 点击立刻改选中态（乐观更新），写请求按回答串行排队；
     失败回退到“最后一次服务端确认的状态”，并给一行可点的“重试”。
   - 同一回答的请求用模块级锁串行（含 GET），避免多实例并发写。
   - 挂载两次互不影响：状态、队列、AbortController 全在 mount 的闭包里。
   ============================================================ */
(function (global) {
  "use strict";

  var STORAGE_KEY = "gyt_voter_id";
  /* 只认自己生成的形状：anon- 前缀 + 至少 16 个字母数字。
     历史版本若存过 UUID（anon- 后面带连字符）也继续沿用，不丢旧票。 */
  var VOTER_ID_RE = /^anon-[A-Za-z0-9-]{16,}$/;
  var MAX_ANSWER_ID = 200;

  /* localStorage 不可用时（隐私模式 / file://）退化成这个内存值，
     同一个页面生命周期内保持稳定。 */
  var memoryVoterId = null;
  /* 每个 answerId 一条 Promise 链：同一回答同一时刻只有一个在途请求。 */
  var locks = Object.create(null);

  /* ---------------- 匿名标记 ---------------- */

  function hexFromBytes(bytes) {
    var out = "";
    for (var i = 0; i < bytes.length; i++) {
      out += ("0" + bytes[i].toString(16)).slice(-2);
    }
    return out;
  }

  function makeVoterId() {
    var c = global.crypto;
    /* 首选 16 字节 = 128 bit 真随机 */
    if (c && typeof c.getRandomValues === "function") {
      try {
        var bytes = new Uint8Array(16);
        c.getRandomValues(bytes);
        return "anon-" + hexFromBytes(bytes);
      } catch (e) { /* 继续降级 */ }
    }
    /* 次选 randomUUID（v4 约 122 bit），去连字符后长度一致 */
    if (c && typeof c.randomUUID === "function") {
      try {
        return "anon-" + c.randomUUID().replace(/-/g, "");
      } catch (e) { /* 继续降级 */ }
    }
    /* 最后的降级：极老的 WebView 没有 Web Crypto。仅用于区分匿名用户。 */
    var fallback = "";
    for (var i = 0; i < 4; i++) {
      fallback += ("00000000" + Math.floor(Math.random() * 0x100000000).toString(16)).slice(-8);
    }
    return "anon-" + fallback + Date.now().toString(16);
  }

  function getVoterId() {
    if (memoryVoterId) return memoryVoterId;
    var stored = null;
    try {
      stored = global.localStorage.getItem(STORAGE_KEY);
    } catch (e) {
      stored = null;
    }
    if (stored && VOTER_ID_RE.test(stored)) {
      memoryVoterId = stored;
      return stored;
    }
    var id = makeVoterId();
    memoryVoterId = id;
    try {
      global.localStorage.setItem(STORAGE_KEY, id);
    } catch (e) {
      /* 存不下就只留内存，不影响投票 */
    }
    return id;
  }

  /* ---------------- 小工具 ---------------- */

  function normalizeAnswerId(value) {
    if (typeof value === "number" && isFinite(value)) value = String(value);
    if (typeof value !== "string") return "";
    var id = value.trim();
    if (!id || id.length > MAX_ANSWER_ID) return "";
    /* 控制字符（含换行）会让 URL 和 data-* 变形，直接判无效 */
    if (/[\u0000-\u001f\u007f]/.test(id)) return "";
    return id;
  }

  function normalizeVote(vote) {
    return vote === "up" || vote === "down" ? vote : null;
  }

  function readCounts(payload) {
    if (!payload || typeof payload.counts !== "object" || !payload.counts) return null;
    var up = Number(payload.counts.up);
    var down = Number(payload.counts.down);
    if (!isFinite(up) || !isFinite(down) || up < 0 || down < 0) return null;
    return { up: Math.floor(up), down: Math.floor(down) };
  }

  function newAbort() {
    if (typeof global.AbortController === "function") return new global.AbortController();
    return { signal: undefined, abort: function () {} };
  }

  /* 同一 key 的任务串成一条链。task 无论前一个成败都会执行；
     锁本身吞掉结果，避免未处理的 rejection。 */
  function serialized(key, task) {
    var prev = locks[key] || Promise.resolve();
    var run = prev.then(task, task);
    var tail = run.then(function () {}, function () {});
    locks[key] = tail;
    tail.then(function () {
      if (locks[key] === tail) delete locks[key];
    });
    return run;
  }

  /* ---------------- 请求 ---------------- */

  function feedbackUrl(answerId) {
    return "/api/feedback/" + encodeURIComponent(answerId);
  }

  function requestHeaders(extra) {
    var headers = {
      "Accept": "application/json",
      "X-Voter-Id": getVoterId()
    };
    if (extra) {
      for (var k in extra) {
        if (Object.prototype.hasOwnProperty.call(extra, k)) headers[k] = extra[k];
      }
    }
    return headers;
  }

  function readBody(res) {
    return res.json().then(
      function (data) { return data; },
      function () { return null; }
    );
  }

  function send(url, init) {
    return global.fetch(url, init).then(function (res) {
      if (!res || !res.ok) {
        var err = new Error("feedback_http_" + (res ? res.status : "0"));
        err.status = res ? res.status : 0;
        throw err;
      }
      return readBody(res);
    });
  }

  function getVote(answerId, signal) {
    return send(feedbackUrl(answerId), {
      method: "GET",
      headers: requestHeaders(),
      credentials: "same-origin",
      signal: signal
    });
  }

  function putVote(answerId, vote, signal) {
    return send(feedbackUrl(answerId), {
      method: "PUT",
      headers: requestHeaders({ "Content-Type": "application/json" }),
      credentials: "same-origin",
      body: JSON.stringify({ vote: vote }),
      signal: signal
    });
  }

  /* ---------------- 挂载 ---------------- */

  function mount(container, answerId, options) {
    if (!container || container.nodeType !== 1) return null;
    var id = normalizeAnswerId(answerId);
    if (!id) return null;

    var channel = options && options.channel === "social" ? "social" : "official";

    /* 同一个容器里重复挂载同一回答：先卸掉旧的，避免按钮叠成两排。
       不同容器里的同 answerId 实例互不影响。 */
    for (var i = container.children.length - 1; i >= 0; i--) {
      var old = container.children[i];
      if (
        old && old.classList && old.classList.contains("gyt-feedback") &&
        old.getAttribute("data-answer-id") === id &&
        old.getAttribute("data-channel") === channel
      ) {
        if (typeof old.destroy === "function") old.destroy();
        else container.removeChild(old);
      }
    }

    var state = {
      confirmedVote: null, /* 最后一次服务端确认的票态 */
      counts: null,        /* 最后一次服务端给的计数 */
      desired: null,       /* 界面上显示的票态（乐观值） */
      touched: false,      /* 用户是否已经点过（用于忽略迟到的 GET） */
      error: null,         /* { target }：失败时想投的那一票，供重试用 */
      destroyed: false
    };

    var queue = [];        /* 待发送的意图，按点击顺序 */
    var running = false;
    var currentAbort = null;
    var initialAbort = null;

    var root = document.createElement("div");
    root.className = "gyt-feedback";
    root.setAttribute("role", "group");
    root.setAttribute("aria-label", "这条回答有帮助吗？");
    root.setAttribute("data-answer-id", id);
    root.setAttribute("data-channel", channel);

    var actions = document.createElement("div");
    actions.className = "gyt-feedback__actions";

    function makeButton(vote, text) {
      var btn = document.createElement("button");
      btn.setAttribute("type", "button");
      btn.className = "gyt-feedback__btn";
      btn.setAttribute("data-vote", vote);
      btn.setAttribute("aria-pressed", "false");
      btn.setAttribute("aria-label", text);

      var label = document.createElement("span");
      label.className = "gyt-feedback__label";
      label.textContent = text;

      var count = document.createElement("span");
      count.className = "gyt-feedback__count";
      count.setAttribute("aria-hidden", "true");
      count.hidden = true;

      btn.appendChild(label);
      btn.appendChild(count);
      btn.addEventListener("click", function () { onVote(vote); });
      return btn;
    }

    var upBtn = makeButton("up", "有帮助");
    var downBtn = makeButton("down", "没帮助");
    var upCount = upBtn.querySelector(".gyt-feedback__count");
    var downCount = downBtn.querySelector(".gyt-feedback__count");

    actions.appendChild(upBtn);
    actions.appendChild(downBtn);
    root.appendChild(actions);

    var errorEl = null;

    function formatCount(n) {
      return n > 9999 ? "9999+" : String(n);
    }

    function applyCount(btn, el, n, label) {
      if (typeof n === "number" && isFinite(n) && n >= 0) {
        el.hidden = false;
        el.textContent = formatCount(n);
        btn.setAttribute("aria-label", label + "，" + n + " 人");
      } else {
        el.hidden = true;
        el.textContent = "";
        btn.setAttribute("aria-label", label);
      }
    }

    function renderError() {
      if (!state.error) {
        if (errorEl && errorEl.parentNode) errorEl.parentNode.removeChild(errorEl);
        return;
      }
      if (!errorEl) {
        errorEl = document.createElement("p");
        errorEl.className = "gyt-feedback__error";
        errorEl.setAttribute("role", "status");
        errorEl.setAttribute("aria-live", "polite");

        var msg = document.createElement("span");
        msg.className = "gyt-feedback__error-text";
        /* 503 也走这一句：反馈挂掉不该影响问答本身 */
        msg.textContent = "反馈没保存成功";

        var retry = document.createElement("button");
        retry.setAttribute("type", "button");
        retry.className = "gyt-feedback__retry";
        retry.textContent = "重试";
        retry.addEventListener("click", onRetry);

        errorEl.appendChild(msg);
        errorEl.appendChild(retry);
      }
      if (!errorEl.parentNode) root.appendChild(errorEl);
    }

    function render() {
      if (state.destroyed) return;
      upBtn.setAttribute("aria-pressed", state.desired === "up" ? "true" : "false");
      downBtn.setAttribute("aria-pressed", state.desired === "down" ? "true" : "false");
      var counts = state.counts;
      applyCount(upBtn, upCount, counts ? counts.up : null, "有帮助");
      applyCount(downBtn, downCount, counts ? counts.down : null, "没帮助");
      renderError();
    }

    function applyServer(payload, target) {
      var hasVote = payload && Object.prototype.hasOwnProperty.call(payload, "vote");
      state.confirmedVote = hasVote ? normalizeVote(payload.vote) : target;
      var counts = readCounts(payload);
      if (counts) state.counts = counts;
      /* 队列里还有更新的意图时，界面继续显示最新意图；
         队列空了才把乐观态校回服务端值。 */
      if (!queue.length) state.desired = state.confirmedVote;
      render();
    }

    function enqueue(target) {
      if (state.destroyed) return;
      queue.push(target);
      if (!running) drain();
    }

    function drain() {
      running = true;
      root.setAttribute("aria-busy", "true");

      function finish() {
        running = false;
        root.removeAttribute("aria-busy");
      }

      function step() {
        if (state.destroyed) return finish();

        var target;
        while (queue.length) {
          target = queue.shift();
          /* 想投的已经是确认态：不用发请求（连点回到原点时省一次） */
          if (target !== state.confirmedVote) break;
          target = undefined;
        }

        if (target === undefined) {
          state.desired = state.confirmedVote;
          render();
          return finish();
        }

        var controller = newAbort();
        currentAbort = controller;
        serialized(id, function () {
          return putVote(id, target, controller.signal);
        }).then(function (payload) {
          currentAbort = null;
          if (state.destroyed) return finish();
          applyServer(payload, target);
          step();
        }, function (err) {
          currentAbort = null;
          if (state.destroyed) return finish();
          if (err && err.name === "AbortError") return finish();
          /* 失败：丢掉后面排队的意图，回到最后一次服务端确认的状态 */
          queue.length = 0;
          state.desired = state.confirmedVote;
          state.error = { target: target };
          render();
          finish();
        });
      }

      step();
    }

    function onVote(vote) {
      if (state.destroyed) return;
      state.touched = true;
      /* 再点当前已选按钮 = 撤销 */
      var next = state.desired === vote ? null : vote;
      state.desired = next;
      state.error = null;
      render();
      enqueue(next);
    }

    function onRetry() {
      if (state.destroyed || !state.error) return;
      var target = state.error.target;
      state.touched = true;
      state.error = null;
      state.desired = target;
      render();
      enqueue(target);
    }

    function loadInitial() {
      var controller = newAbort();
      initialAbort = controller;
      serialized(id, function () {
        return getVote(id, controller.signal);
      }).then(function (payload) {
        initialAbort = null;
        /* 用户已经点过就不回填，避免把乐观态覆盖成旧值 */
        if (state.destroyed || state.touched) return;
        state.confirmedVote = normalizeVote(payload && payload.vote);
        var counts = readCounts(payload);
        if (counts) state.counts = counts;
        state.desired = state.confirmedVote;
        render();
      }, function () {
        initialAbort = null;
        /* GET 失败静默：按钮保持可投，错误只由写请求触发 */
      });
    }

    function destroy() {
      if (state.destroyed) return;
      state.destroyed = true;
      queue.length = 0;
      if (currentAbort) {
        try { currentAbort.abort(); } catch (e) { /* 忽略 */ }
        currentAbort = null;
      }
      if (initialAbort) {
        try { initialAbort.abort(); } catch (e) { /* 忽略 */ }
        initialAbort = null;
      }
      if (root.parentNode) root.parentNode.removeChild(root);
    }

    /* 卸载入口：页面切换时调一次，abort 在途请求。
       不用 defineProperty 也行，但这样不会出现在 console 的 DOM 打印里。 */
    Object.defineProperty(root, "destroy", {
      value: destroy,
      writable: false,
      configurable: true,
      enumerable: false
    });

    container.appendChild(root);
    render();
    loadInitial();
    return root;
  }

  global.GYTFeedback = { mount: mount };
})(window);
