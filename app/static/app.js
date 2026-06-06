// Competitor Intelligence Agent — chat UI logic (vanilla JS, no dependencies).
"use strict";

(function () {
  // ---------- DOM references ----------
  const messagesEl = document.getElementById("messages");
  const promptEl = document.getElementById("prompt");
  const sendBtn = document.getElementById("send");
  const companyEl = document.getElementById("company");
  const periodEl = document.getElementById("period");
  const formEl = document.getElementById("composer");

  // ---------- helpers ----------

  // Scroll the messages container to the bottom.
  function scrollToBottom() {
    messagesEl.scrollTop = messagesEl.scrollHeight;
  }

  // Create a message bubble of a given role: "user" | "assistant" | "error".
  // Text is set via textContent to prevent XSS.
  function appendMessage(role, text) {
    const msg = document.createElement("div");
    msg.className = "msg msg-" + role;

    const bubble = document.createElement("div");
    bubble.className = "bubble";
    bubble.textContent = text;

    msg.appendChild(bubble);
    messagesEl.appendChild(msg);
    scrollToBottom();
    return msg; // returned so callers can extend it (e.g. add sources)
  }

  // Append an assistant "loading" indicator and return its element so it can
  // be replaced once the response arrives.
  function appendLoading() {
    const msg = document.createElement("div");
    msg.className = "msg msg-assistant";

    const bubble = document.createElement("div");
    bubble.className = "bubble";

    const typing = document.createElement("span");
    typing.className = "typing";
    for (let i = 0; i < 3; i++) {
      const dot = document.createElement("span");
      dot.className = "dot";
      typing.appendChild(dot);
    }

    bubble.appendChild(typing);
    msg.appendChild(bubble);
    messagesEl.appendChild(msg);
    scrollToBottom();
    return msg;
  }

  // Render the assistant answer plus an optional sources list into a message
  // element (replacing any loading indicator inside it).
  function renderAnswer(msgEl, answer, sources) {
    msgEl.innerHTML = ""; // clear loading indicator

    const bubble = document.createElement("div");
    bubble.className = "bubble";
    bubble.textContent = answer || "(пустой ответ)";
    msgEl.appendChild(bubble);

    if (Array.isArray(sources) && sources.length > 0) {
      const list = document.createElement("ul");
      list.className = "sources";

      const title = document.createElement("li");
      title.className = "sources-title";
      title.textContent = "Источники";
      list.appendChild(title);

      sources.forEach(function (url) {
        if (!url) return;
        const li = document.createElement("li");
        const a = document.createElement("a");
        a.href = url;
        a.textContent = url;
        a.target = "_blank";
        a.rel = "noopener";
        li.appendChild(a);
        list.appendChild(li);
      });

      msgEl.appendChild(list);
    }

    scrollToBottom();
  }

  // Build the request payload, omitting optional fields when not provided.
  function buildPayload(query) {
    const payload = { query: query };

    const company = (companyEl.value || "").trim();
    payload.company = company !== "" ? company : null;

    const periodRaw = (periodEl.value || "").trim();
    const period = parseInt(periodRaw, 10);
    payload.period_days = !isNaN(period) && periodRaw !== "" ? period : null;

    return payload;
  }

  // Toggle the send button / textarea disabled state during a request.
  function setBusy(busy) {
    sendBtn.disabled = busy;
    promptEl.disabled = busy;
  }

  // ---------- main send flow ----------

  async function sendQuery() {
    const query = (promptEl.value || "").trim();
    if (query === "") return; // ignore empty input

    appendMessage("user", query);
    promptEl.value = "";

    const loadingEl = appendLoading();
    setBusy(true);

    try {
      const response = await fetch("/query", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(buildPayload(query)),
      });

      if (!response.ok) {
        // Try to extract a useful detail message from the error response.
        let detail = response.status + " " + response.statusText;
        try {
          const errData = await response.json();
          if (errData && errData.detail) detail = errData.detail;
        } catch (e) {
          /* response had no JSON body; keep the status text */
        }
        loadingEl.remove();
        appendMessage("error", "Ошибка запроса: " + detail);
        return;
      }

      const data = await response.json();
      renderAnswer(loadingEl, data.answer, data.sources);
    } catch (err) {
      // Network failure or unexpected error: never leave a stuck loader.
      loadingEl.remove();
      appendMessage("error", "Ошибка запроса: " + (err && err.message ? err.message : "сеть недоступна"));
    } finally {
      setBusy(false);
      promptEl.focus();
    }
  }

  // ---------- event wiring ----------

  // Submit via the form (covers button click).
  formEl.addEventListener("submit", function (e) {
    e.preventDefault();
    sendQuery();
  });

  // Enter sends; Shift+Enter inserts a newline.
  promptEl.addEventListener("keydown", function (e) {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      sendQuery();
    }
  });

  promptEl.focus();
})();
