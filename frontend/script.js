const chatForm = document.getElementById('chatForm');
const questionInput = document.getElementById('questionInput');
const messagesEl = document.getElementById('messages');
const welcomeEl = document.getElementById('welcome');
const chatArea = document.getElementById('chatArea');
const sendBtn = document.getElementById('sendBtn');

document.querySelectorAll('.suggestion-chip').forEach(chip => {
  chip.addEventListener('click', () => {
    questionInput.value = chip.dataset.q;
    chatForm.dispatchEvent(new Event('submit'));
  });
});

function scrollToBottom() { chatArea.scrollTop = chatArea.scrollHeight; }

function addUserMessage(text) {
  const msg = document.createElement('div');
  msg.className = 'msg user';
  msg.innerHTML = `<div class="msg-avatar">You</div><div class="msg-content"><div class="msg-bubble">${escapeHtml(text)}</div></div>`;
  messagesEl.appendChild(msg);
  scrollToBottom();
}

function addTypingIndicator() {
  const msg = document.createElement('div');
  msg.className = 'msg bot';
  msg.id = 'typingMsg';
  msg.innerHTML = `<div class="msg-avatar">C9</div><div class="msg-content"><div class="msg-bubble"><div class="typing"><span></span><span></span><span></span></div></div></div>`;
  messagesEl.appendChild(msg);
  scrollToBottom();
}

function removeTypingIndicator() {
  const el = document.getElementById('typingMsg');
  if (el) el.remove();
}

function addBotMessage(answer, graphFacts, numChunks) {
  const msg = document.createElement('div');
  msg.className = 'msg bot';
  const formatted = answer.split('\n\n').map(p => `<p>${escapeHtml(p).replace(/\`([^\`]+)\`/g, '<code>$1</code>')}</p>`).join('');
  const factsHtml = graphFacts.length ? graphFacts.map(f => `<div class="fact">${escapeHtml(f)}</div>`).join('') : '<div class="fact">No direct graph relationships used</div>';
  msg.innerHTML = `<div class="msg-avatar">C9</div><div class="msg-content"><div class="msg-bubble">${formatted}</div><div class="sources-toggle" onclick="this.nextElementSibling.classList.toggle('open')">📊 ${numChunks} document chunks · ${graphFacts.length} graph facts used</div><div class="sources-panel">${factsHtml}</div></div>`;
  messagesEl.appendChild(msg);
  scrollToBottom();
}

function escapeHtml(str) {
  const div = document.createElement('div');
  div.textContent = str;
  return div.innerHTML;
}

chatForm.addEventListener('submit', async (e) => {
  e.preventDefault();
  const question = questionInput.value.trim();
  if (!question) return;

  welcomeEl.style.display = 'none';
  addUserMessage(question);
  questionInput.value = '';
  sendBtn.disabled = true;
  addTypingIndicator();

  try {
    const res = await fetch('/api/chat', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ question })
    });
    const data = await res.json();
    removeTypingIndicator();
    addBotMessage(data.answer, data.graph_facts, data.num_chunks);
  } catch (err) {
    removeTypingIndicator();
    const msg = document.createElement('div');
    msg.className = 'msg bot';
    msg.innerHTML = `<div class="msg-avatar">C9</div><div class="msg-content"><div class="msg-bubble">Something went wrong. Please try again.</div></div>`;
    messagesEl.appendChild(msg);
  }
  sendBtn.disabled = false;
  scrollToBottom();
});
