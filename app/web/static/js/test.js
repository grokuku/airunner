// ─── Test Panel — Dialogue rapide avec le modèle ─────

let _testModelId = null;

async function renderTest() {
  const el = document.getElementById('page-test');

  // Charger les modèles disponibles
  let models = [];
  try { models = await getModels(); } catch (e) { /* ignore */ }

  if (!_testModelId && models.length > 0) {
    _testModelId = models[0].id;
  }

  el.innerHTML = `
    <div class="flex flex-col lg:flex-row gap-4 h-[calc(100vh-8rem)]">
      <!-- Panneau de contrôle (gauche) -->
      <div class="lg:w-80 bg-dark-800 rounded-xl border border-dark-600 p-4 flex flex-col gap-3 shrink-0">
        <h2 class="text-sm font-semibold text-gray-300">🧪 Test — Paramètres</h2>

        <label class="text-xs text-gray-400">Modèle</label>
        <select id="testModel" class="bg-dark-700 border border-dark-600 rounded-lg px-3 py-2 text-sm"
                onchange="_testModelId = this.value">
          ${models.map(m =>
            `<option value="${m.id}" ${m.id === _testModelId ? 'selected' : ''}>
              ${m.name || m.id} (${m.params_b}B)
            </option>`
          ).join('')}
          ${models.length === 0 ? '<option value="">Aucun modèle</option>' : ''}
        </select>

        <label class="text-xs text-gray-400">Température</label>
        <div class="flex items-center gap-2">
          <input id="testTemp" type="range" min="0" max="2" step="0.1" value="0.7"
                 class="flex-1 accent-blue-500" oninput="document.getElementById('testTempVal').textContent=this.value">
          <span id="testTempVal" class="text-xs font-mono w-8 text-right">0.7</span>
        </div>

        <label class="text-xs text-gray-400">Max tokens</label>
        <select id="testMaxTokens" class="bg-dark-700 border border-dark-600 rounded-lg px-3 py-2 text-sm">
          <option value="256">256</option>
          <option value="512" selected>512</option>
          <option value="1024">1 024</option>
          <option value="2048">2 048</option>
          <option value="4096">4 096</option>
        </select>

        <label class="text-xs text-gray-400">Contexte</label>
        <select id="testCtxSize" class="bg-dark-700 border border-dark-600 rounded-lg px-3 py-2 text-sm">
          <option value="4096">4 096</option>
          <option value="8192" selected>8 192</option>
          <option value="16384">16 384</option>
          <option value="32768">32 768</option>
        </select>

        <div class="mt-auto pt-2 border-t border-dark-600">
          <button id="testClearBtn" onclick="clearTest()"
                  class="w-full px-3 py-2 bg-dark-600 hover:bg-dark-500 rounded-lg text-xs transition">
            🗑️ Effacer la conversation
          </button>
        </div>
      </div>

      <!-- Zone de chat (droite) -->
      <div class="flex-1 flex flex-col bg-dark-800 rounded-xl border border-dark-600 overflow-hidden">
        <!-- Messages -->
        <div id="testMessages" class="flex-1 overflow-y-auto p-4 space-y-3 scrollbar-thin">
          <div class="text-center text-gray-500 py-12 text-sm">
            Sélectionnez un modèle et écrivez un message.
          </div>
        </div>

        <!-- Stats bar -->
        <div id="testStats" class="hidden px-4 py-1.5 bg-dark-900/50 border-t border-dark-600 text-xs text-gray-400 flex gap-4">
          <span id="testStatsSpeed">⚡ -- tok/s</span>
          <span id="testStatsTokens">📝 -- tokens</span>
          <span id="testStatsTime">⏱ -- s</span>
        </div>

        <!-- Input -->
        <div class="p-3 border-t border-dark-600 flex gap-2">
          <textarea id="testInput" rows="2" placeholder="Écrivez votre message…"
                    class="flex-1 bg-dark-700 border border-dark-600 rounded-xl px-4 py-3 text-sm resize-none focus:outline-none focus:border-blue-500"
                    onkeydown="if(event.key==='Enter'&&!event.shiftKey){event.preventDefault();sendTestMessage()}"></textarea>
          <button id="testSendBtn" onclick="sendTestMessage()"
                  class="px-5 py-3 bg-blue-600 hover:bg-blue-700 disabled:bg-dark-600 disabled:cursor-not-allowed rounded-xl font-semibold transition text-sm"
                  ${models.length === 0 ? 'disabled' : ''}>
            ▶️
          </button>
          <button id="testStopBtn" onclick="stopTest()"
                  class="hidden px-4 py-3 bg-red-700 hover:bg-red-600 rounded-xl transition text-sm">⏹</button>
        </div>
      </div>
    </div>
  `;
}

let _testMessages = [{ role: 'system', content: 'Tu es un assistant IA utile et concis.' }];
let _testRunning = false;
let _testStartTime = null;

function addTestMessage(role, content) {
  const container = document.getElementById('testMessages');
  const placeholder = container.querySelector('.text-gray-500.py-12');
  if (placeholder) placeholder.remove();

  const div = document.createElement('div');
  div.className = role === 'user'
    ? 'bg-blue-900/20 rounded-lg px-4 py-2.5 ml-8'
    : 'bg-dark-700/50 rounded-lg px-4 py-2.5 mr-8';

  const label = document.createElement('div');
  label.className = 'text-xs ' + (role === 'user' ? 'text-blue-300' : 'text-green-300') + ' mb-1';
  label.textContent = role === 'user' ? '👤 Vous' : '🤖 Modèle';
  div.appendChild(label);

  const text = document.createElement('div');
  text.className = 'text-sm whitespace-pre-wrap';
  text.textContent = content;
  text.id = role === 'assistant' ? 'testStreamingText' : '';
  div.appendChild(text);

  container.appendChild(div);
  container.scrollTop = container.scrollHeight;
  return text;
}

async function sendTestMessage() {
  const input = document.getElementById('testInput');
  const message = input.value.trim();
  if (!message || _testRunning) return;

  const modelId = document.getElementById('testModel').value;
  if (!modelId) return;

  _testModelId = modelId;
  input.value = '';

  _testMessages.push({ role: 'user', content: message });
  addTestMessage('user', message);

  // Show assistant placeholder
  addTestMessage('assistant', '⏳');
  const textEl = document.getElementById('testStreamingText');

  // Toggle UI
  _testRunning = true;
  _testStartTime = Date.now();
  document.getElementById('testSendBtn').classList.add('hidden');
  document.getElementById('testStopBtn').classList.remove('hidden');
  document.getElementById('testStats').classList.remove('hidden');

  const temp = parseFloat(document.getElementById('testTemp').value) || 0.7;
  const maxTokens = parseInt(document.getElementById('testMaxTokens').value) || 512;

  let response = '';
  let tokenCount = 0;

  chatStream(modelId, _testMessages, { temp, max_tokens: maxTokens },
    (event) => {
      if (event.type === 'token') {
        response += event.text + ' ';
        tokenCount++;
        if (textEl) {
          textEl.textContent = response;
          textEl.scrollIntoView({ behavior: 'smooth' });
        }
        const elapsed = ((Date.now() - _testStartTime) / 1000).toFixed(1);
        document.getElementById('testStatsSpeed').textContent = `⚡ ${event.speed || '?'} tok/s`;
        document.getElementById('testStatsTokens').textContent = `📝 ${tokenCount} tokens`;
        document.getElementById('testStatsTime').textContent = `⏱ ${elapsed}s`;
      } else if (event.type === 'error') {
        if (textEl) textEl.textContent = '❌ ' + event.message;
      }
    },
    () => {
      // Done
      _testRunning = false;
      document.getElementById('testSendBtn').classList.remove('hidden');
      document.getElementById('testStopBtn').classList.add('hidden');
      if (response) {
        _testMessages.push({ role: 'assistant', content: response.trim() });
      }
    },
    (error) => {
      _testRunning = false;
      document.getElementById('testSendBtn').classList.remove('hidden');
      document.getElementById('testStopBtn').classList.add('hidden');
      if (textEl) {
        textEl.textContent = '❌ ' + error;
        textEl.className = 'text-sm text-red-400';
      }
    }
  );
}

async function stopTest() {
  try {
    await stopRun();
    _testRunning = false;
    document.getElementById('testSendBtn').classList.remove('hidden');
    document.getElementById('testStopBtn').classList.add('hidden');
    const textEl = document.getElementById('testStreamingText');
    if (textEl && textEl.textContent === '⏳') textEl.textContent = '⏹ Arrêté.';
  } catch (e) {
    console.warn('Stop failed:', e);
  }
}

function clearTest() {
  _testMessages = [{ role: 'system', content: 'Tu es un assistant IA utile et concis.' }];
  const container = document.getElementById('testMessages');
  const msgs = container.querySelectorAll('div');
  container.innerHTML = `
    <div class="text-center text-gray-500 py-12 text-sm">
      Conversation effacée.
    </div>`;
  document.getElementById('testStats').classList.add('hidden');
}
