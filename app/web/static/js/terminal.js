// ─── Chat — interface de chat fusionnée ────────────

let _chatModelId = null;
let _chatMessages = [];
let _isRunning = false;

async function renderTerminal() {
  const el = document.getElementById('page-terminal');

  let models = [];
  try { models = await getModels(); } catch (e) { /* ignore */ }

  if (!_chatModelId && models.length > 0) {
    _chatModelId = models[0].id;
  }

  el.innerHTML = `
    <div class="flex flex-col lg:flex-row gap-4 h-[calc(100vh-8rem)]">
      <!-- Panneau de contrôle (gauche) -->
      <div class="lg:w-72 bg-dark-800 rounded-xl border border-dark-600 p-4 flex flex-col gap-3 shrink-0">
        <h2 class="text-sm font-semibold text-gray-300">⚙️ Paramètres</h2>

        <label class="text-xs text-gray-400">Modèle</label>
        <select id="chatModelSelect" class="bg-dark-700 border border-dark-600 rounded-lg px-3 py-2 text-sm">
          ${models.map(m => `<option value="${m.id}" ${m.id === _chatModelId ? 'selected' : ''}>${m.name || m.id} (${m.params_b}B)</option>`).join('')}
          ${models.length === 0 ? '<option value="">Aucun modèle disponible</option>' : ''}
        </select>

        <label class="text-xs text-gray-400">Température</label>
        <div class="flex items-center gap-2">
          <input id="chatTemp" type="range" min="0" max="2" step="0.1" value="0.7"
                 class="flex-1 accent-blue-500" oninput="document.getElementById('chatTempVal').textContent=this.value">
          <span id="chatTempVal" class="text-xs font-mono w-8 text-right">0.7</span>
        </div>

        <label class="text-xs text-gray-400">Max tokens</label>
        <select id="chatMaxTokens" class="bg-dark-700 border border-dark-600 rounded-lg px-3 py-2 text-sm">
          <option value="256">256</option>
          <option value="512" selected>512</option>
          <option value="1024">1 024</option>
          <option value="2048">2 048</option>
          <option value="4096">4 096</option>
        </select>

        <label class="text-xs text-gray-400">Contexte</label>
        <select id="chatCtxSize" class="bg-dark-700 border border-dark-600 rounded-lg px-3 py-2 text-sm">
          <option value="4096">4 096</option>
          <option value="8192" selected>8 192</option>
          <option value="16384">16 384</option>
          <option value="32768">32 768</option>
          <option value="65536">65 536</option>
        </select>

        <div class="mt-auto pt-2 border-t border-dark-600 space-y-2">
          <button onclick="clearChat()"
                  class="w-full px-3 py-2 bg-dark-600 hover:bg-dark-500 rounded-lg text-xs transition">
            🗑️ Effacer la conversation
          </button>
        </div>
      </div>

      <!-- Zone de chat (droite) -->
      <div class="flex-1 flex flex-col bg-dark-800 rounded-xl border border-dark-600 overflow-hidden">
        <!-- Messages -->
        <div id="chatMessages" class="flex-1 overflow-y-auto p-4 space-y-3 scrollbar-thin">
          <div class="text-center text-gray-500 py-12 text-sm">
            ${models.length === 0
              ? 'Aucun modèle disponible. Allez dans l\'onglet Modèles pour en télécharger.'
              : 'Sélectionnez un modèle et écrivez un message.'}
          </div>
        </div>

        <!-- Stats bar -->
        <div id="chatStats" class="hidden px-4 py-1.5 bg-dark-900/50 border-t border-dark-600 text-xs text-gray-400 flex gap-4 flex-wrap">
          <span id="chatStatsSpeed">⚡ -- tok/s</span>
          <span id="chatStatsTokens">📝 -- tokens</span>
          <span id="chatStatsVRAM">🎮 -- VRAM</span>
          <span id="chatStatsTime">⏱ -- s</span>
        </div>

        <!-- Input -->
        <div class="p-3 border-t border-dark-600 flex gap-2">
          <textarea id="chatInput" rows="2" placeholder="Écrivez votre message…"
                    class="flex-1 bg-dark-700 border border-dark-600 rounded-xl px-4 py-3 text-sm resize-none focus:outline-none focus:border-blue-500"
                    onkeydown="if(event.key==='Enter'&&!event.shiftKey){event.preventDefault();sendMessage()}"></textarea>
          <button id="chatSendBtn" onclick="sendMessage()"
                  class="px-5 py-3 bg-blue-600 hover:bg-blue-700 disabled:bg-dark-600 disabled:cursor-not-allowed rounded-xl font-semibold transition text-sm"
                  ${models.length === 0 ? 'disabled' : ''}>
            ▶️
          </button>
          <button id="chatStopBtn" onclick="stopChat()"
                  class="hidden px-4 py-3 bg-red-700 hover:bg-red-600 rounded-xl transition text-sm">⏹</button>
        </div>
      </div>
    </div>
  `;
}


// ─── Chat logic ────────────────────────────────────

async function sendMessage() {
  const input = document.getElementById('chatInput');
  const message = input.value.trim();
  if (!message || _isRunning) return;

  const modelId = document.getElementById('chatModelSelect').value;
  if (!modelId) return;

  _chatModelId = modelId;
  input.value = '';

  _chatMessages.push({ role: 'user', content: message });
  addMessage('user', message);

  // Assistant placeholder
  const assistantDiv = addMessage('assistant', '⏳');

  // UI state
  _isRunning = true;
  const startTime = Date.now();
  document.getElementById('chatSendBtn').classList.add('hidden');
  document.getElementById('chatStopBtn').classList.remove('hidden');
  document.getElementById('chatStats').classList.remove('hidden');

  const temp = parseFloat(document.getElementById('chatTemp').value) || 0.7;
  const maxTokens = parseInt(document.getElementById('chatMaxTokens').value) || 512;
  const ctxSize = parseInt(document.getElementById('chatCtxSize').value) || 8192;

  let fullResponse = '';
  let tokenCount = 0;

  chatStream(modelId, _chatMessages, { temp, max_tokens: maxTokens, ctx_size: ctxSize },
    (event) => {
      if (event.type === 'token') {
        fullResponse += event.text + ' ';
        if (assistantDiv) {
          assistantDiv.textContent = fullResponse;
          assistantDiv.scrollIntoView({ behavior: 'smooth' });
        }
        tokenCount++;
        const elapsed = ((Date.now() - startTime) / 1000).toFixed(1);
        document.getElementById('chatStatsSpeed').textContent = `⚡ ${event.speed || '?'} tok/s`;
        document.getElementById('chatStatsTokens').textContent = `📝 ${tokenCount} tokens`;
        document.getElementById('chatStatsTime').textContent = `⏱ ${elapsed}s`;
      } else if (event.type === 'stats') {
        if (event.vram_gb) {
          document.getElementById('chatStatsVRAM').textContent = `🎮 ${event.vram_gb}G VRAM`;
        }
      } else if (event.type === 'error') {
        if (assistantDiv) assistantDiv.textContent = '❌ ' + event.message;
      } else if (event.type === 'log') {
        console.log('[llama.cpp]', event.text);
      }
    },
    () => {
      _isRunning = false;
      document.getElementById('chatSendBtn').classList.remove('hidden');
      document.getElementById('chatStopBtn').classList.add('hidden');
      if (fullResponse) {
        _chatMessages.push({ role: 'assistant', content: fullResponse.trim() });
      }
    },
    (error) => {
      _isRunning = false;
      document.getElementById('chatSendBtn').classList.remove('hidden');
      document.getElementById('chatStopBtn').classList.add('hidden');
      if (assistantDiv) {
        assistantDiv.textContent = '❌ ' + error;
        assistantDiv.className = 'text-sm text-red-400';
      }
    }
  );
}

async function stopChat() {
  try {
    await stopRun();
    _isRunning = false;
    document.getElementById('chatSendBtn').classList.remove('hidden');
    document.getElementById('chatStopBtn').classList.add('hidden');
    addMessage('system', '⏹ Génération arrêtée.');
  } catch (e) {
    console.warn('Stop failed:', e);
  }
}

function clearChat() {
  _chatMessages = [{ role: 'system', content: 'Tu es un assistant IA utile.' }];
  const container = document.getElementById('chatMessages');
  container.innerHTML = `
    <div class="text-center text-gray-500 py-12 text-sm">
      Conversation effacée.
    </div>`;
}

function addMessage(role, content) {
  const container = document.getElementById('chatMessages');

  // Enlever le placeholder si présent
  const placeholder = container.querySelector('.text-gray-500.py-8, .text-gray-500.py-12');
  if (placeholder) placeholder.remove();

  const div = document.createElement('div');
  div.className = role === 'user'
    ? 'bg-blue-900/20 rounded-lg px-4 py-2.5 ml-8'
    : 'bg-dark-700/50 rounded-lg px-4 py-2.5 mr-8';

  const label = document.createElement('div');
  label.className = 'text-xs ' + (role === 'user' ? 'text-blue-300' : 'text-green-300') + ' mb-1';
  label.textContent = role === 'user' ? '👤 Vous' : role === 'assistant' ? '🤖 Assistant' : '🔧 Système';
  div.appendChild(label);

  const text = document.createElement('div');
  text.className = 'text-sm whitespace-pre-wrap';
  text.textContent = content;
  div.appendChild(text);

  container.appendChild(div);
  container.scrollTop = container.scrollHeight;
  return text;
}
