// ─── Terminal ────────────────────────────────────────

let _chatModelId = null;
let _chatMessages = [];
let _isRunning = false;

async function renderTerminal() {
  const el = document.getElementById('page-terminal');

  // Charger les modèles pour le sélecteur
  let models = [];
  try {
    models = await getModels();
  } catch (e) { /* ignore */ }

  if (!_chatModelId && models.length > 0) {
    _chatModelId = models[0].id;
  }

  el.innerHTML = `
    <!-- Controls -->
    <div class="flex gap-3 items-center mb-4 flex-wrap">
      <select id="chatModelSelect" class="bg-dark-800 border border-dark-600 rounded-lg px-3 py-2 text-sm flex-1 max-w-xs">
        ${models.map(m => `<option value="${m.id}" ${m.id === _chatModelId ? 'selected' : ''}>${m.name || m.id}</option>`).join('')}
        ${models.length === 0 ? '<option value="">Aucun modèle disponible</option>' : ''}
      </select>
      <button id="chatConfigBtn" onclick="navigate('config')" class="px-3 py-2 bg-dark-600 hover:bg-dark-500 rounded-lg text-sm transition">⚙️</button>
      <button id="chatClearBtn" onclick="clearChat()" class="px-3 py-2 bg-dark-600 hover:bg-dark-500 rounded-lg text-sm transition">🗑️ Effacer</button>
      <div id="chatStats" class="text-xs text-gray-400 ml-auto hidden"></div>
    </div>

    <!-- Messages -->
    <div id="chatMessages" class="bg-dark-800 rounded-xl border border-dark-600 p-4 h-[55vh] overflow-y-auto mb-4 scrollbar-thin space-y-3">
      <div class="text-center text-gray-500 py-8 text-sm">
        ${models.length === 0
          ? 'Aucun modèle disponible. Allez dans l\'onglet Modèles pour en télécharger.'
          : 'Sélectionnez un modèle et écrivez un message pour commencer.'}
      </div>
    </div>

    <!-- Input -->
    <div class="flex gap-2">
      <textarea id="chatInput" rows="2" placeholder="Écrivez votre message..."
                class="flex-1 bg-dark-800 border border-dark-600 rounded-xl px-4 py-3 text-sm resize-none focus:outline-none focus:border-blue-500"
                onkeydown="if(event.key==='Enter'&&!event.shiftKey){event.preventDefault();sendChatMessage()}"></textarea>
      <button id="chatSendBtn" onclick="sendChatMessage()"
              class="px-6 py-3 bg-blue-600 hover:bg-blue-700 disabled:bg-dark-600 disabled:cursor-not-allowed rounded-xl font-semibold transition"
              ${models.length === 0 ? 'disabled' : ''}>
        Envoyer
      </button>
      <button id="chatStopBtn" onclick="stopChat()" class="hidden px-4 py-3 bg-red-700 hover:bg-red-600 rounded-xl transition">⏹</button>
    </div>
  `;
}

async function sendChatMessage() {
  const input = document.getElementById('chatInput');
  const message = input.value.trim();
  if (!message || _isRunning) return;

  const modelId = document.getElementById('chatModelSelect').value;
  if (!modelId) return;

  _chatModelId = modelId;
  input.value = '';

  // Ajouter le message utilisateur
  _chatMessages.push({ role: 'user', content: message });
  addMessage('user', message);

  // Afficher le message assistant en cours
  const assistantDiv = addMessage('assistant', '');

  // UI: désactiver l'input, montrer le stop
  _isRunning = true;
  document.getElementById('chatSendBtn').classList.add('hidden');
  document.getElementById('chatStopBtn').classList.remove('hidden');
  const statsEl = document.getElementById('chatStats');
  statsEl.classList.remove('hidden');

  let fullResponse = '';
  let tokenCount = 0;

  chatStream(modelId, _chatMessages, {}, (event) => {
    if (event.type === 'token') {
      fullResponse += event.text + ' ';
      assistantDiv.textContent = fullResponse;
      assistantDiv.scrollIntoView({ behavior: 'smooth' });
      tokenCount++;
      statsEl.textContent = `⚡ ${event.speed || '?'} tok/s • ${event.tokens || tokenCount} tokens`;
    } else if (event.type === 'stats') {
      if (event.vram_gb) statsEl.textContent += ` • 🎮 ${event.vram_gb}G VRAM`;
    } else if (event.type === 'error') {
      assistantDiv.textContent = '❌ ' + event.message;
    } else if (event.type === 'log') {
      console.log('[llama.cpp]', event.text);
    }
  }, () => {
    // Done
    _isRunning = false;
    document.getElementById('chatSendBtn').classList.remove('hidden');
    document.getElementById('chatStopBtn').classList.add('hidden');
    statsEl.classList.add('hidden');

    if (fullResponse) {
      _chatMessages.push({ role: 'assistant', content: fullResponse.trim() });
    }
  }, (error) => {
    // Error
    _isRunning = false;
    document.getElementById('chatSendBtn').classList.remove('hidden');
    document.getElementById('chatStopBtn').classList.add('hidden');
    statsEl.classList.add('hidden');
    assistantDiv.textContent = '❌ ' + error;
    assistantDiv.className = 'text-red-400';
  });
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
    <div class="text-center text-gray-500 py-8 text-sm">
      Conversation effacée. Écrivez un message pour commencer une nouvelle discussion.
    </div>`;
}

function addMessage(role, content) {
  const container = document.getElementById('chatMessages');

  // Enlever le placeholder si présent
  const placeholder = container.querySelector('.text-gray-500.py-8');
  if (placeholder) placeholder.remove();

  const div = document.createElement('div');
  div.className = role === 'user'
    ? 'bg-blue-900/20 rounded-lg px-4 py-2 ml-8'
    : 'bg-dark-700/50 rounded-lg px-4 py-2 mr-8';

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
