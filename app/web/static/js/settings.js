// ─── Settings (CORS & configuration serveur) ────────

async function renderSettings() {
  const el = document.getElementById('page-settings');

  // Indicateur d'état du token
  const tokenConfigured = hasAuthToken();
  const tokenStatus = tokenConfigured
    ? `<span class="text-green-300">Token configuré ✓</span>`
    : `<span class="text-gray-400">Aucun token (auth désactivée)</span>`;

  el.innerHTML = `
    <div class="max-w-2xl">
      <h1 class="text-2xl font-bold mb-1">⚙️ Settings</h1>
      <p class="text-sm text-gray-400 mb-6">Configuration du serveur AI Runner</p>

      <!-- Auth Token -->
      <div class="bg-dark-800 rounded-xl p-5 border border-dark-600 mb-4">
        <div class="flex items-center gap-2 mb-2">
          <span class="text-lg">🔐</span>
          <h2 class="text-base font-semibold">Auth Token</h2>
        </div>
        <p class="text-xs text-gray-400 mb-3">
          Si le backend a configuré <code class="text-gray-300">config.server.auth_token</code>,
          renseignez ici le même token. Il est stocké uniquement dans votre navigateur (localStorage)
          et envoyé via l'en-tête <code class="text-gray-300">Authorization: Bearer {token}</code>.
          Si aucun token n'est défini, l'authentification est désactivée (aucun en-tête envoyé).
        </p>

        <div id="authTokenStatus" class="mb-3 text-sm font-medium">${tokenStatus}</div>

        <input id="authTokenInput" type="password"
          class="w-full bg-dark-900 text-gray-100 border border-dark-600 rounded-lg p-3 text-sm font-mono focus:outline-none focus:border-blue-500"
          placeholder="${tokenConfigured ? '•••••••••••• (token configuré)' : 'Collez votre token ici…'}"
          autocomplete="off" spellcheck="false" />

        <div id="authTokenMessage" class="mt-3 text-sm hidden"></div>

        <div class="flex gap-3 mt-4">
          <button id="saveAuthTokenBtn"
            onclick="saveAuthToken()"
            class="px-5 py-2.5 bg-blue-600 hover:bg-blue-700 rounded-lg text-sm font-semibold transition">
            💾 Save Token
          </button>
          <button id="clearAuthTokenBtn"
            onclick="removeAuthToken()"
            class="px-4 py-2.5 bg-dark-600 hover:bg-red-700 rounded-lg text-sm transition">
            🗑️ Effacer
          </button>
        </div>
      </div>

      <!-- CORS Origins -->
      <div class="bg-dark-800 rounded-xl p-5 border border-dark-600 mb-4">
        <div class="flex items-center gap-2 mb-2">
          <span class="text-lg">🌐</span>
          <h2 class="text-base font-semibold">Origines CORS</h2>
        </div>
        <p class="text-xs text-gray-400 mb-3">
          Une origine par ligne (URL complète, ex: <code class="text-gray-300">https://app.exemple.com</code>).
          Utilisez <code class="text-gray-300">*</code> pour autoriser toutes les origines (non recommandé en production).
        </p>

        <textarea id="corsOrigins"
          class="w-full h-40 bg-dark-900 text-gray-100 border border-dark-600 rounded-lg p-3 text-sm font-mono focus:outline-none focus:border-blue-500 scrollbar-thin resize-y"
          placeholder="https://localhost:3000&#10;https://app.exemple.com"
          spellcheck="false"></textarea>

        <div id="settingsMessage" class="mt-3 text-sm hidden"></div>

        <div class="flex gap-3 mt-4">
          <button id="saveSettingsBtn"
            onclick="saveSettings()"
            class="px-5 py-2.5 bg-blue-600 hover:bg-blue-700 rounded-lg text-sm font-semibold transition">
            💾 Save
          </button>
          <button id="reloadSettingsBtn"
            onclick="renderSettings()"
            class="px-4 py-2.5 bg-dark-600 hover:bg-dark-500 rounded-lg text-sm transition">
            🔄 Recharger
          </button>
        </div>
      </div>
    </div>
  `;

  // Charger la config existante
  await loadSettings();
}

async function loadSettings() {
  const ta = document.getElementById('corsOrigins');
  if (!ta) return;

  ta.disabled = true;
  ta.placeholder = '⏳ Chargement de la configuration…';

  try {
    const config = await getServerConfig();
    const origins = Array.isArray(config.cors_origins) ? config.cors_origins : [];
    ta.value = origins.join('\n');
  } catch (e) {
    showSettingsMessage(`❌ Impossible de charger la configuration: ${e.message}`, 'error');
  } finally {
    ta.disabled = false;
    ta.placeholder = 'https://localhost:3000\nhttps://app.exemple.com';
  }
}

async function saveSettings() {
  const ta = document.getElementById('corsOrigins');
  const btn = document.getElementById('saveSettingsBtn');
  if (!ta) return;

  const origins = ta.value
    .split('\n')
    .map(l => l.trim())
    .filter(l => l.length > 0);

  const originalText = btn.textContent;
  btn.textContent = '⏳ Sauvegarde…';
  btn.disabled = true;

  try {
    await updateServerConfig({ cors_origins: origins });
    showSettingsMessage(`✅ Configuration sauvegardée (${origins.length} origine${origins.length > 1 ? 's' : ''}).`, 'success');
  } catch (e) {
    showSettingsMessage(`❌ Échec de la sauvegarde: ${e.message}`, 'error');
  } finally {
    btn.textContent = originalText;
    btn.disabled = false;
  }
}

// ─── Auth Token save / clear ──────────────────────────

function saveAuthToken() {
  const input = document.getElementById('authTokenInput');
  const btn = document.getElementById('saveAuthTokenBtn');
  if (!input) return;

  const token = input.value.trim();
  if (!token) {
    showAuthTokenMessage('⚠️ Veuillez saisir un token non vide.', 'error');
    return;
  }

  const originalText = btn.textContent;
  btn.textContent = '⏳…';
  btn.disabled = true;

  setAuthToken(token);
  input.value = '';
  input.placeholder = '•••••••••••• (token configuré)';
  showAuthTokenMessage('✅ Token sauvegardé dans le navigateur.', 'success');
  updateAuthTokenStatus();

  btn.textContent = originalText;
  btn.disabled = false;
}

function removeAuthToken() {
  const btn = document.getElementById('clearAuthTokenBtn');
  const originalText = btn.textContent;
  btn.textContent = '⏳…';
  btn.disabled = true;

  clearAuthToken();
  const input = document.getElementById('authTokenInput');
  if (input) {
    input.value = '';
    input.placeholder = 'Collez votre token ici…';
  }
  showAuthTokenMessage('🗑️ Token supprimé.', 'success');
  updateAuthTokenStatus();

  btn.textContent = originalText;
  btn.disabled = false;
}

function updateAuthTokenStatus() {
  const el = document.getElementById('authTokenStatus');
  if (!el) return;
  if (hasAuthToken()) {
    el.innerHTML = `<span class="text-green-300">Token configuré ✓</span>`;
  } else {
    el.innerHTML = `<span class="text-gray-400">Aucun token (auth désactivée)</span>`;
  }
}

function showAuthTokenMessage(msg, type) {
  const el = document.getElementById('authTokenMessage');
  if (!el) return;
  el.textContent = msg;
  el.classList.remove('hidden', 'text-green-300', 'text-red-300');
  el.classList.add(type === 'error' ? 'text-red-300' : 'text-green-300');
}

function showSettingsMessage(msg, type) {
  const el = document.getElementById('settingsMessage');
  if (!el) return;
  el.textContent = msg;
  el.classList.remove('hidden', 'text-green-300', 'text-red-300');
  el.classList.add(type === 'error' ? 'text-red-300' : 'text-green-300');
}