// ─── Configuration ───────────────────────────────────

let _configModelId = null;

async function renderConfig() {
  const el = document.getElementById('page-config');
  const modelId = window._configModelId;

  if (!modelId) {
    el.innerHTML = `
      <div class="text-center py-12 text-gray-400">
        <p class="mb-4">Sélectionnez un modèle depuis le Dashboard ou l'onglet Modèles.</p>
        <button onclick="navigate('models')" class="px-4 py-2 bg-blue-600 rounded-lg text-sm">📂 Voir les modèles</button>
      </div>`;
    return;
  }

  el.innerHTML = '<p class="text-gray-500">⏳ Analyse du modèle...</p>';

  try {
    const [model, suggestion] = await Promise.all([
      getModel(modelId),
      getConfigSuggestion(modelId),
    ]);

    const params = suggestion.params || {};

    el.innerHTML = `
      <!-- Model Info -->
      <div class="bg-dark-800 rounded-xl p-4 border border-dark-600 mb-4">
        <div class="flex justify-between items-start">
          <div>
            <h2 class="text-lg font-semibold">${model.name || modelId}</h2>
            <div class="flex gap-2 mt-1">
              <span class="text-xs px-2 py-0.5 rounded bg-dark-600">${model.architecture}</span>
              <span class="text-xs px-2 py-0.5 rounded bg-dark-600">${model.params_b}B paramètres</span>
              ${model.is_moe ? `<span class="text-xs px-2 py-0.5 rounded bg-purple-900">🔄 MoE • ${model.active_params_b}B actifs</span>` : ''}
              <span class="text-xs px-2 py-0.5 rounded bg-dark-600">${model.quant}</span>
            </div>
          </div>
          <div class="text-2xl font-bold ${suggestion.strategy === 'moe_offload' ? 'text-purple-400' : suggestion.strategy === 'dense_full' ? 'text-green-400' : 'text-yellow-400'}">
            ${suggestion.estimated_speed}
          </div>
        </div>
      </div>

      <!-- VRAM / RAM ──────────────────────────── -->
      <div class="grid grid-cols-1 md:grid-cols-2 gap-4 mb-4">
        <div class="bg-dark-800 rounded-xl p-4 border border-dark-600">
          <div class="text-xs text-gray-400 mb-2">🎮 VRAM</div>
          <div class="flex justify-between text-sm">
            <span>Poids: ${suggestion.vram.weights_gb} GB</span>
            <span>Cache KV: ${suggestion.vram.cache_kv_gb} GB</span>
          </div>
          <div class="mt-2 h-3 bg-dark-700 rounded-full overflow-hidden">
            <div class="h-full rounded-full progress-bar ${suggestion.vram.total_gb > suggestion.vram.available_gb * 0.8 ? 'bg-yellow-500' : 'bg-blue-500'}"
                 style="width:${Math.min(100, (suggestion.vram.total_gb / Math.max(suggestion.vram.available_gb, 1)) * 100)}%"></div>
          </div>
          <div class="flex justify-between text-xs text-gray-400 mt-1">
            <span>${suggestion.vram.total_gb}G utilisés</span>
            <span>${suggestion.vram.available_gb}G disponibles</span>
          </div>
        </div>
        ${suggestion.ram ? `
        <div class="bg-dark-800 rounded-xl p-4 border border-dark-600">
          <div class="text-xs text-gray-400 mb-2">🧠 RAM (offloading)</div>
          <div class="flex justify-between text-sm">
            <span>Experts: ${suggestion.ram.weights_gb} GB</span>
          </div>
          <div class="mt-2 h-3 bg-dark-700 rounded-full overflow-hidden">
            <div class="h-full rounded-full bg-green-500 progress-bar"
                 style="width:${Math.min(100, (suggestion.ram.weights_gb / Math.max(suggestion.ram.available_gb, 1)) * 100)}%"></div>
          </div>
          <div class="flex justify-between text-xs text-gray-400 mt-1">
            <span>${suggestion.ram.weights_gb}G utilisés</span>
            <span>${suggestion.ram.available_gb}G disponibles</span>
          </div>
        </div>` : ''}
      </div>

      <!-- Strategy Info ───────────────────────── -->
      <div class="bg-dark-800 rounded-xl p-4 border border-dark-600 mb-4">
        <div class="flex items-center gap-2 mb-2">
          <span class="text-sm font-semibold">🧠 Stratégie</span>
          <span class="text-xs px-2 py-0.5 rounded ${strategyBadge(suggestion.strategy)}">${strategyLabel(suggestion.strategy)}</span>
        </div>
        <div class="text-sm text-gray-300">
          ${strategyDescription(suggestion.strategy, model)}
        </div>
        ${suggestion.warnings?.length ? `
        <div class="mt-2 space-y-1">
          ${suggestion.warnings.map(w => `<div class="text-xs text-yellow-400">⚠️ ${w}</div>`).join('')}
        </div>` : ''}
      </div>

      <!-- Parameters ─────────────────────────── -->
      <div class="bg-dark-800 rounded-xl p-4 border border-dark-600 mb-4">
        <h3 class="text-sm font-semibold mb-3">⚙️ Paramètres</h3>
        <div class="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-4 gap-3">
          <param-field label="Quant" value="${params.quant || '?'}" />
          <param-field label="Contexte" value="${params.ctx_size || '?'} tokens" />
          <param-field label="GPU Layers" value="${params.ngl === 99 ? 'Tout' : params.ngl || 0}" />
          <param-field label="Cache KV" value="${params.cache_type_k || '?'}" />
          <param-field label="Threads" value="${params.threads || '?'}" />
          <param-field label="Batch" value="${params.batch_size || '?'}" />
          <param-field label="Flash Attn" value="${params.flash_attn ? '✅' : '❌'}" />
          <param-field label="Temperature" value="${params.temp || '0.7'}" />
        </div>
        ${params.override_tensor?.length ? `
        <div class="mt-3">
          <div class="text-xs text-gray-400 mb-1">Override tensor (MoE):</div>
          ${params.override_tensor.map(t => `<code class="text-xs bg-dark-700 px-2 py-0.5 rounded block mb-0.5">${t}</code>`).join('')}
        </div>` : ''}
      </div>

      <!-- Command Preview ────────────────────── -->
      <div class="bg-dark-800 rounded-xl p-4 border border-dark-600 mb-4">
        <div class="flex justify-between items-center mb-2">
          <h3 class="text-sm font-semibold">📋 Commande llama.cpp</h3>
          <button onclick="copyCommand()" class="text-xs px-2 py-1 bg-dark-600 hover:bg-dark-500 rounded transition">📋 Copier</button>
        </div>
        <pre id="commandPreview" class="text-xs text-gray-300 bg-dark-900 p-3 rounded-lg overflow-x-auto whitespace-pre-wrap max-h-40 overflow-y-auto scrollbar-thin">${suggestion.command_preview || 'Aucune commande'}</pre>
      </div>

      <!-- Actions ────────────────────────────── -->
      <div class="flex gap-3">
        <button onclick="launchChat('${modelId}')" class="flex-1 px-4 py-3 bg-blue-600 hover:bg-blue-700 rounded-xl font-semibold transition">▶️ Lancer le chat</button>
        <button onclick="navigate('terminal')" class="px-4 py-3 bg-dark-600 hover:bg-dark-500 rounded-xl text-sm transition">💬 Terminal</button>
      </div>
    `;
  } catch (e) {
    el.innerHTML = `<div class="text-red-400 py-8">❌ ${e.message}</div>`;
  }
}

function paramField({ label, value }) {
  return `
    <div class="bg-dark-900 rounded-lg p-2">
      <div class="text-xs text-gray-400">${label}</div>
      <div class="text-sm font-mono">${value}</div>
    </div>`;
}

function strategyLabel(s) {
  const labels = { 'moe_offload': 'MoE Offload', 'dense_full': 'Full GPU', 'dense_partial': 'Offload partiel' };
  return labels[s] || s;
}

function strategyBadge(s) {
  const colors = { 'moe_offload': 'bg-purple-900 text-purple-200', 'dense_full': 'bg-green-900 text-green-200', 'dense_partial': 'bg-yellow-900 text-yellow-200' };
  return colors[s] || 'bg-gray-700';
}

function strategyDescription(s, model) {
  switch (s) {
    case 'moe_offload':
      return `Le modèle MoE ${model.name || ''} est configuré pour l'offloading optimal : les couches d'attention restent sur GPU (compute-intensive), tandis que les experts sont délégués au CPU (memory-intensive). Résultat : performances quasi natives avec une empreinte VRAM réduite.`;
    case 'dense_full':
      return `Le modèle tient intégralement dans la VRAM. Aucun offloading nécessaire — performances maximales.`;
    case 'dense_partial':
      return `Le modèle est trop grand pour la VRAM disponible. Offloading partiel activé : certaines couches sont déléguées au CPU. Les performances seront réduites.`;
    default:
      return 'Configuration automatique basée sur les règles de la vidéo Codacus.';
  }
}

function copyCommand() {
  const pre = document.getElementById('commandPreview');
  navigator.clipboard.writeText(pre.textContent).then(() => flash('📋 Commande copiée !'));
}

async function launchChat(modelId) {
  window._chatModelId = modelId;
  window._chatMessages = [{ role: 'system', content: 'Tu es un assistant IA utile.' }];
  navigate('terminal');
  await renderTerminal();
}
