// ─── Dashboard ───────────────────────────────────────

async function renderDashboard() {
  const el = document.getElementById('page-dashboard');
  el.innerHTML = '<div class="text-center py-12 text-gray-400">⏳ Chargement...</div>';

  try {
    const [status, models, llVersion] = await Promise.all([
      getStatus(), getModels(),
      getLlamacppVersion().catch(() => ({ installed: null, latest: null }))
    ]);

    el.innerHTML = `
      <!-- llama.cpp Version -->
      ${renderLlamacppVersion(llVersion)}

      <!-- System Cards -->
      <div class="grid grid-cols-1 md:grid-cols-3 gap-4 mb-6">
        ${renderGPUCard(status)}
        ${renderRAMCard(status)}
        ${renderCPUCard(status)}
      </div>

      <!-- Models Grid -->
      <h2 class="text-lg font-semibold mb-3">📦 Modèles locaux (${models.length})</h2>
      <div class="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-3 mb-6">
        ${models.length === 0 ? '<p class="text-gray-500 col-span-full">Aucun modèle. Allez dans l\'onglet Modèles pour en télécharger.</p>' : ''}
        ${models.map(m => modelCard(m)).join('')}
      </div>

      <!-- Quick actions -->
      <h2 class="text-lg font-semibold mb-3">⚡ Actions rapides</h2>
      <div class="flex gap-3 flex-wrap">
        <button onclick="navigate('models')" class="px-4 py-2 bg-blue-600 hover:bg-blue-700 rounded-lg text-sm transition">📥 Télécharger un modèle</button>
        <button onclick="scanModels().then(() => renderDashboard())" class="px-4 py-2 bg-dark-600 hover:bg-dark-500 rounded-lg text-sm transition">🔄 Scanner les modèles</button>
        <button onclick="navigate('terminal')" class="px-4 py-2 bg-purple-600 hover:bg-purple-700 rounded-lg text-sm transition">💬 Ouvrir le terminal</button>
        <button onclick="unloadModel().then(r => flash('VRAM libérée'))" class="px-4 py-2 bg-red-800 hover:bg-red-700 rounded-lg text-sm transition">🗑️ Libérer VRAM</button>
      </div>
    `;
  } catch (e) {
    el.innerHTML = `<div class="text-red-400 py-8">❌ Erreur: ${e.message}</div>`;
  }
}

function renderLlamacppVersion(v) {
  const installed = v.installed || '❌ Non installé';
  const latest = v.latest || '—';
  const hasUpdate = v.update_available;
  const isInstalled = !!v.installed;

  return `
    <div class="bg-dark-800 rounded-xl p-3 border border-dark-600 mb-4 flex items-center justify-between">
      <div class="flex items-center gap-3">
        <span class="text-lg">🦙</span>
        <div>
          <div class="text-xs text-gray-400">llama.cpp</div>
          <div class="text-sm font-mono">
            Installé: <span class="${isInstalled ? 'text-green-300' : 'text-red-400'}">${installed}</span>
            ${latest !== '—' ? `&nbsp;|&nbsp; Dernier: <span class="text-blue-300">${latest}</span>` : ''}
          </div>
        </div>
      </div>
      <div>
        ${!isInstalled ? `
          <button onclick="downloadLlamacpp()" class="px-3 py-1.5 bg-green-700 hover:bg-green-600 rounded-lg text-xs transition">
            ⬇ Télécharger
          </button>
        ` : ''}
        ${hasUpdate ? `
          <button onclick="downloadLlamacpp()" class="px-3 py-1.5 bg-yellow-700 hover:bg-yellow-600 rounded-lg text-xs transition">
            🔄 Mettre à jour (${v.latest})
          </button>
        ` : ''}
        ${isInstalled && !hasUpdate ? `
          <span class="text-xs text-green-400">✅ À jour</span>
        ` : ''}
      </div>
    </div>`;
}

async function downloadLlamacpp() {
  const btn = event?.target || document.querySelector('[onclick*="downloadLlamacpp"]');
  if (btn) { btn.textContent = '⏳ Téléchargement...'; btn.disabled = true; }
  try {
    const result = await updateLlamacpp();
    flash('✅ llama.cpp ' + (result.current_version || '') + ' installé !');
    renderDashboard();
  } catch (e) {
    flash('❌ ' + e.message);
    if (btn) { btn.textContent = '⬇ Télécharger'; btn.disabled = false; }
  }
}


function renderGPUCard(status) {
  const gpus = status.gpu ?? [];
  if (status.mode === 'cuda' && gpus.length) {
    const gpuCount = gpus.length;
    const combinedTotal = gpus.reduce((s, g) => s + g.vram_total_gb, 0);
    const combinedUsed = gpus.reduce((s, g) => s + g.vram_used_gb, 0);
    const combinedPct = combinedTotal > 0 ? ((combinedUsed / combinedTotal) * 100).toFixed(0) : 0;

    // Carte combinée (première ligne)
    let html = `
      <div class="bg-dark-800 rounded-xl p-4 border border-dark-600">
        <div class="text-xs text-gray-400 mb-1">🎮 GPU${gpuCount > 1 ? ` (×${gpuCount})` : ''}</div>
        ${gpuCount > 1 ? `
          <div class="font-semibold text-sm">${combinedTotal.toFixed(0)}G combinée</div>
          <div class="mt-2 flex justify-between text-xs text-gray-400">
            <span>VRAM totale</span>
            <span>${combinedUsed.toFixed(1)}G / ${combinedTotal.toFixed(0)}G</span>
          </div>
          <div class="mt-1 h-2 bg-dark-700 rounded-full overflow-hidden">
            <div class="h-full bg-blue-500 rounded-full progress-bar" style="width:${combinedPct}%"></div>
          </div>
        ` : `
          <div class="font-semibold text-sm truncate">${gpus[0].name}</div>
          <div class="mt-2 flex justify-between text-xs text-gray-400">
            <span>VRAM</span>
            <span>${gpus[0].vram_used_gb}G / ${gpus[0].vram_total_gb}G</span>
          </div>
          <div class="mt-1 h-2 bg-dark-700 rounded-full overflow-hidden">
            <div class="h-full bg-blue-500 rounded-full progress-bar" style="width:${combinedPct}%"></div>
          </div>
        `}
      </div>`;

    // Cartes individuelles si multi-GPU
    if (gpuCount > 1) {
      html += gpus.map(g => {
        const pct = g.vram_total_gb > 0 ? ((g.vram_used_gb / g.vram_total_gb) * 100).toFixed(0) : 0;
        return `
          <div class="bg-dark-800 rounded-xl p-3 border border-dark-600">
            <div class="text-xs text-gray-400 mb-1">🎮 GPU ${g.index}</div>
            <div class="font-semibold text-xs truncate">${g.name}</div>
            <div class="mt-1 flex justify-between text-xs text-gray-400">
              <span>VRAM</span>
              <span>${g.vram_used_gb}G / ${g.vram_total_gb}G</span>
            </div>
            <div class="mt-1 h-1.5 bg-dark-700 rounded-full overflow-hidden">
              <div class="h-full bg-cyan-500 rounded-full progress-bar" style="width:${pct}%"></div>
            </div>
          </div>`;
      }).join('');
    }

    return html;
  }
  return `
    <div class="bg-dark-800 rounded-xl p-4 border border-dark-600">
      <div class="text-xs text-gray-400 mb-1">🎮 GPU</div>
      <div class="text-sm text-gray-500">Aucun GPU détecté</div>
    </div>`;
}

function renderRAMCard(status) {
  const ram = status.ram;
  const pct = ram.total_gb > 0 ? ((1 - ram.available_gb / ram.total_gb) * 100).toFixed(0) : 0;
  return `
    <div class="bg-dark-800 rounded-xl p-4 border border-dark-600">
      <div class="text-xs text-gray-400 mb-1">🧠 RAM</div>
      <div class="font-semibold">${ram.available_gb}G libre</div>
      <div class="mt-2 flex justify-between text-xs text-gray-400">
        <span>Usage</span>
        <span>${pct}%</span>
      </div>
      <div class="mt-1 h-2 bg-dark-700 rounded-full overflow-hidden">
        <div class="h-full bg-green-500 rounded-full progress-bar" style="width:${pct}%"></div>
      </div>
    </div>`;
}

function renderCPUCard(status) {
  return `
    <div class="bg-dark-800 rounded-xl p-4 border border-dark-600">
      <div class="text-xs text-gray-400 mb-1">⚙️ CPU</div>
      <div class="font-semibold">${status.cpu.cores}c/${status.cpu.threads}t</div>
      <div class="text-xs text-gray-400 mt-1 truncate">${status.cpu.model}</div>
    </div>`;
}

function modelCard(m) {
  const badge = m.is_moe ? '🔄 MoE' : '📐 Dense';
  const quant = m.quant || '?';
  return `
    <div class="bg-dark-800 rounded-xl p-3 border border-dark-600 hover:border-blue-500/50 transition cursor-pointer"
         onclick="navigate('config'); window._configModelId = '${m.id}'">
      <div class="text-xs text-gray-400 truncate">${m.name || m.id}</div>
      <div class="flex gap-2 mt-2">
        <span class="text-xs px-1.5 py-0.5 rounded bg-dark-600">${m.params_b}B</span>
        <span class="text-xs px-1.5 py-0.5 rounded bg-dark-600">${badge}</span>
        <span class="text-xs px-1.5 py-0.5 rounded bg-dark-600">${quant}</span>
      </div>
      <div class="text-xs text-gray-500 mt-2">${m.file_size_gb} GB</div>
    </div>`;
}

function flash(msg) {
  const el = document.createElement('div');
  el.className = 'fixed bottom-4 right-4 bg-green-800 text-green-200 px-4 py-2 rounded-lg shadow-lg z-50 text-sm animate-bounce';
  el.textContent = msg;
  document.body.appendChild(el);
  setTimeout(() => el.remove(), 2000);
}
