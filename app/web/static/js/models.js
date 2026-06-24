// ─── Models ──────────────────────────────────────────

let _hfResults = [];
let _hfSearchQuery = '';

async function renderModels() {
  const el = document.getElementById('page-models');
  el.innerHTML = `
    <div class="flex gap-2 mb-4">
      <button class="tab-btn px-4 py-2 rounded-lg bg-blue-600 text-sm" data-tab="local" onclick="switchModelsTab('local')">📂 Locaux</button>
      <button class="tab-btn px-4 py-2 rounded-lg bg-dark-600 text-sm" data-tab="hf" onclick="switchModelsTab('hf')">🌐 HuggingFace</button>
    </div>
    <div id="models-local" class="tab-content"></div>
    <div id="models-hf" class="tab-content hidden"></div>
    <div id="models-download-progress" class="hidden"></div>
  `;
  await renderLocalModels();
}

function switchModelsTab(tab) {
  document.querySelectorAll('.tab-content').forEach(el => el.classList.add('hidden'));
  document.getElementById(`models-${tab}`).classList.remove('hidden');
  document.querySelectorAll('.tab-btn').forEach(b => {
    b.classList.toggle('bg-blue-600', b.dataset.tab === tab);
    b.classList.toggle('bg-dark-600', b.dataset.tab !== tab);
  });
  if (tab === 'hf') renderHFModels();
}

// ─── Locaux ─────────────────────────────────────────

async function renderLocalModels() {
  const el = document.getElementById('models-local');
  el.innerHTML = '<p class="text-gray-500">⏳ Scan...</p>';

  try {
    const models = await scanModels();
    if (models.length === 0) {
      el.innerHTML = '<p class="text-gray-500">Aucun modèle local. Allez dans l\'onglet HuggingFace pour en télécharger.</p>';
      return;
    }

    el.innerHTML = `<div class="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-3">${models.map(m => localModelCard(m)).join('')}</div>`;
  } catch (e) {
    el.innerHTML = `<p class="text-red-400">❌ ${e.message}</p>`;
  }
}

function localModelCard(m) {
  const badge = m.is_moe ? '🔄 MoE' : '📐 Dense';
  return `
    <div class="bg-dark-800 rounded-xl p-4 border border-dark-600">
      <div class="font-semibold text-sm truncate">${m.name || m.id}</div>
      <div class="flex flex-wrap gap-1.5 mt-2">
        <span class="text-xs px-2 py-0.5 rounded bg-dark-600">${m.params_b}B</span>
        <span class="text-xs px-2 py-0.5 rounded bg-dark-600">${badge}</span>
        <span class="text-xs px-2 py-0.5 rounded bg-dark-600">${m.quant}</span>
        <span class="text-xs px-2 py-0.5 rounded bg-dark-600">${m.file_size_gb} GB</span>
      </div>
      ${m.architecture ? `<div class="text-xs text-gray-500 mt-2">${m.architecture} • ${m.block_count} layers</div>` : ''}
      <div class="flex gap-2 mt-3">
        <button onclick="navigate('config'); window._configModelId = '${m.id}'" class="flex-1 px-3 py-1.5 bg-blue-600 hover:bg-blue-700 rounded-lg text-xs transition">⚙️ Config</button>
        <button onclick="deleteModelConfirm('${m.id}')" class="px-3 py-1.5 bg-red-800 hover:bg-red-700 rounded-lg text-xs transition">🗑️</button>
      </div>
    </div>`;
}

async function deleteModelConfirm(id) {
  if (!confirm(`Supprimer le modèle ${id} ?`)) return;
  try {
    await deleteModel(id);
    flash('✅ Modèle supprimé');
    renderLocalModels();
  } catch (e) {
    flash('❌ ' + e.message);
  }
}

// ─── HuggingFace ────────────────────────────────────

async function renderHFModels() {
  const el = document.getElementById('models-hf');
  el.innerHTML = `
    <div class="flex gap-2 mb-4">
      <input id="hfSearchInput" type="text" placeholder="Rechercher (ex: qwen GGUF, llama GGUF...)"
             class="flex-1 bg-dark-800 border border-dark-600 rounded-lg px-4 py-2 text-sm focus:outline-none focus:border-blue-500"
             value="${_hfSearchQuery}" onkeydown="if(event.key==='Enter') doHFSearch()"/>
      <button onclick="doHFSearch()" class="px-4 py-2 bg-blue-600 hover:bg-blue-700 rounded-lg text-sm">🔍</button>
    </div>
    <div id="hf-results" class="space-y-2"></div>
  `;

  if (_hfSearchQuery) doHFSearch();
}

async function doHFSearch() {
  const input = document.getElementById('hfSearchInput');
  const query = input.value.trim();
  if (!query) return;

  _hfSearchQuery = query;
  const el = document.getElementById('hf-results');
  el.innerHTML = '<p class="text-gray-500">⏳ Recherche...</p>';

  try {
    const data = await hfSearch(query);
    _hfResults = data.results || [];

    if (_hfResults.length === 0) {
      el.innerHTML = '<p class="text-gray-500">Aucun résultat. Essayez une autre recherche.</p>';
      return;
    }

    el.innerHTML = _hfResults.map(r => hfResultCard(r)).join('');
  } catch (e) {
    el.innerHTML = `<p class="text-red-400">❌ ${e.message}</p>`;
  }
}

function hfResultCard(r) {
  const ggufFiles = (r.files || []).filter(f => f.name.endsWith('.gguf') && !f.name.includes('-00001-of-'));
  const sizeDisplay = ggufFiles.length > 0
    ? ggufFiles.map(f => `<option value="${f.name}">${f.name} (${(f.size / 1e9).toFixed(1)} GB)</option>`).join('')
    : '<option>Chargement...</option>';

  return `
    <div class="bg-dark-800 rounded-xl p-4 border border-dark-600">
      <div class="flex justify-between items-start">
        <div>
          <div class="font-semibold text-sm">${r.name}</div>
          <div class="text-xs text-gray-400">${r.repo_id}</div>
        </div>
        <div class="text-xs text-gray-500">⬇ ${(r.downloads / 1000).toFixed(0)}K</div>
      </div>
      <div class="mt-3 flex gap-2 flex-wrap">
        <select id="file-select-${r.repo_id.replace(/\//g, '-')}" class="flex-1 bg-dark-700 border border-dark-600 rounded px-2 py-1 text-xs">
          ${sizeDisplay}
        </select>
        <button onclick="downloadHFModel('${r.repo_id}', '${r.repo_id.replace(/\//g, '-')}')"
                class="px-3 py-1.5 bg-green-700 hover:bg-green-600 rounded-lg text-xs transition">⬇ Télécharger</button>
      </div>
      <div id="progress-${r.repo_id.replace(/\//g, '-')}" class="mt-2 hidden"></div>
    </div>`;
}

function downloadHFModel(repoId, safeId) {
  const select = document.getElementById(`file-select-${safeId}`);
  const filename = select.value;
  if (!filename) return;

  const progressEl = document.getElementById(`progress-${safeId}`);
  progressEl.classList.remove('hidden');
  progressEl.innerHTML = `
    <div class="h-2 bg-dark-700 rounded-full overflow-hidden">
      <div id="pbar-${safeId}" class="h-full bg-green-500 rounded-full progress-bar" style="width:0%"></div>
    </div>
    <div id="pinfo-${safeId}" class="text-xs text-gray-400 mt-1">0 / ? MB — 0 MB/s</div>`;

  // SSE stream for download progress
  const source = new EventSource('/api/v1/models/hf-search'); // dummy, we use fetch instead
  // Since we can't use EventSource with POST, we use the fetch-based SSE approach
  fetch(`${API}/models/hf-download`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ repo_id: repoId, filename }),
  }).then(async (resp) => {
    if (!resp.ok) throw new Error(`Erreur ${resp.status}`);
    const reader = resp.body.getReader();
    const decoder = new TextDecoder();
    let buffer = '';

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split('\n');
      buffer = lines.pop() || '';
      for (const line of lines) {
        if (line.startsWith('data: ')) {
          try {
            const ev = JSON.parse(line.slice(6));
            if (ev.type === 'progress') {
              const pct = ev.percent || 0;
              document.getElementById(`pbar-${safeId}`).style.width = pct + '%';
              const mb = (ev.downloaded / 1e6).toFixed(0);
              const total = (ev.total_bytes / 1e6).toFixed(0);
              document.getElementById(`pinfo-${safeId}`).textContent =
                `${mb} / ${total} MB — ${ev.speed_gbps || 0} MB/s — ${ev.eta_s || '?'}s`;
            } else if (ev.type === 'done') {
              document.getElementById(`pbar-${safeId}`).style.width = '100%';
              document.getElementById(`pinfo-${safeId}`).textContent = '✅ Terminé !';
              flash('✅ Modèle téléchargé !');
              renderLocalModels();
            } else if (ev.type === 'error') {
              document.getElementById(`pinfo-${safeId}`).textContent = `❌ ${ev.message}`;
            }
          } catch (e) { /* ignore */ }
        }
      }
    }
  }).catch(err => {
    document.getElementById(`pinfo-${safeId}`).textContent = `❌ ${err.message}`;
  });
}
