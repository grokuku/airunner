// ─── Benchmark — Auto-config et tests de performance ──

let _benchmarkRunning = false;
let _benchmarkResults = [];

async function renderBenchmark() {
  const el = document.getElementById('page-benchmark');

  // Charger les modèles disponibles
  let models = [];
  try { models = await getModels(); } catch (e) { /* ignore */ }

  const modelId = window._configModelId || (models.length > 0 ? models[0].id : '');

  el.innerHTML = `
    <div class="flex flex-col gap-4">
      <!-- Controls -->
      <div class="bg-dark-800 rounded-xl border border-dark-600 p-4">
        <div class="flex items-center gap-4 flex-wrap">
          <div class="flex-1 min-w-[200px]">
            <label class="text-xs text-gray-400 mb-1 block">Modèle</label>
            <select id="benchmarkModel" class="w-full bg-dark-700 border border-dark-600 rounded-lg px-3 py-2 text-sm">
              ${models.map(m => `<option value="${m.id}" ${m.id === modelId ? 'selected' : ''}>${m.name || m.id} (${m.params_b}B)</option>`).join('')}
              ${models.length === 0 ? '<option value="">Aucun modèle disponible</option>' : ''}
            </select>
          </div>
          <div>
            <label class="text-xs text-gray-400 mb-1 block">Priorité</label>
            <select id="benchmarkPriority" class="bg-dark-700 border border-dark-600 rounded-lg px-3 py-2 text-sm">
              <option value="speed">⚡ Speed (max tok/s)</option>
              <option value="quality">🎯 Quality (max marge VRAM)</option>
            </select>
          </div>
          <div>
            <label class="text-xs text-gray-400 mb-1 block">Contexte</label>
            <select id="benchmarkCtxSize" class="bg-dark-700 border border-dark-600 rounded-lg px-3 py-2 text-sm">
              <option value="2048">2 048</option>
              <option value="4096">4 096</option>
              <option value="8192" selected>8 192</option>
              <option value="16384">16 384</option>
              <option value="32768">32 768</option>
              <option value="65536">65 536</option>
            </select>
          </div>
          <div>
            <label class="text-xs text-gray-400 mb-1 block">Cache KV</label>
            <select id="benchmarkCacheType" class="bg-dark-700 border border-dark-600 rounded-lg px-3 py-2 text-sm">
              <option value="auto">🔄 Auto (teste les 2)</option>
              <option value="q8_0">📌 Q8 (meilleure qualité)</option>
              <option value="q4_0">📌 Q4 (moins de VRAM)</option>
            </select>
          </div>
          <div>
            <label class="text-xs text-gray-400 mb-1 block">Flash Attn</label>
            <select id="benchmarkFlashAttn" class="bg-dark-700 border border-dark-600 rounded-lg px-3 py-2 text-sm">
              <option value="auto">🔄 Auto (teste on/off)</option>
              <option value="on">📌 On</option>
              <option value="off">📌 Off</option>
            </select>
          </div>
          <div class="pt-5">
            <button id="benchmarkStartBtn" onclick="startBenchmark()"
                    class="px-5 py-2 bg-blue-600 hover:bg-blue-700 disabled:bg-dark-600 disabled:cursor-not-allowed rounded-lg font-semibold transition text-sm"
                    ${models.length === 0 || _benchmarkRunning ? 'disabled' : ''}>
              🚀 Lancer l'auto-config
            </button>
            <button id="benchmarkStopBtn" onclick="stopBenchmark()"
                    class="hidden px-4 py-2 bg-red-700 hover:bg-red-600 rounded-lg transition text-sm ml-2">⏹ Arrêter</button>
          </div>
        </div>
      </div>

      <!-- Progression -->
      <div id="benchmarkProgress" class="hidden">
        <div class="bg-dark-800 rounded-xl border border-dark-600 p-4">
          <div class="flex justify-between items-center mb-2">
            <span class="text-sm font-semibold" id="benchmarkProgressLabel">⏳ Test en cours...</span>
            <span class="text-xs text-gray-400" id="benchmarkProgressCount">0/0</span>
          </div>
          <div class="h-2 bg-dark-700 rounded-full overflow-hidden">
            <div id="benchmarkProgressBar" class="h-full bg-blue-500 rounded-full progress-bar" style="width:0%"></div>
          </div>
          <div id="benchmarkCurrentConfig" class="mt-2 text-xs text-gray-400"></div>
        </div>
      </div>

      <!-- Résultats -->
      <div id="benchmarkResults" class="hidden">
        <div class="bg-dark-800 rounded-xl border border-dark-600 p-4">
          <div class="flex justify-between items-center mb-3">
            <h3 class="text-sm font-semibold">📊 Résultats</h3>
            <button id="benchmarkSaveBtn" onclick="saveBestConfig()"
                    class="hidden px-3 py-1.5 bg-green-700 hover:bg-green-600 rounded-lg text-xs transition">💾 Sauvegarder comme preset</button>
          </div>
          <div id="benchmarkBestBadge" class="hidden mb-3 p-3 bg-green-900/30 border border-green-700/50 rounded-lg">
            <div class="text-xs text-green-300 font-semibold mb-1">🏆 Meilleure configuration</div>
            <div id="benchmarkBestInfo" class="text-sm"></div>
          </div>
          <div class="overflow-x-auto">
            <table class="w-full text-xs">
              <thead>
                <tr class="text-gray-400 border-b border-dark-600">
                  <th class="text-left py-2 px-2">Configuration</th>
                  <th class="text-right py-2 px-2">⚡ tok/s</th>
                  <th class="text-right py-2 px-2" title="VRAM mesurée">🎮 VRAM réel</th>
                  <th class="text-right py-2 px-2" title="VRAM estimée par le moteur de règles">📐 VRAM estimé</th>
                  <th class="text-right py-2 px-2">🎯 Écart</th>
                  <th class="text-right py-2 px-2">🧠 RAM</th>
                  <th class="text-right py-2 px-2">⭐ Score</th>
                  <th class="text-left py-2 px-2"></th>
                </tr>
              </thead>
              <tbody id="benchmarkResultsBody">
              </tbody>
            </table>
          </div>
        </div>
      </div>

      <!-- Historique des benchmarks sauvegardés -->
      <div id="benchmarkHistory" class="bg-dark-800 rounded-xl border border-dark-600 p-4">
        <h3 class="text-sm font-semibold mb-2">📜 Presets sauvegardés</h3>
        <div id="benchmarkPresetsList" class="text-sm text-gray-500">
          ⏳ Chargement...
        </div>
      </div>
    </div>
  `;

  loadSavedPresets();
}

// ─── Benchmark Runner ──────────────────────────────

async function startBenchmark() {
  const modelId = document.getElementById('benchmarkModel').value;
  const priority = document.getElementById('benchmarkPriority').value;
  const ctxSize = document.getElementById('benchmarkCtxSize').value;
  const cacheType = document.getElementById('benchmarkCacheType').value;
  const flashAttn = document.getElementById('benchmarkFlashAttn').value;
  if (!modelId) return flash('❌ Sélectionnez un modèle');

  _benchmarkRunning = true;
  _benchmarkResults = [];
  document.getElementById('benchmarkStartBtn').disabled = true;
  document.getElementById('benchmarkStopBtn').classList.remove('hidden');
  document.getElementById('benchmarkProgress').classList.remove('hidden');
  document.getElementById('benchmarkResults').classList.add('hidden');
  document.getElementById('benchmarkBestBadge').classList.add('hidden');
  document.getElementById('benchmarkSaveBtn').classList.add('hidden');
  document.getElementById('benchmarkResultsBody').innerHTML = '';

  const url = `/api/v1/benchmark/auto?model_id=${encodeURIComponent(modelId)}&priority=${priority}&ctx_size=${ctxSize}&cache_type=${cacheType}&flash_attn=${flashAttn}`;

  try {
    const resp = await fetch(url, { method: 'POST' });
    if (!resp.ok) {
      const err = await resp.json().catch(() => ({ detail: resp.statusText }));
      throw new Error(err.detail || `Erreur ${resp.status}`);
    }

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
            const event = JSON.parse(line.slice(6));
            handleBenchmarkEvent(event);
          } catch (e) { /* ignore parse errors */ }
        }
      }
    }
  } catch (e) {
    flash('❌ ' + e.message);
  }

  _benchmarkRunning = false;
  document.getElementById('benchmarkStartBtn').disabled = false;
  document.getElementById('benchmarkStopBtn').classList.add('hidden');
  document.getElementById('benchmarkProgressBar').style.width = '100%';
  document.getElementById('benchmarkProgressLabel').textContent = _benchmarkResults.length > 0 ? '✅ Terminé' : '❌ Échoué';
}

function stopBenchmark() {
  _benchmarkRunning = false;
  document.getElementById('benchmarkStartBtn').disabled = false;
  document.getElementById('benchmarkStopBtn').classList.add('hidden');
  document.getElementById('benchmarkProgressLabel').textContent = '⏹ Arrêté';
}

function handleBenchmarkEvent(event) {
  switch (event.type) {
    case 'start':
      document.getElementById('benchmarkProgressCount').textContent = `0/${event.total}`;
      break;

    case 'progress':
      const pct = Math.round((event.current / event.total) * 100);
      document.getElementById('benchmarkProgressBar').style.width = pct + '%';
      document.getElementById('benchmarkProgressCount').textContent = `${event.current}/${event.total}`;
      document.getElementById('benchmarkProgressLabel').textContent = `⏳ Test ${event.current}/${event.total}`;
      document.getElementById('benchmarkCurrentConfig').textContent = `🔧 ${event.config.label || 'Config ' + event.current}`;
      break;

    case 'result':
      addBenchmarkResult(event);
      break;

    case 'best':
      showBenchmarkBest(event);
      break;

    case 'error':
      flash('❌ ' + (event.message || 'Erreur benchmark'));
      break;

    case 'done':
      document.getElementById('benchmarkProgressLabel').textContent = '✅ Terminé';
      break;
  }
}

function addBenchmarkResult(event) {
  _benchmarkResults.push(event);
  const tbody = document.getElementById('benchmarkResultsBody');
  document.getElementById('benchmarkResults').classList.remove('hidden');

  const tok_s = event.tok_s || 0;
  const vram = event.vram_gb || 0;
  const estimate = event.estimate_vram_gb || 0;
  const diff = event.diff_pct || 0;
  const ram = event.ram_gb || 0;
  const score = event.score || 0;
  const hasError = !!event.error;

  // Formater l'écart estimé vs réel
  let diffText = '—';
  let diffClass = 'text-gray-500';
  if (estimate > 0 && vram > 0) {
    diffText = (diff > 0 ? '+' : '') + diff + '%';
    diffClass = diff > 20 ? 'text-red-400' : diff > 5 ? 'text-yellow-400' : 'text-green-400';
  }

  const row = document.createElement('tr');
  row.className = 'border-b border-dark-700/50';
  row.innerHTML = `
    <td class="py-2 px-2">
      <div class="flex items-center gap-2">
        <span class="font-medium ${hasError ? 'text-red-400' : ''}">${event.config?.label || '—'}</span>
        ${hasError ? `<span class="text-xs text-red-500" title="${event.error}">⚠️</span>` : ''}
      </div>
    </td>
    <td class="text-right py-2 px-2 font-mono ${tok_s > 0 ? 'text-green-300' : 'text-gray-500'}">${tok_s > 0 ? tok_s : '—'}</td>
    <td class="text-right py-2 px-2 font-mono">${vram > 0 ? vram.toFixed(1) + 'G' : '—'}</td>
    <td class="text-right py-2 px-2 font-mono text-gray-400">${estimate > 0 ? estimate.toFixed(1) + 'G' : '—'}</td>
    <td class="text-right py-2 px-2 font-mono ${diffClass}">${diffText}</td>
    <td class="text-right py-2 px-2 font-mono">${ram > 0 ? ram.toFixed(1) + 'G' : '—'}</td>
    <td class="text-right py-2 px-2 font-mono ${score > 80 ? 'text-green-300' : score > 50 ? 'text-yellow-300' : 'text-gray-400'}">${score > 0 ? score : '—'}</td>
    <td class="py-2 px-2">${hasError ? `<span class="text-xs text-red-500">${event.error?.slice(0, 40)}</span>` : ''}</td>
  `;
  tbody.appendChild(row);
}

function showBenchmarkBest(event) {
  const badge = document.getElementById('benchmarkBestBadge');
  const saveBtn = document.getElementById('benchmarkSaveBtn');
  badge.classList.remove('hidden');
  saveBtn.classList.remove('hidden');
  document.getElementById('benchmarkBestInfo').innerHTML = `
    <div class="flex gap-4 flex-wrap">
      <span>⚡ <strong>${event.tok_s || 0}</strong> tok/s</span>
      <span>🎮 <strong>${(event.vram_gb || 0).toFixed(1)}</strong> Go VRAM</span>
      <span>🧠 <strong>${(event.ram_gb || 0).toFixed(1)}</strong> Go RAM</span>
      <span class="text-green-300">Config: <strong>${event.label || '—'}</strong></span>
    </div>
  `;
}

async function saveBestConfig() {
  const modelId = document.getElementById('benchmarkModel').value;
  try {
    const resp = await fetch(`/api/v1/benchmark/save-preset?model_id=${encodeURIComponent(modelId)}&label=optimized`, {
      method: 'POST',
    });
    if (!resp.ok) throw new Error((await resp.json()).detail);
    const data = await resp.json();
    flash('✅ Preset sauvegardé !');
    loadSavedPresets();
  } catch (e) {
    flash('❌ ' + e.message);
  }
}

async function loadSavedPresets() {
  // Pour l'instant, on liste les fichiers .json dans /api/.../presets/
  // Simple fallback : le backend n'a pas d'endpoint list-presets,
  // on vérifie juste si le preset existe via l'API history
  const el = document.getElementById('benchmarkPresetsList');
  try {
    const history = await getHistory();
    const runs = history.runs || [];
    if (runs.length === 0) {
      el.innerHTML = '<p class="text-gray-500 text-xs">Aucun preset sauvegardé. Lancez un benchmark puis cliquez sur 💾 Sauvegarder.</p>';
    } else {
      el.innerHTML = `
        <div class="text-xs text-gray-400 mb-2">Historique des runs :</div>
        <div class="space-y-1 max-h-40 overflow-y-auto scrollbar-thin">
          ${runs.slice(0, 10).map(r => `
            <div class="bg-dark-700/30 rounded px-3 py-1.5 text-xs flex justify-between">
              <span>${r.model_id}</span>
              <span class="text-gray-400">${r.tokens_generated || 0} tok • ${r.avg_speed || '?'} tok/s</span>
            </div>
          `).join('')}
        </div>`;
    }
  } catch (e) {
    el.innerHTML = `<p class="text-red-400 text-xs">❌ ${e.message}</p>`;
  }
}
