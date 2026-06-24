// API Client pour AI Runner
const API = '/api/v1';

// ─── Utils ───────────────────────────────────────────

async function apiGet(path) {
  const resp = await fetch(`${API}${path}`);
  if (!resp.ok) throw new Error(`GET ${path}: ${resp.status} ${resp.statusText}`);
  return resp.json();
}

async function apiPost(path, body = {}) {
  const resp = await fetch(`${API}${path}`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  });
  if (!resp.ok) {
    const err = await resp.json().catch(() => ({ detail: resp.statusText }));
    throw new Error(err.detail || `POST ${path}: ${resp.status}`);
  }
  return resp.json();
}

function apiStream(path, body, onEvent, onDone, onError) {
  fetch(`${API}${path}`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  }).then(async (resp) => {
    if (!resp.ok) {
      const err = await resp.json().catch(() => ({ detail: resp.statusText }));
      onError?.(err.detail || `Erreur ${resp.status}`);
      return;
    }
    const reader = resp.body.getReader();
    const decoder = new TextDecoder();
    let buffer = '';

    while (true) {
      const { done, value } = await reader.read();
      if (done) { onDone?.(); break; }
      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split('\n');
      buffer = lines.pop() || '';
      for (const line of lines) {
        if (line.startsWith('data: ')) {
          try {
            const event = JSON.parse(line.slice(6));
            onEvent?.(event);
          } catch (e) { /* ignore parse errors */ }
        }
      }
    }
  }).catch(err => onError?.(err.message));
}

// ─── System ──────────────────────────────────────────

async function getStatus() { return apiGet('/status'); }

// ─── Models ──────────────────────────────────────────

async function getModels() { return apiGet('/models'); }
async function scanModels() { return apiPost('/models/scan'); }
async function getModel(id) { return apiGet(`/models/${encodeURIComponent(id)}`); }
async function analyzeModel(id) { return apiPost(`/models/${encodeURIComponent(id)}/analyze`); }
async function deleteModel(id) { return fetch(`${API}/models/${encodeURIComponent(id)}`, { method: 'DELETE' }).then(r => r.json()); }
async function hfSearch(query) { return apiGet(`/models/hf-search?q=${encodeURIComponent(query)}&page=1`); }

// ─── Config ──────────────────────────────────────────

async function getConfigSuggestion(modelId, ctxSize = 8192, temp = 0.7) {
  return apiPost('/config/suggest', { model_id: modelId, ctx_size: ctxSize, temp });
}

// ─── llama.cpp ────────────────────────────────────────

async function getLlamacppVersion() { return apiGet('/llamacpp/version'); }
async function updateLlamacpp() { return apiPost('/llamacpp/update'); }


// ─── Chat ────────────────────────────────────────────

function chatStream(modelId, messages, params, onEvent, onDone, onError) {
  return apiStream('/chat', {
    model_id: modelId,
    messages,
    params: params || {},
    stream: true,
  }, onEvent, onDone, onError);
}

async function chatSync(modelId, messages, params = {}) {
  return apiPost('/chat', { model_id: modelId, messages, params, stream: false });
}

// ─── Control ─────────────────────────────────────────

async function stopRun() { return apiPost('/stop'); }
async function unloadModel() { return apiPost('/models/unload'); }
async function getHistory() { return apiGet('/history'); }

// ─── ComfyUI ─────────────────────────────────────────

async function comfyStatus() { return apiPost('/comfyui/status'); }
async function comfyRelease() { return apiPost('/comfyui/release'); }

// ─── GPU Indicator ───────────────────────────────────

async function updateGPUIndicator() {
  try {
    const status = await getStatus();
    const el = document.getElementById('gpuIndicator');
    if (status.mode === 'cuda' && status.gpu?.length) {
      const gpu = status.gpu[0];
      el.textContent = `${gpu.name} • ${gpu.vram_free_gb}G libre`;
      el.className = 'text-xs px-2 py-0.5 rounded-full bg-green-900 text-green-300';
    } else {
      el.textContent = 'CPU mode';
      el.className = 'text-xs px-2 py-0.5 rounded-full bg-dark-600 text-gray-400';
    }
  } catch (e) {
    console.warn('Status update failed:', e);
  }
}
