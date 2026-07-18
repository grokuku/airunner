// ─── Auth Token management (Bearer token for /api/v1/*) ───────────────
// Le token est stocké côté client (localStorage) et envoyé via l'en-tête
// Authorization: Bearer {token} sur chaque appel API.
// Si aucun token n'est stocké, aucun en-tête n'est ajouté (auth désactivée).

const AUTH_TOKEN_KEY = 'ai_runner_auth_token';

/**
 * Retourne le token d'auth stocké dans localStorage, ou null.
 */
function getAuthToken() {
  try {
    return localStorage.getItem(AUTH_TOKEN_KEY) || null;
  } catch (e) {
    console.warn('localStorage unavailable:', e);
    return null;
  }
}

/**
 * Sauvegarde le token d'auth dans localStorage.
 * Passer une chaîne vide ou null pour effacer.
 */
function setAuthToken(token) {
  try {
    if (token) {
      localStorage.setItem(AUTH_TOKEN_KEY, token);
    } else {
      localStorage.removeItem(AUTH_TOKEN_KEY);
    }
  } catch (e) {
    console.warn('localStorage unavailable:', e);
  }
  updateAuthIndicator();
}

/**
 * Supprime le token d'auth de localStorage.
 */
function clearAuthToken() {
  try {
    localStorage.removeItem(AUTH_TOKEN_KEY);
  } catch (e) {
    console.warn('localStorage unavailable:', e);
  }
  updateAuthIndicator();
}

/**
 * Retourne true si un token est configuré.
 */
function hasAuthToken() {
  return !!getAuthToken();
}

/**
 * Construit l'objet Headers avec l'en-tête Authorization si un token est présent.
 * Fusionne avec les headers existants passés en argument.
 */
function withAuthHeaders(extra = {}) {
  const headers = { ...extra };
  const token = getAuthToken();
  if (token) {
    headers['Authorization'] = `Bearer ${token}`;
  }
  return headers;
}

/**
 * Gère une réponse 401: affiche un message et propose d'aller configurer le token.
 */
function handle401() {
  console.warn('API returned 401 — authentification requise.');
  // Afficher un message non-bloquant en haut de la page
  let banner = document.getElementById('auth401Banner');
  if (!banner) {
    banner = document.createElement('div');
    banner.id = 'auth401Banner';
    banner.className = 'fixed top-0 left-0 right-0 z-[100] bg-red-900/95 text-red-100 text-sm px-4 py-2 flex items-center justify-between shadow-lg';
    banner.innerHTML = `
      <span>🔐 Authentification requise — configurez votre Auth Token dans les Settings.</span>
      <div class="flex gap-2">
        <button onclick="navigate('settings'); document.getElementById('auth401Banner')?.remove();"
          class="px-3 py-1 bg-red-700 hover:bg-red-600 rounded text-xs font-semibold">Ouvrir Settings</button>
        <button onclick="document.getElementById('auth401Banner')?.remove();"
          class="px-3 py-1 bg-dark-700 hover:bg-dark-600 rounded text-xs">✕</button>
      </div>
    `;
    document.body.prepend(banner);
  }
}

/**
 * Met à jour l'indicateur visuel d'auth dans la barre de navigation.
 */
function updateAuthIndicator() {
  const el = document.getElementById('authIndicator');
  if (!el) return;
  if (hasAuthToken()) {
    el.textContent = '🔓';
    el.title = 'Auth token configuré';
    el.classList.remove('text-gray-500');
    el.classList.add('text-green-400');
  } else {
    el.textContent = '🔒';
    el.title = 'Aucun token — auth désactivée';
    el.classList.remove('text-green-400');
    el.classList.add('text-gray-500');
  }
}

// Mettre à jour l'indicateur au chargement
document.addEventListener('DOMContentLoaded', updateAuthIndicator);