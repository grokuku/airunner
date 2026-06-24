// ─── Benchmark — Tests de performance (à venir) ──────

async function renderBenchmark() {
  const el = document.getElementById('page-benchmark');

  el.innerHTML = `
    <div class="max-w-3xl mx-auto">
      <div class="bg-dark-800 rounded-xl border border-dark-600 p-6 mb-4">
        <h2 class="text-lg font-semibold mb-2">📊 Benchmark — À venir</h2>
        <p class="text-sm text-gray-400 mb-4">
          Cette page permettra de tester automatiquement différentes configurations
          pour trouver les meilleurs paramètres pour votre modèle et votre matériel.
        </p>

        <!-- Roadmap des benchmarks -->
        <div class="space-y-3">
          <div class="bg-dark-700/50 rounded-lg p-3 border border-dark-600 opacity-50">
            <div class="flex justify-between">
              <span class="text-sm font-medium">🚀 Débit (tok/s)</span>
              <span class="text-xs text-yellow-400">🔧 Planifié</span>
            </div>
            <p class="text-xs text-gray-500 mt-1">
              Teste le nombre de tokens par seconde pour différentes valeurs de
              <code class="text-blue-300">-ngl</code>, <code class="text-blue-300">threads</code>, et <code class="text-blue-300">batch-size</code>.
              Permet de trouver le meilleur équilibre GPU/CPU.
            </p>
          </div>

          <div class="bg-dark-700/50 rounded-lg p-3 border border-dark-600 opacity-50">
            <div class="flex justify-between">
              <span class="text-sm font-medium">🧠 Empreinte VRAM</span>
              <span class="text-xs text-yellow-400">🔧 Planifié</span>
            </div>
            <p class="text-xs text-gray-500 mt-1">
              Mesure la consommation VRAM réelle pour chaque configuration :
              poids du modèle, cache KV, overhead. Compare avec les estimations du moteur de règles.
            </p>
          </div>

          <div class="bg-dark-700/50 rounded-lg p-3 border border-dark-600 opacity-50">
            <div class="flex justify-between">
              <span class="text-sm font-medium">⚖️ Comparaison de quants</span>
              <span class="text-xs text-yellow-400">🔧 Planifié</span>
            </div>
            <p class="text-xs text-gray-500 mt-1">
              Compare Q4_K_M vs Q5_K_M vs Q8_0 sur la même requête :
              qualité de la réponse, vitesse, consommation VRAM.
            </p>
          </div>

          <div class="bg-dark-700/50 rounded-lg p-3 border border-dark-600 opacity-50">
            <div class="flex justify-between">
              <span class="text-sm font-medium">🎯 Auto-optimisation</span>
              <span class="text-xs text-yellow-400">🔧 Planifié</span>
            </div>
            <p class="text-xs text-gray-500 mt-1">
              Exécute automatiquement une série de benchmarks pour trouver
              la configuration optimale. Résultat directement applicable.
            </p>
          </div>
        </div>
      </div>

      <!-- Liste des benchmarks déjà réalisés (quand il y en aura) -->
      <div class="bg-dark-800 rounded-xl border border-dark-600 p-6">
        <h3 class="text-sm font-semibold mb-2">📜 Historique des benchmarks</h3>
        <p class="text-sm text-gray-500">Aucun benchmark réalisé pour l'instant.</p>
      </div>

      <!-- Stats d'utilisation -->
      <div class="bg-dark-800 rounded-xl border border-dark-600 p-6 mt-4">
        <h3 class="text-sm font-semibold mb-2">📈 Statistiques d'utilisation</h3>
        <div id="benchmarkHistory" class="text-sm text-gray-500">
          ⏳ Chargement...
        </div>
      </div>
    </div>
  `;

  // Charger l'historique des runs
  try {
    const history = await getHistory();
    const runs = history.runs || [];
    const el2 = document.getElementById('benchmarkHistory');
    if (runs.length === 0) {
      el2.innerHTML = '<p class="text-gray-500">Aucun run enregistré.</p>';
    } else {
      el2.innerHTML = `
        <div class="text-xs text-gray-400 mb-2">Derniers runs :</div>
        <div class="space-y-1">
          ${runs.slice(0, 10).map(r => `
            <div class="bg-dark-700/30 rounded px-3 py-1.5 text-xs flex justify-between">
              <span>${r.model_id}</span>
              <span class="text-gray-400">${r.tokens_generated || 0} tok • ${r.avg_speed || '?'} tok/s</span>
            </div>
          `).join('')}
        </div>`;
    }
  } catch (e) {
    document.getElementById('benchmarkHistory').innerHTML = `
      <p class="text-red-400 text-xs">❌ ${e.message}</p>`;
  }
}
