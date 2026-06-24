# Agent.md — AI Runner

## Règles générales

- Garder le répertoire source propre : pas de fichiers temporaires, pas de caches
- Les commits et push sont faits par l'utilisateur, jamais par l'agent
- Ne pas modifier les fichiers de configuration utilisateur (config/config.yaml)
- Ne pas modifier les fichiers dans les dossiers volumes (models/, data/, presets/)
- Préférer des modifications ciblées plutôt que des réécritures complètes
- Toujours lire le fichier existant avant de le modifier

## Convention de code

- Python : suivre PEP 8, utiliser des type hints
- JavaScript : vanilla, pas de framework, conventions ES6+
- HTML : Tailwind CSS via CDN, pas de build step
- Docstrings en français pour les modules, en anglais pour les APIs publiques

## Architecture

- `app/core/` : logique métier (parser GGUF, règles, détection système)
- `app/api/` : endpoints REST (préfixe /api/v1/)
- `app/web/static/` : interface web SPA (vanilla JS)
- Les modules core ne doivent pas importer les modules API
- Les modules API peuvent importer les modules core

## Tests

- Les tests sont dans `tests/`
- Exécuter `python3 -m pytest tests/ -v` avant de proposer du code
- 30 tests doivent passer
