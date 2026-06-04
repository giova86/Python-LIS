#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────
# LIS — avvio simultaneo di backend (FastAPI) e frontend (Vite)
# Uso:  ./start.sh
# Premi Ctrl+C per fermare entrambi.
# ──────────────────────────────────────────────────────────────
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BACKEND="$ROOT/backend"
FRONTEND="$ROOT/frontend"

GREEN='\033[0;32m'; YELLOW='\033[0;33m'; BLUE='\033[0;34m'; RESET='\033[0m'

# Chiude entrambi i processi figli all'uscita
cleanup() {
  echo -e "\n${YELLOW}» Arresto in corso…${RESET}"
  kill 0 2>/dev/null || true
}
trap cleanup EXIT INT TERM

# ── Backend ───────────────────────────────────────────────────
echo -e "${BLUE}» Avvio backend (FastAPI · :8000)…${RESET}"
cd "$BACKEND"

# Attiva il virtualenv se presente, altrimenti usa python di sistema
if [ -f "$BACKEND/venv/bin/activate" ]; then
  # shellcheck disable=SC1091
  source "$BACKEND/venv/bin/activate"
fi

uvicorn main:app --host 0.0.0.0 --port 8000 --reload &
BACKEND_PID=$!

# ── Frontend ──────────────────────────────────────────────────
echo -e "${GREEN}» Avvio frontend (Vite · :5173)…${RESET}"
cd "$FRONTEND"

if [ ! -d "$FRONTEND/node_modules" ]; then
  echo -e "${YELLOW}» node_modules mancante — eseguo npm install…${RESET}"
  npm install
fi

npm run dev &
FRONTEND_PID=$!

echo -e "${GREEN}» Tutto attivo.${RESET} Backend PID $BACKEND_PID · Frontend PID $FRONTEND_PID"
echo -e "  Frontend → http://localhost:5173"
echo -e "  Backend  → http://localhost:8000/health"

# Attende i processi; se uno muore, il trap chiude l'altro
wait
