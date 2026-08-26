#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────
# LIS — avvio simultaneo di backend (FastAPI) e frontend (Vite)
# Uso:  ./start.sh
# Premi Ctrl+C per fermare entrambi (e tutti i loro processi figli).
# ──────────────────────────────────────────────────────────────
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BACKEND="$ROOT/backend"
FRONTEND="$ROOT/frontend"

GREEN='\033[0;32m'; YELLOW='\033[0;33m'; BLUE='\033[0;34m'
CYAN='\033[0;36m'; DIM='\033[2m'; BOLD='\033[1m'; RESET='\033[0m'

BACKEND_PID=""
FRONTEND_PID=""

# Stampa ogni riga di stdin con un tag colorato davanti, es. "[backend]"
tag_output() {
  local label="$1" color="$2" line
  while IFS= read -r line; do
    printf "${color}${BOLD}[%s]${RESET} %s\n" "$label" "$line"
  done
}

# Elenca (in post-order) un PID e tutti i suoi discendenti
collect_tree() {
  local pid="$1" kid
  for kid in $(pgrep -P "$pid" 2>/dev/null); do
    collect_tree "$kid"
  done
  echo "$pid"
}

kill_tree() {
  local root="$1" sig="$2" pid
  [ -z "$root" ] && return 0
  for pid in $(collect_tree "$root"); do
    kill -"$sig" "$pid" 2>/dev/null
  done
}

any_alive() {
  local root="$1" pid
  [ -z "$root" ] && return 1
  for pid in $(collect_tree "$root"); do
    kill -0 "$pid" 2>/dev/null && return 0
  done
  return 1
}

# Invia ripetutamente $sig a entrambi gli alberi di processi per $tries tentativi:
# npm/uvicorn possono generare figli (vite, il reloader) con un piccolo ritardo,
# quindi un singolo giro rischia di non vederli ancora e lasciarli orfani.
signal_until_dead() {
  local sig="$1" tries="$2" i=0
  while [ "$i" -lt "$tries" ]; do
    kill_tree "$BACKEND_PID" "$sig"
    kill_tree "$FRONTEND_PID" "$sig"
    if ! any_alive "$BACKEND_PID" && ! any_alive "$FRONTEND_PID"; then
      return 0
    fi
    sleep 0.5
    i=$((i + 1))
  done
  ! any_alive "$BACKEND_PID" && ! any_alive "$FRONTEND_PID"
}

cleanup() {
  trap - EXIT INT TERM
  echo -e "\n${YELLOW}» Arresto in corso…${RESET}"

  # Fino a 5s di terminazione garbata, poi passa alle maniere forti per altri 3s
  if ! signal_until_dead TERM 10; then
    signal_until_dead KILL 6
  fi

  echo -e "${GREEN}» Backend e frontend arrestati.${RESET}"
  exit 0
}
trap cleanup EXIT INT TERM

echo -e "${CYAN}${BOLD}┌─────────────────────────────────────────┐${RESET}"
echo -e "${CYAN}${BOLD}│  LIS — Riconoscimento Alfabeto           │${RESET}"
echo -e "${CYAN}${BOLD}└─────────────────────────────────────────┘${RESET}"

# ── Backend ───────────────────────────────────────────────────
echo -e "${BLUE}» Avvio backend (FastAPI · :8000)…${RESET}"
(
  cd "$BACKEND"
  if [ -f "$BACKEND/venv/bin/activate" ]; then
    # shellcheck disable=SC1091
    source "$BACKEND/venv/bin/activate"
  fi
  exec uvicorn main:app --host 0.0.0.0 --port 8000 --reload
) > >(tag_output backend "$BLUE") 2>&1 &
BACKEND_PID=$!

# ── Frontend ──────────────────────────────────────────────────
if [ ! -d "$FRONTEND/node_modules" ]; then
  echo -e "${YELLOW}» node_modules mancante — eseguo npm install…${RESET}"
  (cd "$FRONTEND" && npm install)
fi

echo -e "${GREEN}» Avvio frontend (Vite · :5173)…${RESET}"
(
  cd "$FRONTEND"
  exec npm run dev
) > >(tag_output frontend "$GREEN") 2>&1 &
FRONTEND_PID=$!

echo -e "${CYAN}${BOLD}» Tutto attivo.${RESET}"
echo -e "  ${DIM}Frontend${RESET} → http://localhost:5173"
echo -e "  ${DIM}Backend${RESET}  → http://localhost:8000/health"
echo -e "  ${DIM}Premi Ctrl+C per fermare backend e frontend.${RESET}"
echo

# Attende entrambi i processi principali
wait "$BACKEND_PID" "$FRONTEND_PID"
