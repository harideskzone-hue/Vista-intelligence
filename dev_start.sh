#!/bin/bash
# ═══════════════════════════════════════════════════════════════════════════
# dev_start.sh — SIH26187 Single-Command Dev Launcher
# ═══════════════════════════════════════════════════════════════════════════
# Usage:  bash dev_start.sh [--debug] [--safe-mode] [--skip-warmup]
#
# Flags:
#   --debug       Enable bbox overlay + stats bar in live_scorer
#   --safe-mode   Skip heavy models (InsightFace) for debugging
#   --skip-warmup Skip model preloading (faster startup)
# ═══════════════════════════════════════════════════════════════════════════

set -euo pipefail

# ── Parse flags ────────────────────────────────────────────────────────────
DEBUG_MODE=false
SAFE_MODE=false
SKIP_WARMUP=false
for arg in "$@"; do
    case "$arg" in
        --debug)      DEBUG_MODE=true ;;
        --safe-mode)  SAFE_MODE=true ;;
        --skip-warmup) SKIP_WARMUP=true ;;
    esac
done

# ── Resolve project root (where this script lives) ────────────────────────
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# ── Constants ──────────────────────────────────────────────────────────────
# We now use local directories within the repo for a fully standalone setup.
USER_DATA="$SCRIPT_DIR/data"
VENV_DIR="$SCRIPT_DIR/.venv"
VENV_PY="$VENV_DIR/bin/python3"
LOG_DIR="$SCRIPT_DIR/logs"
API_PORT=5001
LOCK_FILE="/tmp/magic_click_dev.lock"



# Pids to track for cleanup
DB_PID=""
SCORER_PID=""
WORKER_PID=""

# ── Colors ─────────────────────────────────────────────────────────────────
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'  # No Color
BOLD='\033[1m'

# ── Banner ─────────────────────────────────────────────────────────────────
echo ""
echo -e "${CYAN}╔══════════════════════════════════════════╗${NC}"
echo -e "${CYAN}║  ${BOLD}SIH26187 — Dev Pipeline Launcher${NC}${CYAN}     ║${NC}"
echo -e "${CYAN}╚══════════════════════════════════════════╝${NC}"
echo ""

# ═══════════════════════════════════════════════════════════════════════════
# CLEANUP TRAP — kills all children on exit/Ctrl-C
# ═══════════════════════════════════════════════════════════════════════════
cleanup() {
    echo ""
    echo -e "${YELLOW}[Launcher] 🛑 Shutting down all services…${NC}"

    # Kill tracked PIDs
    for pid in $DB_PID $SCORER_PID $WORKER_PID; do
        if [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null; then
            kill "$pid" 2>/dev/null
            wait "$pid" 2>/dev/null || true
        fi
    done

    # Kill any stale processes on our port
    lsof -ti ":$API_PORT" 2>/dev/null | xargs kill 2>/dev/null || true

    # Remove lock file
    rm -f "$LOCK_FILE"

    echo -e "${GREEN}[Launcher] ✓  All services stopped.${NC}"
}
trap cleanup EXIT INT TERM

# ═══════════════════════════════════════════════════════════════════════════
# CHECK 1: Idempotency — prevent duplicate launches
# ═══════════════════════════════════════════════════════════════════════════
if [ -f "$LOCK_FILE" ]; then
    OLD_PID=$(cat "$LOCK_FILE" 2>/dev/null || echo "")
    if [ -n "$OLD_PID" ] && kill -0 "$OLD_PID" 2>/dev/null; then
        echo -e "${RED}[FATAL] Another dev_start.sh is already running (PID $OLD_PID).${NC}"
        echo "        Kill it first:  kill $OLD_PID"
        exit 1
    fi
    rm -f "$LOCK_FILE"
fi
echo $$ > "$LOCK_FILE"

# ═══════════════════════════════════════════════════════════════════════════
# CHECK 2: Python version (require 3.10–3.12)
# ═══════════════════════════════════════════════════════════════════════════
echo -e "${BLUE}[CHECK] Python version…${NC}"
PYTHON=""
for candidate in python3.11 python3.12 python3.10 python3; do
    if command -v "$candidate" &>/dev/null; then
        ver=$("$candidate" -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')" 2>/dev/null || echo "0.0")
        major=$(echo "$ver" | cut -d. -f1)
        minor=$(echo "$ver" | cut -d. -f2)
        if [ "$major" = "3" ] && [ "$minor" -ge 10 ] && [ "$minor" -le 12 ]; then
            PYTHON="$candidate"
            break
        fi
    fi
done

if [ -z "$PYTHON" ]; then
    echo -e "${RED}[FATAL] Python 3.10–3.12 not found. Install Python 3.11 from python.org${NC}"
    exit 1
fi
echo -e "  ${GREEN}✓${NC} Found $PYTHON ($($PYTHON --version 2>&1))"

# ═══════════════════════════════════════════════════════════════════════════
# CHECK 3: Disk space (require ≥ 2GB free)
# ═══════════════════════════════════════════════════════════════════════════
echo -e "${BLUE}[CHECK] Disk space…${NC}"
FREE_KB=$(df -k "$HOME" | tail -1 | awk '{print $4}')
FREE_GB=$(echo "scale=1; $FREE_KB / 1048576" | bc 2>/dev/null || echo "unknown")

if [ "$FREE_KB" -lt 2097152 ] 2>/dev/null; then
    echo -e "${RED}[FATAL] Less than 2 GB free disk space (${FREE_GB} GB). Free up space first.${NC}"
    exit 1
fi
echo -e "  ${GREEN}✓${NC} ${FREE_GB} GB free"

# ═══════════════════════════════════════════════════════════════════════════
# CHECK 4: Port conflict detection
# ═══════════════════════════════════════════════════════════════════════════
echo -e "${BLUE}[CHECK] Port $API_PORT availability…${NC}"
if lsof -ti ":$API_PORT" &>/dev/null; then
    STALE_PID=$(lsof -ti ":$API_PORT" | head -1)
    echo -e "  ${YELLOW}⚠${NC}  Port $API_PORT in use by PID $STALE_PID — killing it…"
    kill "$STALE_PID" 2>/dev/null || true
    sleep 1
    if lsof -ti ":$API_PORT" &>/dev/null; then
        kill -9 "$STALE_PID" 2>/dev/null || true
        sleep 1
    fi
fi
echo -e "  ${GREEN}✓${NC} Port $API_PORT is free"

# ═══════════════════════════════════════════════════════════════════════════
# STEP 1: Create / verify venv
# ═══════════════════════════════════════════════════════════════════════════
echo -e "\n${BLUE}[SETUP] Virtual environment…${NC}"
mkdir -p "$USER_DATA"
mkdir -p "$LOG_DIR"

if [ ! -f "$VENV_PY" ]; then
    echo "  Creating venv at $VENV_DIR …"
    "$PYTHON" -m venv "$VENV_DIR"
    echo -e "  ${GREEN}✓${NC} Venv created"
else
    echo -e "  ${GREEN}✓${NC} Venv exists at $VENV_DIR"
fi

# ═══════════════════════════════════════════════════════════════════════════
# STEP 2: Install / verify dependencies
# ═══════════════════════════════════════════════════════════════════════════
echo -e "${BLUE}[SETUP] Dependencies…${NC}"

# Quick check: can we import fastapi and ultralytics?
DEPS_OK=$("$VENV_PY" -c "import fastapi, ultralytics; print('ok')" 2>/dev/null || echo "missing")

if [ "$DEPS_OK" != "ok" ]; then
    echo "  Installing dependencies (this may take a few minutes on first run)…"
    "$VENV_PY" -m pip install --upgrade --timeout 60 pip -q 2>/dev/null || true

    if [ -f "$SCRIPT_DIR/face_api/requirements.txt" ]; then
        "$VENV_PY" -m pip install --timeout 180 -r "$SCRIPT_DIR/face_api/requirements.txt" -q \
            2>&1 | tee -a "$LOG_DIR/pip_install.log"
    fi
    if [ -f "$SCRIPT_DIR/face_engine/requirements.txt" ]; then
        "$VENV_PY" -m pip install --timeout 180 -r "$SCRIPT_DIR/face_engine/requirements.txt" -q \
            2>&1 | tee -a "$LOG_DIR/pip_install.log"
    fi
    echo -e "  ${GREEN}✓${NC} Dependencies installed"
else
    echo -e "  ${GREEN}✓${NC} All dependencies present"
fi

# ═══════════════════════════════════════════════════════════════════════════
# STEP 3: Model validation + auto-download
# ═══════════════════════════════════════════════════════════════════════════
echo -e "${BLUE}[SETUP] AI Models…${NC}"
"$VENV_PY" -c "
import sys
sys.path.insert(0, '$SCRIPT_DIR/face_engine')
from model_manager import validate_models, download_missing, print_status_table

status = validate_models()
missing = [f for f, s in status.items() if not s['exists'] or not s['sha_ok']]

if missing:
    print('  Downloading missing models...')
    if not download_missing():
        print('  FATAL: Required models unavailable.')
        sys.exit(1)

print_status_table()
"
if [ $? -ne 0 ]; then
    echo -e "${RED}[FATAL] Model setup failed. Check network connection.${NC}"
    exit 1
fi

# ═══════════════════════════════════════════════════════════════════════════
# STEP 4: Model warmup (preload to avoid cold-start)
# ═══════════════════════════════════════════════════════════════════════════
if [ "$SKIP_WARMUP" = false ]; then
    echo -e "${BLUE}[WARMUP] Preloading YOLO + InsightFace…${NC}"
    "$VENV_PY" -c "
import sys, time
sys.path.insert(0, '$SCRIPT_DIR/face_engine')
from model_manager import warmup_models
t0 = time.time()
r = warmup_models()
for name, info in r.items():
    if info.get('loaded'):
        print(f'  ✓ {name}: loaded in {info[\"time_s\"]}s')
    else:
        print(f'  ⚠ {name}: {info.get(\"error\", \"unknown\")}')
print(f'  Total warmup: {time.time()-t0:.1f}s')
" 2>&1 || echo -e "  ${YELLOW}⚠${NC} Warmup had warnings (non-fatal)"
fi

# ═══════════════════════════════════════════════════════════════════════════
# STEP 5: Launch face_api (FastAPI)
# ═══════════════════════════════════════════════════════════════════════════
echo -e "\n${BLUE}[LAUNCH] Starting face_api (FastAPI, port $API_PORT)…${NC}"

KMP_DUPLICATE_LIB_OK=True PYTHONPATH="$SCRIPT_DIR/face_api:$SCRIPT_DIR" "$VENV_PY" "$SCRIPT_DIR/face_api/run.py" \
    >> "$LOG_DIR/api.log" 2>&1 &
DB_PID=$!
echo "  PID: $DB_PID"

# Poll for health
echo -e "${BLUE}[LAUNCH] Waiting for API health…${NC}"
for i in $(seq 1 30); do
    if curl -sf "http://localhost:$API_PORT/api/health" > /dev/null 2>&1; then
        echo -e "  ${GREEN}✓${NC} API ready at http://localhost:$API_PORT"
        break
    fi
    if [ "$i" -eq 30 ]; then
        echo -e "${RED}[FATAL] API did not start within 30 seconds.${NC}"
        echo "  Check logs: $LOG_DIR/api.log"
        tail -20 "$LOG_DIR/api.log" 2>/dev/null
        exit 1
    fi
    sleep 1
done



# ═══════════════════════════════════════════════════════════════════════════
# STEP 7: Open dashboard in browser
# ═══════════════════════════════════════════════════════════════════════════
echo -e "${BLUE}[LAUNCH] Opening dashboard…${NC}"
open "http://localhost:$API_PORT/" 2>/dev/null || true

# ═══════════════════════════════════════════════════════════════════════════
# STEP 8: Launch live_scorer (foreground — captures Ctrl+C)
# ═══════════════════════════════════════════════════════════════════════════
echo -e "${BLUE}[LAUNCH] Starting live_scorer (camera pipeline)…${NC}"
echo ""
echo -e "${CYAN}┌─────────────────────────────────────────────┐${NC}"
echo -e "${CYAN}│  Dashboard  → http://localhost:$API_PORT/         │${NC}"
echo -e "${CYAN}│  API Docs   → http://localhost:$API_PORT/docs     │${NC}"
echo -e "${CYAN}│  API Log    → $LOG_DIR/api.log   │${NC}"
echo -e "${CYAN}│  Press Ctrl-C to shut down everything       │${NC}"
echo -e "${CYAN}└─────────────────────────────────────────────┘${NC}"
echo ""

SCORER_ARGS=""
if [ "$DEBUG_MODE" = true ]; then
    SCORER_ARGS="--debug"
fi

cd "$SCRIPT_DIR/face_engine"
"$VENV_PY" -u "$SCRIPT_DIR/face_engine/live_scorer.py" $SCORER_ARGS \
    2>&1 | tee -a "$LOG_DIR/scorer.log"

# If live_scorer exits, the trap will clean up everything
