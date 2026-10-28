#!/data/data/com.termux/files/usr/bin/bash
# Persistent Purffle autopilot supervisor for Termux.
set -u

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="${PURFFLE_ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)}"
STATE_DIR="$ROOT/data/service"
LOG_DIR="$ROOT/data/logs"
LOG_FILE="$LOG_DIR/purffle-service.log"
PID_FILE="$STATE_DIR/supervisor.pid"
LOCK_FILE="$STATE_DIR/supervisor.lock"
PYTHON_BIN="${PYTHON_BIN:-python}"

mkdir -p "$STATE_DIR" "$LOG_DIR"

log() { printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*" >> "$LOG_FILE"; }
alive() { [[ -n "${1:-}" ]] && kill -0 "$1" 2>/dev/null; }

start_service() {
    if ! command -v flock >/dev/null 2>&1; then
        echo "Missing flock. Install it once with: pkg install util-linux" >&2
        return 1
    fi
    if [[ ! -f "$ROOT/purffle_shorts/__main__.py" ]]; then
        echo "Purffle package not found at $ROOT. Set PURFFLE_ROOT to the repository path." >&2
        return 1
    fi
    if [[ -f "$PID_FILE" ]]; then
        old_pid="$(cat "$PID_FILE" 2>/dev/null || true)"
        if alive "$old_pid"; then
            echo "Purffle service already running (PID $old_pid)."
            return 0
        fi
        rm -f "$PID_FILE"
    fi
    if command -v termux-wake-lock >/dev/null 2>&1; then
        termux-wake-lock >/dev/null 2>&1 || true
    fi
    nohup "$SCRIPT_DIR/purffle-service.sh" supervise >> "$LOG_FILE" 2>&1 < /dev/null &
    echo "$!" > "$PID_FILE"
    sleep 1
    if alive "$(cat "$PID_FILE" 2>/dev/null || true)"; then
        echo "Purffle service started. Log: $LOG_FILE"
        echo "It follows your existing .env settings; it does not enable uploads."
    else
        rm -f "$PID_FILE"
        echo "Service failed to start. Check: $LOG_FILE" >&2
        return 1
    fi
}

stop_service() {
    if [[ ! -f "$PID_FILE" ]]; then
        echo "Purffle service is not running (no PID file)."
        return 0
    fi
    pid="$(cat "$PID_FILE" 2>/dev/null || true)"
    if ! alive "$pid"; then
        rm -f "$PID_FILE"
        echo "Removed stale PID file; service was not running."
        return 0
    fi
    kill -TERM "$pid" 2>/dev/null || true
    for _ in $(seq 1 15); do
        alive "$pid" || break
        sleep 1
    done
    if alive "$pid"; then
        echo "Stop requested; process is taking longer than expected. Check $LOG_FILE"
        return 1
    fi
    rm -f "$PID_FILE"
    echo "Purffle service stopped."
}

supervise() {
    if ! command -v flock >/dev/null 2>&1; then
        log "ERROR: flock missing; install with pkg install util-linux"
        exit 1
    fi
    exec 9>"$LOCK_FILE"
    if ! flock -n 9; then
        log "Another supervisor owns the lock; exiting."
        exit 0
    fi

    STOPPING=0
    CHILD_PID=""
    echo "$$" > "$PID_FILE"
    shutdown_child() {
        STOPPING=1
        if [[ -n "$CHILD_PID" ]] && alive "$CHILD_PID"; then
            kill -TERM "$CHILD_PID" 2>/dev/null || true
            wait "$CHILD_PID" 2>/dev/null || true
        fi
        rm -f "$PID_FILE"
    }
    trap shutdown_child TERM INT

    cd "$ROOT" || exit 1
    log "Supervisor started in $ROOT"
    while [[ "$STOPPING" -eq 0 ]]; do
        log "Starting Purffle autopilot; settings are read from the existing environment/.env."
        "$PYTHON_BIN" -m purffle_shorts run >> "$LOG_FILE" 2>&1 &
        CHILD_PID=$!
        wait "$CHILD_PID"
        code=$?
        CHILD_PID=""
        [[ "$STOPPING" -eq 1 ]] && break
        log "Purffle exited with code $code; restarting in 20 seconds."
        sleep 20 &
        CHILD_PID=$!
        wait "$CHILD_PID" || true
        CHILD_PID=""
    done
    rm -f "$PID_FILE"
    log "Supervisor stopped."
}

case "${1:-status}" in
    start) start_service ;;
    stop) stop_service ;;
    restart) stop_service && start_service ;;
    status)
        if [[ -f "$PID_FILE" ]] && alive "$(cat "$PID_FILE" 2>/dev/null || true)"; then
            echo "Purffle service is running (PID $(cat "$PID_FILE"))."
            echo "Log: $LOG_FILE"
        else
            rm -f "$PID_FILE"
            echo "Purffle service is stopped."
        fi
        ;;
    logs) tail -n "${2:-80}" -f "$LOG_FILE" ;;
    supervise) supervise ;;
    *)
        echo "Usage: $0 {start|stop|restart|status|logs [lines]}" >&2
        exit 2
        ;;
esac
