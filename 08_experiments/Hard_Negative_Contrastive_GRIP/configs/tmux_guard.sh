# Re-exec a launch script inside a detached tmux session.
#
# Usage from a training/eval script, after arguments are known:
#   TMUX_SESSION=expH-20260918
#   # shellcheck source=tmux_guard.sh
#   source "$HNG/configs/tmux_guard.sh"
#   tmux_guard_reexec "$0" "$@"
#
# Already-in-tmux and SKIP_TMUX=1 skip the wrap (short debug only).

tmux_guard_reexec() {
  if [[ "${SKIP_TMUX:-0}" == "1" ]]; then
    return 0
  fi
  if [[ -n "${TMUX:-}" ]]; then
    return 0
  fi

  local self="${1:-}"
  if [[ -z "$self" ]]; then
    echo "error: tmux_guard_reexec needs the launch script path" >&2
    exit 1
  fi
  shift

  if ! command -v tmux >/dev/null 2>&1; then
    echo "error: tmux is required for server training." >&2
    echo "  install tmux, or set SKIP_TMUX=1 only for a short debug run." >&2
    exit 1
  fi

  local session="${TMUX_SESSION:-}"
  if [[ -z "$session" ]]; then
    echo "error: TMUX_SESSION is empty; set it before calling tmux_guard_reexec" >&2
    exit 1
  fi
  if [[ ! "$session" =~ ^[A-Za-z0-9_.-]+$ ]]; then
    echo "error: TMUX_SESSION may contain only letters, numbers, dot, underscore, and hyphen: $session" >&2
    exit 1
  fi

  if tmux has-session -t "$session" 2>/dev/null; then
    echo "tmux session '$session' already exists; not starting a second GPU job."
    echo "  attach: tmux attach -t $session"
    echo "  other session: TMUX_SESSION=new-name $self"
    exit 0
  fi

  local envfile
  envfile="$(mktemp /tmp/tmux-train-env.XXXXXX)"
  export -p >"$envfile"
  local quoted_env quoted_self quoted_args
  quoted_env="$(printf '%q' "$envfile")"
  quoted_self="$(printf '%q' "$self")"
  quoted_args="$(printf '%q ' "$@")"

  if ! tmux new-session -d -s "$session" -c "$PWD" \
    "set -euo pipefail; source $quoted_env; export SKIP_TMUX=1; rm -f $quoted_env; exec bash $quoted_self $quoted_args"; then
    rm -f "$envfile"
    echo "error: failed to create tmux session '$session'" >&2
    exit 1
  fi

  echo "launched in tmux session '$session' (detached)"
  echo "  attach:  tmux attach -t $session"
  echo "  panes:   tmux capture-pane -pt $session"
  echo "  list:    tmux ls"
  exit 0
}
