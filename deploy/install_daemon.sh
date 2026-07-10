#!/usr/bin/env bash
# qtrade 后台守护一键安装脚本（macOS launchd）。
#
# 它会：
#   1. 把 deploy/com.qtrade.daemon.plist 拷到 ~/Library/LaunchAgents/
#   2. 自动把里面的 python 路径替换成当前 venv 的 python
#   3. 创建日志目录
#   4. launchctl load + start
#
# 用法：
#   bash deploy/install_daemon.sh         # 安装并启动
#   bash deploy/install_daemon.sh stop    # 停止
#   bash deploy/install_daemon.sh remove  # 卸载
#   bash deploy/install_daemon.sh status  # 查看状态

set -euo pipefail

LABEL="com.qtrade.daemon"
PLIST_SRC="$(cd "$(dirname "$0")" && pwd)/com.qtrade.daemon.plist"
PLIST_DST="$HOME/Library/LaunchAgents/$LABEL.plist"
LOG_DIR="$HOME/Library/Logs/qtrade"
ACTION="${1:-install}"

case "$ACTION" in
  install)
    PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
    VENV_PYTHON="$PROJECT_DIR/.venv/bin/python"
    if [ ! -x "$VENV_PYTHON" ]; then
      echo "✗ 找不到 venv python: $VENV_PYTHON"
      echo "  请先在项目目录下：python3 -m venv .venv && source .venv/bin/activate && pip install -e ."
      exit 1
    fi

    mkdir -p "$LOG_DIR"
    mkdir -p "$(dirname "$PLIST_DST")"

    # 把模板里的路径替换成实际路径
    sed -e "s|/Users/dayang/Desktop/量化交易平台/.venv/bin/python|$VENV_PYTHON|g" \
        -e "s|/Users/dayang/Desktop/量化交易平台|$PROJECT_DIR|g" \
        -e "s|/Users/dayang/Library/Logs/qtrade|$LOG_DIR|g" \
        "$PLIST_SRC" > "$PLIST_DST"

    # 卸载旧的（避免重复 load）
    launchctl unload "$PLIST_DST" 2>/dev/null || true

    launchctl load "$PLIST_DST"
    launchctl start "$LABEL"

    echo "✓ qtrade daemon 已安装并启动"
    echo "  日志：$LOG_DIR/daemon.log"
    echo "  查看状态：bash $0 status"
    ;;

  stop)
    launchctl stop "$LABEL" 2>/dev/null || true
    launchctl unload "$PLIST_DST" 2>/dev/null || true
    echo "✓ qtrade daemon 已停止"
    ;;

  remove)
    launchctl stop "$LABEL" 2>/dev/null || true
    launchctl unload "$PLIST_DST" 2>/dev/null || true
    rm -f "$PLIST_DST"
    echo "✓ qtrade daemon 已卸载（日志保留在 $LOG_DIR）"
    ;;

  status)
    if launchctl list | grep -q "$LABEL"; then
      echo "● qtrade daemon 在跑"
      launchctl list | grep "$LABEL"
      echo ""
      echo "最近日志（tail）："
      tail -n 30 "$LOG_DIR/daemon.log" 2>/dev/null || echo "  暂无日志"
    else
      echo "○ qtrade daemon 未运行"
    fi
    ;;

  *)
    echo "用法: $0 [install|stop|remove|status]"
    exit 1
    ;;
esac
