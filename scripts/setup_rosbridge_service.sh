#!/usr/bin/env bash
# Install on the Xavier HOST, outside robot-dev.
set -euo pipefail
if [[ "${1:-}" == --help ]]; then
    echo '使用：sudo bash setup_rosbridge_service.sh（在 Xavier 主機執行）'
    exit 0
fi
[[ $# == 0 && $EUID == 0 ]] || { echo '請在 Xavier 主機以 sudo bash 執行，不接受參數。' >&2; exit 1; }
command -v docker >/dev/null
[[ "$(docker inspect -f '{{.HostConfig.NetworkMode}}' robot-dev)" == host ]] || {
    echo 'robot-dev 不是 host 網路；請先確認 9090 的 Docker port mapping，未變更設定。' >&2; exit 1;
}
docker start robot-dev >/dev/null
docker exec robot-dev test -r /workspace/install_humble_rosbridge/local_setup.bash || {
    echo '缺少 rosbridge overlay，請先執行 scripts/install_rosbridge_humble.sh。' >&2; exit 1;
}
script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
install -m 755 "$script_dir/systemd/robot-rosbridge" /usr/local/bin/robot-rosbridge
install -m 644 "$script_dir/systemd/robot-rosbridge.service" /etc/systemd/system/robot-rosbridge.service
systemctl daemon-reload
systemctl enable robot-rosbridge.service
systemctl restart robot-rosbridge.service
echo '已設為開機自動啟動。檢查：sudo journalctl -u robot-rosbridge -n 40 --no-pager'
echo '監聽檢查：sudo ss -ltnp sport = :9090'
