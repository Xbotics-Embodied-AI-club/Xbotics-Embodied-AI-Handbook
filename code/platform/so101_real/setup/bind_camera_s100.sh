#!/usr/bin/env bash
set -euo pipefail

# RDK S100 fixed USB layout for the SO101 cameras.
# Bind only video index 0; index 1 is the companion metadata node.
top_camera_id_path="platform-34000000.pcie-pci-0000:07:00.0-usb-0:2:1.0"
wrist_camera_id_path="platform-34000000.pcie-pci-0000:07:00.0-usb-0:1:1.0"

rule_file="/etc/udev/rules.d/99-camera-s100.rules"
dry_run=false

usage() {
  cat <<'EOF'
Usage: ./bind_camera_s100.sh [--dry-run]

Bind the fixed RDK S100 camera USB positions:
  /dev/top_camera
  /dev/wrist_camera
EOF
}

while (($#)); do
  case "$1" in
    --dry-run)
      dry_run=true
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

top_camera_rule="SUBSYSTEM==\"video4linux\", KERNEL==\"video*\", ENV{ID_PATH}==\"$top_camera_id_path\", ATTR{index}==\"0\", ATTRS{idVendor}==\"0bda\", ATTRS{idProduct}==\"3035\", SYMLINK+=\"top_camera\", MODE=\"0666\""
wrist_camera_rule="SUBSYSTEM==\"video4linux\", KERNEL==\"video*\", ENV{ID_PATH}==\"$wrist_camera_id_path\", ATTR{index}==\"0\", ATTRS{idVendor}==\"05a3\", ATTRS{idProduct}==\"9230\", SYMLINK+=\"wrist_camera\", MODE=\"0666\""

echo "S100 fixed camera bindings:"
echo "  top camera: $top_camera_id_path video-index0 -> /dev/top_camera"
echo "  wrist camera: $wrist_camera_id_path video-index0 -> /dev/wrist_camera"
echo "  rule file: $rule_file"
echo
printf '%s\n%s\n' "$top_camera_rule" "$wrist_camera_rule"

if [[ "$dry_run" == true ]]; then
  exit 0
fi

printf '%s\n%s\n' "$top_camera_rule" "$wrist_camera_rule" | sudo tee "$rule_file" >/dev/null
sudo udevadm control --reload-rules
sudo udevadm trigger --subsystem-match=video4linux
udevadm settle || true

echo
echo "Installed $rule_file. Current links:"
ls -l /dev/top_camera /dev/wrist_camera
