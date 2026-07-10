#!/usr/bin/env bash
set -euo pipefail

# RDK S100 fixed USB layout for the SO101 pair.
# These paths are tied to the physical S100 USB positions, not insertion order.
leader_id_path="platform-34000000.pcie-pci-0000:06:00.0-usb-0:1:1.0"
follower_id_path="platform-34000000.pcie-pci-0000:06:00.0-usb-0:2:1.0"

vendor_id="1a86"
product_id="55d3"
rule_file="/etc/udev/rules.d/99-so101-s100-serial.rules"
dry_run=false

usage() {
  cat <<'EOF'
Usage: ./bind_uarm_serial_port_s100.sh [--dry-run]

Bind the fixed RDK S100 SO101 USB positions:
  /dev/ttyLeader
  /dev/ttyFollower
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

leader_rule="SUBSYSTEM==\"tty\", ENV{ID_PATH}==\"$leader_id_path\", ATTRS{idVendor}==\"$vendor_id\", ATTRS{idProduct}==\"$product_id\", SYMLINK+=\"ttyLeader\", MODE=\"0666\", ENV{ID_MM_DEVICE_IGNORE}=\"1\""
follower_rule="SUBSYSTEM==\"tty\", ENV{ID_PATH}==\"$follower_id_path\", ATTRS{idVendor}==\"$vendor_id\", ATTRS{idProduct}==\"$product_id\", SYMLINK+=\"ttyFollower\", MODE=\"0666\", ENV{ID_MM_DEVICE_IGNORE}=\"1\""

echo "S100 fixed SO101 serial bindings:"
echo "  leader:   $leader_id_path -> /dev/ttyLeader"
echo "  follower: $follower_id_path -> /dev/ttyFollower"
echo "  rule file: $rule_file"
echo
printf '%s\n%s\n' "$leader_rule" "$follower_rule"

if [[ "$dry_run" == true ]]; then
  exit 0
fi

printf '%s\n%s\n' "$leader_rule" "$follower_rule" | sudo tee "$rule_file" >/dev/null
sudo udevadm control --reload-rules
sudo udevadm trigger --subsystem-match=tty
udevadm settle || true

echo
echo "Installed $rule_file. Current links:"
ls -l /dev/ttyLeader /dev/ttyFollower
