#!/usr/bin/env bash
set -euo pipefail

link_name="uarmRight"
port=""
rule_file="/etc/udev/rules.d/99-uarm.rules"
dry_run=false

usage() {
  cat <<'EOF'
Usage: scripts/bind_uarm_serial_port.sh [--port /dev/ttyACM0] [--name uarmRight] [--dry-run]

Reads the USB serial number from a connected uArm USB serial adapter and installs
a udev rule that creates a stable /dev/<name> symlink. Existing rules are kept:
running this once for uarmLeft and later for uarmRight appends the second rule.
If the same serial or same symlink name already exists, that one rule is updated.

Examples:
  scripts/bind_uarm_serial_port.sh --port /dev/ttyACM0 --name uarmLeft
  scripts/bind_uarm_serial_port.sh --port /dev/ttyACM1 --name uarmRight
  scripts/bind_uarm_serial_port.sh --dry-run
EOF
}

while (($#)); do
  case "$1" in
    --port)
      port="${2:-}"
      shift 2
      ;;
    --name)
      link_name="${2:-}"
      shift 2
      ;;
    --rule-file)
      rule_file="${2:-}"
      shift 2
      ;;
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

if [[ -z "$link_name" || "$link_name" == */* ]]; then
  echo "--name must be a /dev symlink basename, for example uarmRight" >&2
  exit 2
fi

read_property() {
  local dev="$1"
  local key="$2"
  udevadm info -q property -n "$dev" | awk -F= -v key="$key" '$1 == key {print $2; exit}'
}

is_uarm_candidate() {
  local dev="$1"
  [[ -e "$dev" ]] || return 1
  [[ "$(read_property "$dev" ID_VENDOR_ID)" == "1a86" ]] || return 1
  [[ "$(read_property "$dev" ID_MODEL_ID)" == "55d3" ]] || return 1
}

if [[ -z "$port" ]]; then
  candidates=()
  shopt -s nullglob
  for dev in /dev/ttyACM* /dev/ttyUSB*; do
    if is_uarm_candidate "$dev"; then
      candidates+=("$dev")
    fi
  done
  shopt -u nullglob

  if ((${#candidates[@]} == 0)); then
    echo "No uArm USB serial adapter found. Plug it in and retry." >&2
    exit 1
  fi
  if ((${#candidates[@]} > 1)); then
    echo "Multiple uArm USB serial adapters found:" >&2
    for dev in "${candidates[@]}"; do
      echo "  $dev serial=$(read_property "$dev" ID_SERIAL_SHORT)" >&2
    done
    echo "Pass --port /dev/ttyACM* to choose one." >&2
    exit 2
  fi
  port="${candidates[0]}"
fi

if [[ ! -e "$port" ]]; then
  echo "Port does not exist: $port" >&2
  exit 1
fi

vendor_id="$(read_property "$port" ID_VENDOR_ID)"
product_id="$(read_property "$port" ID_MODEL_ID)"
serial="$(read_property "$port" ID_SERIAL_SHORT)"
product="$(read_property "$port" ID_MODEL)"

if [[ -z "$vendor_id" || -z "$product_id" || -z "$serial" ]]; then
  echo "Could not read vendor/product/serial from $port." >&2
  udevadm info -q property -n "$port" >&2
  exit 1
fi

rule="SUBSYSTEM==\"tty\", ATTRS{idVendor}==\"$vendor_id\", ATTRS{idProduct}==\"$product_id\", ATTRS{serial}==\"$serial\", SYMLINK+=\"$link_name\", MODE=\"0666\", ENV{ID_MM_DEVICE_IGNORE}=\"1\""

existing_serial_rule=""
existing_link_rule=""
if [[ -f "$rule_file" ]]; then
  existing_serial_rule="$(awk -v serial="$serial" 'index($0, "ATTRS{serial}==\"" serial "\"") > 0 {print; exit}' "$rule_file")"
  existing_link_rule="$(awk -v link_name="$link_name" 'index($0, "SYMLINK+=\"" link_name "\"") > 0 {print; exit}' "$rule_file")"
fi

echo "Detected uArm adapter:"
echo "  port: /dev/$(basename "$(readlink -f "$port")")"
echo "  product: ${product:-unknown}"
echo "  vendor/product: $vendor_id:$product_id"
echo "  serial: $serial"
echo "  symlink: /dev/$link_name"
if [[ -n "$existing_serial_rule" || -n "$existing_link_rule" ]]; then
  echo "  action: update existing rule"
else
  echo "  action: append new rule"
fi
echo
echo "udev rule:"
echo "$rule"

if [[ "$dry_run" == true ]]; then
  exit 0
fi

tmp="$(mktemp)"
trap 'rm -f "$tmp"' EXIT

if [[ -f "$rule_file" ]]; then
  # Preserve all existing rules except conflicts for this exact serial or symlink name.
  # This makes repeated runs append new uArms while keeping the command idempotent.
  awk -v serial="$serial" -v link_name="$link_name" '
    index($0, "ATTRS{serial}==\"" serial "\"") == 0 &&
    index($0, "SYMLINK+=\"" link_name "\"") == 0
  ' "$rule_file" > "$tmp"
else
  : > "$tmp"
fi

printf '%s\n' "$rule" >> "$tmp"

if [[ -f "$rule_file" ]]; then
  sudo cp -a "$rule_file" "$rule_file.bak.$(date +%Y%m%d_%H%M%S)"
fi
sudo install -m 0644 "$tmp" "$rule_file"
sudo udevadm control --reload-rules
sudo udevadm trigger --subsystem-match=tty
udevadm settle || true

echo
echo "Installed $rule_file. Current link:"
ls -l "/dev/$link_name" 2>/dev/null || echo "  /dev/$link_name not visible yet; replug the uArm USB cable if needed."
