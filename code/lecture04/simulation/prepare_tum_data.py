"""prepare_tum_data.py —— 从 TUM RGB-D 序列中提取「一对」时间戳对齐的 RGB + 深度图。

TUM RGB-D 的 rgb.txt / depth.txt 里，彩色相机与深度相机的时间戳**不同步**，因此
必须先按时间戳做最近邻匹配（等价于官方 associate.py），再从配对结果里取一对图。

用法：
    python prepare_tum_data.py <解压后的序列目录> [--index 0]

    # 例如（fr1/desk，解压得到 rgbd_dataset_freiburg1_desk/）：
    #   python prepare_tum_data.py /tmp/rgbd_dataset_freiburg1_desk --index 0

输出：
    data/tum_rgb.png         # 彩色图（640x480）
    data/tum_depth.png       # 16bit 深度图（值 / 5000 = 米，0 表示无效）
    data/tum_camera_info.yaml  # RGB / depth 内参与深度 scale（fr1 官方值）
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys

import yaml

# TUM fr1 序列官方内参（见 TUM RGB-D benchmark 说明页）
FR1_RGB = dict(fx=517.3, fy=516.5, cx=318.6, cy=255.3)
FR1_DEPTH = dict(fx=525.0, fy=525.0, cx=319.5, cy=239.5)
DEPTH_SCALE = 5000.0   # 16bit 深度值 / 5000 = 米


def read_file_list(path: str) -> list[tuple[float, str]]:
    """读取 TUM 的 rgb.txt / depth.txt（格式：`timestamp filepath`，跳过 # 注释）。"""
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            ts, fp = line.split()
            data.append((float(ts), fp))
    return data


def associate(
    rgb_list: list[tuple[float, str]],
    depth_list: list[tuple[float, str]],
    max_difference: float = 0.02,
) -> list[tuple[str, str]]:
    """等价于官方 associate.py：对每个 RGB 时间戳找最近的 depth 时间戳。

    返回 [(rgb_filepath, depth_filepath), ...]，按 RGB 时间戳升序排列。
    """
    rgb_by_ts = {ts: fp for ts, fp in rgb_list}
    depth_by_ts = {ts: fp for ts, fp in depth_list}

    candidates = []
    for rgb_ts in rgb_by_ts:
        for depth_ts in depth_by_ts:
            diff = abs(rgb_ts - depth_ts)
            if diff < max_difference:
                candidates.append((diff, rgb_ts, depth_ts))
    candidates.sort()

    matched = []
    used_rgb, used_depth = set(), set()
    for _, rgb_ts, depth_ts in candidates:
        if rgb_ts not in used_rgb and depth_ts not in used_depth:
            used_rgb.add(rgb_ts)
            used_depth.add(depth_ts)
            matched.append((rgb_by_ts[rgb_ts], depth_by_ts[depth_ts]))
    return matched


def main() -> None:
    ap = argparse.ArgumentParser(description="从 TUM RGB-D 序列提取一对对齐的 RGB + depth 图")
    ap.add_argument("seq_dir", help="解压后的 TUM 序列目录（含 rgb.txt / depth.txt / rgb/ / depth/）")
    ap.add_argument("--index", type=int, default=0, help="取配对结果的第几对（默认第 0 对）")
    ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "data"))
    args = ap.parse_args()

    seq = args.seq_dir
    rgb_txt = os.path.join(seq, "rgb.txt")
    depth_txt = os.path.join(seq, "depth.txt")
    for p in (rgb_txt, depth_txt):
        if not os.path.exists(p):
            print(f"错误：缺少 {p}", file=sys.stderr)
            sys.exit(1)

    rgb_list = read_file_list(rgb_txt)
    depth_list = read_file_list(depth_txt)
    matches = associate(rgb_list, depth_list)
    print(f"RGB 帧 {len(rgb_list)}，depth 帧 {len(depth_list)}，配对成功 {len(matches)} 对")

    if args.index >= len(matches):
        print(f"错误：--index {args.index} 超出配对数量 {len(matches)}", file=sys.stderr)
        sys.exit(1)

    rgb_fp, depth_fp = matches[args.index]
    rgb_src = os.path.join(seq, rgb_fp)
    depth_src = os.path.join(seq, depth_fp)
    for p in (rgb_src, depth_src):
        if not os.path.exists(p):
            print(f"错误：文件不存在 {p}", file=sys.stderr)
            sys.exit(1)

    os.makedirs(args.out, exist_ok=True)
    rgb_out = os.path.join(args.out, "tum_rgb.png")
    depth_out = os.path.join(args.out, "tum_depth.png")
    shutil.copyfile(rgb_src, rgb_out)
    shutil.copyfile(depth_src, depth_out)

    # 写内参（RGB 与 depth 分开，深度 scale 单独给出）
    info = {
        "source": "TUM RGB-D fr1",
        "rgb": FR1_RGB,
        "depth": FR1_DEPTH,
        "depth_scale": DEPTH_SCALE,          # depth 值 / depth_scale = 米
        "depth_invalid": 0,                  # 0 表示无效深度
        "note": "depth 图已由 TUM 配准到 RGB 平面，二者像素一一对应；反投影 depth 用 depth 内参。",
    }
    yaml_out = os.path.join(args.out, "tum_camera_info.yaml")
    with open(yaml_out, "w", encoding="utf-8") as f:
        yaml.safe_dump(info, f, allow_unicode=True, sort_keys=False)

    print(f"已提取第 {args.index} 对：")
    print(f"  RGB   : {rgb_fp} -> {rgb_out}")
    print(f"  Depth : {depth_fp} -> {depth_out}")
    print(f"  内参  : {yaml_out}")
    print(f"  深度 scale = {DEPTH_SCALE}（值 / {DEPTH_SCALE} = 米）")


if __name__ == "__main__":
    main()
