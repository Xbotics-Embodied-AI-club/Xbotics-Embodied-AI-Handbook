# RDK S100 上 rerun 打不开的修复说明

## 症状
运行 `rerun`，或本目录 `record.sh` / `tele.sh`（带 `--display_data=true`）时，
窗口起不来，终端报：
```
app creation error: ... Adapter does not support drawing to texture format R32Float.
```

## 原因
rerun 的可视化窗口要求显卡支持把 R32Float 当渲染目标。
S100 的 Mali GPU 的 Vulkan 和 GLES 驱动都不支持，只有 CPU 软件渲染器支持。
（ `WGPU_BACKEND=gl` + `LIBGL_ALWAYS_SOFTWARE=1` 没用：S100 的 EGL/GLES 是
板厂私有驱动而非 Mesa，那些 Mesa 变量不生效，还是会落回硬件 GPU。）

## 修复（已生效）
1. 安装软件 Vulkan 渲染器 lavapipe：
   ```
   sudo apt install mesa-vulkan-drivers
   ```
2. 在 S100 的 `~/.bashrc` 和 `~/.zshrc` 写入（`MACHINE=RDKS100` 时自动导出）：
   ```
   export VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/lvp_icd.aarch64.json
   export WGPU_BACKEND=vulkan
   ```

之后 **新开一个终端** 再跑 `record.sh` / `tele.sh` / `rerun` 即可正常出窗口。
（lerobot 的 `--display_data=true` 会启动 `rerun` 子进程并继承终端环境变量。）

## 注意
- 这是 CPU 软件渲染，帧率偏低（演示够用）。终端会打印
  `Software rasterizer detected - expect poor performance`，属正常。
- 还是打不开时，先确认当前终端有变量：
  ```
  echo $VK_ICD_FILENAMES   # 应为 /usr/share/vulkan/icd.d/lvp_icd.aarch64.json
  echo $WGPU_BACKEND        # 应为 vulkan
  ```
  没有就重开终端，或手动 `source ~/.bashrc`。
- 单独测试：直接运行 `rerun`，能弹出空窗口就说明好了。
