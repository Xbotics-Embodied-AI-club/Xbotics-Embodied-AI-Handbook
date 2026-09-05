#!/usr/bin/env bash
# 重编讲义 PDF。**这个脚本是被 stdin 喂进编译机的**，不在本机跑。
#
# 首选 w1：它挂着 NAS，仓库直接可见（`mach info w1` 的 dataset 就是证据），
# 于是**一个文件都不用推、产物直接落在工作树里**。
#
#   $(mach ssh w1) 'bash -s' < render_pdfs_dev1.sh          # WORK 默认取仓库路径
#
# 退路是 dev1（不挂 NAS，得先把 md + assets 推过去）：
#   git -C handbook archive HEAD -- <pathspec…> | remote.sh dev1 'mkdir -p <W> && tar x -C <W>'
#   WORK=<W> remote.sh dev1 'bash -s' < render_pdfs_dev1.sh
#
# 为什么走 stdin 而不是内联 `ssh host 'cd X && quarto render …'`：
# 内联漏掉 cd 会得到误导性的「No valid input files passed to render」，
# 看着像输入不对，其实是 cwd 不对（xb-t6h 记的坑）。
#
# ⚠️ 九讲必须同一台机器出，或者至少同一套版本：不同的 quarto / TeX Live 排出来的
#    版式对不齐，混着编会得到一套看着像同一本、细看行距页码都不一样的 PDF。
#
# ⚠️ **编译链的位置不能当常量记 —— worker 是容器，重建一次工具链就没了。每次现场探测。**
#    2026-09-02 w1 崩过一次重建，quarto / xelatex / TinyTeX 全丢。
#    2026-09-05 复装：w1 与 w4 现在都是 **quarto 1.7.32 + TeX Live 2026**，
#    装在 `~/opt/quarto/bin` 与 `~/.TinyTeX/bin/x86_64-linux` —— 两条都不在默认 PATH 上，
#    所以下面两条路径都要 export；`command -v quarto` 不加它们会误报「没装」。
#    版本一致 ⇒ 两台出的 PDF 可以互换。**优先用 w1**：它挂着 NAS，一个文件都不用推；
#    w4 不挂 NAS，得推源、拉产物。
set -u

# 落点：不给就按挂着 NAS 的机器取仓库路径。w4 / dev1 这类不挂 NAS 的要显式传 WORK。
WORK="${WORK:-/mnt/nas_code/Xpersonal/Xbotics2/handbook}"
# 三个候选前缀都挂上：w4 装在 ~/opt，dev1 在 CPFS，将来哪台装在哪都先探再说。
export PATH="$HOME/opt/quarto/bin:/cpfs-normal/bianchen/setup/quarto/bin:$HOME/.TinyTeX/bin/x86_64-linux:$PATH"

command -v quarto  >/dev/null || { echo "★ quarto 不在 PATH"; exit 1; }
command -v xelatex >/dev/null || { echo "★ xelatex 不在 PATH"; exit 1; }
echo "quarto $(quarto --version) · $(xelatex --version | head -1)"

cd "$WORK" || { echo "★ $WORK 不存在 —— 先推文件"; exit 1; }

# 一讲一行：<md 相对路径>|<成品 PDF 文件名>
# 成品名去 .xelatex 中间层（xb-gef：编译产物直接叫 <主题>.pdf）。
RENDER_LIST=$(cat <<'LIST'
docs/part3-end-to-end/08-端到端策略导论.md|第8讲_端到端策略导论.pdf
docs/part3-end-to-end/09-操作数据闭环.md|第9讲_操作数据闭环.pdf
docs/part3-end-to-end/10-模仿学习实战.md|第10讲_模仿学习实战.pdf
docs/part3-end-to-end/11-VLA模型导览.md|第11讲_VLA模型导览.pdf
docs/part3-end-to-end/12-VLA微调实战.md|第12讲_VLA微调实战.pdf
docs/part3-end-to-end/13-VLA前沿.md|第13讲_VLA前沿.pdf
docs/part4-reinforcement-learning/14-强化学习入门.md|第14讲_强化学习入门.pdf
docs/part4-reinforcement-learning/15-GRPO后训练.md|第15讲_GRPO后训练.pdf
docs/part4-reinforcement-learning/16-真机强化学习.md|第16讲_真机强化学习.pdf
LIST
)

# 只渲某几讲：RENDER_ONLY 是匹配 md 路径的子串，多个用空格分开。
# 为什么要这个：改一讲就重渲七讲，会把另外六个没动过源的 PDF 二进制一并改掉，
# 提交里看不出哪个是这轮真改的。判据是「源动了才重渲」。
# 用法：RENDER_ONLY=16- ...  /  RENDER_ONLY="14- 16-" ...
if [ -n "${RENDER_ONLY:-}" ]; then
  filtered=""
  while IFS= read -r line; do
    [ -n "$line" ] || continue
    for pat in $RENDER_ONLY; do
      case "$line" in *"$pat"*) filtered="$filtered$line"$'\n'; break ;; esac
    done
  done <<< "$RENDER_LIST"
  [ -n "$filtered" ] || { echo "★ RENDER_ONLY='$RENDER_ONLY' 没匹配到任何一讲"; exit 1; }
  RENDER_LIST="$filtered"
  echo "只渲：$(echo "$RENDER_LIST" | grep -c . ) 讲（RENDER_ONLY=$RENDER_ONLY）"
fi

# 日志落一个自己的目录，**不散在仓库根**（第一版就是那样，七个 _log_*.txt 直接进了
# handbook 根，得手工清）。全绿就整个删掉；有失败就留着，路径在失败信息里给出。
LOGS="$WORK/.render-logs"
rm -rf "$LOGS"; mkdir -p "$LOGS"

fail=0
while IFS='|' read -r md pdf; do
  [ -n "$md" ] || continue
  [ -f "$md" ] || { echo "★ 缺输入 $md"; fail=1; continue; }
  dir=$(dirname "$md"); mkdir -p "$dir/pdf"
  echo "=== $(basename "$md") → $pdf  $(date +%H:%M:%S) ==="
  # ⚠️ `--output <名字>` 把产物落在 **cwd**，不是输入文件旁边（2026-08-31 w1 实测）。
  #    按"落在输入旁"去 mv 会得到 No such file，而 quarto 自己已经退 0 —— 看着像编成功了却没产物。
  if quarto render "$md" --to pdf --output "$pdf" >"$LOGS/$pdf.log" 2>&1; then
    [ -f "$pdf" ] || { echo "  ★ quarto 退 0 但找不到 $pdf"; fail=1; continue; }
    mv -f "$pdf" "$dir/pdf/$pdf"
    echo "  ok  $(du -h "$dir/pdf/$pdf" | cut -f1)"
  else
    echo "  ★ 失败，全文在 $LOGS/$pdf.log，尾部："; tail -20 "$LOGS/$pdf.log"
    fail=1
  fi
done <<< "$RENDER_LIST"

echo "=== 产物 ==="
ls -l docs/part3-end-to-end/pdf/ docs/part4-reinforcement-learning/pdf/ 2>/dev/null
echo "=== keep-tex 产生的 .tex ==="
ls -l docs/part3-end-to-end/*.tex docs/part4-reinforcement-learning/*.tex 2>/dev/null | wc -l

if [ "$fail" -eq 0 ]; then
  rm -rf "$LOGS"                      # 全绿就不留日志
  echo ALL_PDF_DONE
else
  echo "PDF_HAS_FAILURE —— 日志留在 $LOGS"; exit 1
fi
