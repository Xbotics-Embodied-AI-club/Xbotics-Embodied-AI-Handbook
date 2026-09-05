"""扫出讲义里用制表符拼的示意图 —— 那种图进 PDF 之后会静默垮掉。

## 为什么要有这个检查

制表符（U+2500 一族：`─│┌┐└┘├┤┬┴┼`）在讲义正文字体里**没有字形**。渲成 PDF 时
横线与箭头整条消失、框内文字错位，而 **LaTeX 不报错**：编译成功、页数正常，
只有真的去看那一页才发现图垮了。这类错误没有自动信号，所以要在源头拦。

一律改成真图：matplotlib 出图走 `assets/figures/` 与统一字体，或直接截图。

## 什么不算

- **文件树**（`├── models.py` 这种）：它是目录清单不是示意图，读者预期就是等宽文本，
  而且它只用竖线与折角、不靠横线连框，字形缺失时仍读得懂。
- **正文里的箭头**（`观察 → 决策 → 行动`）：那是行内文字，不是图。
★ **代码块内部照样要查。** 第一版把 ```` ``` ```` 围起来的块整个跳过，结果一条都没查出来 ——
  而讲12 那张垮掉的图恰恰就写在 ```` ```text ```` 里。等宽字体同样没有制表符字形，
  围栏不提供任何保护。这是「空结果被当成结论」的活例：检查器报零，不等于没有。

用法：`python tools/check_no_ascii_diagrams.py [<目录>]`   默认扫 `docs/`
      发现即以非零码退出，可直接挂进提交门。
"""

import re
import sys
from pathlib import Path

# 画框用的制表符。竖线与折角单独看，因为文件树只用它们。
BOX_H = set("─━═")                    # 横线：框图才用，文件树不用
BOX_OTHER = set("┌┐└┘├┤┬┴┼╔╗╚╝║│┃")   # 竖线与折角：文件树也用
# 文件树的写法是**恰好两条横线 + 空格 + 名字**：`├── models.py`。
# ★ 不能只匹前缀 `├──`：框图的 `└──────────` 也以它开头，第一版就是这么把
#   要抓的那类整个吞掉的 —— 检查器报零，而那一行明明在。第三个横线之后必须不是横线。
TREE_RE = re.compile(r"[├└]──(?!─)")


def offenders(path: Path) -> list[tuple[int, str]]:
    """找出一份 markdown 里像框图的行。

    Args:
        path: markdown 文件。

    Returns:
        `(行号, 行内容)` 列表。

    判据是**横线**：框图靠横线连出边框与箭头，文件树只有竖线和折角。
    所以一行里出现三个以上横线制表符、且不是文件树写法，就判为框图。
    """
    hits = []
    for i, ln in enumerate(path.read_text(errors="ignore").splitlines(), 1):
        if TREE_RE.search(ln):
            continue
        n_h = sum(c in BOX_H for c in ln)
        n_o = sum(c in BOX_OTHER for c in ln)
        if n_h >= 3 or (n_h >= 1 and n_o >= 2):
            hits.append((i, ln.strip()[:70]))
    return hits


def main(argv) -> int:
    root = Path(argv[0]) if argv else Path("docs")
    if not root.is_dir():
        sys.exit(f"★ {root} 不是目录")
    total = 0
    for f in sorted(root.rglob("*.md")):
        hits = offenders(f)
        if not hits:
            continue
        total += len(hits)
        print(f"\n{f}")
        for i, t in hits:
            print(f"  {i}: {t}")
    if total:
        print(f"\n★ {total} 行像用制表符拼的框图 —— 进 PDF 会静默垮掉，改成真图。")
        print("  文件树与正文箭头已排除；若确属误报，把那一行改成代码块或调整本脚本的判据。")
        return 1
    print("没有用制表符拼的框图。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
