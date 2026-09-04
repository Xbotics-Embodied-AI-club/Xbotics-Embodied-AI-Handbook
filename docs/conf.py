"""Sphinx 网站配置；不修改或执行现有 Quarto / PDF 编译流程。"""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent / "_ext"))

project = "Xbotics 具身智能教程"
author = "Xbotics 具身智能社区"
language = "zh_CN"
root_doc = "index"
source_suffix = {".md": "markdown"}

extensions = [
    "myst_parser",
    "sphinx.ext.mathjax",
    "sphinx_copybutton",
    "sphinxcontrib.mermaid",
    "handbook_compat",
]

# 网站只读取课程和阅读指南，不执行实验代码或笔记本。
include_patterns = [
    "index.md",
    "SUMMARY.md",
    "style-guide.md",
    "00-preface/*.md",
    "part[1-6]-*/*.md",
    "appendix/*.md",
]
exclude_patterns = [
    "_build",
    ".quarto",
    "_filters",
    "superpowers",
    "**/_drafts/**",
    "**/draft-*.md",
    "**/.DS_Store",
]

# 兼容现有正文的公式、图片宽度属性和 HTML 图片，保留 PDF 源稿。
myst_enable_extensions = [
    "amsmath",
    "attrs_inline",
    "colon_fence",
    "dollarmath",
    "html_image",
    "tasklist",
]
myst_heading_anchors = 4
# 由兼容层按现有讲稿标题生成单一页面标题，避免与 YAML title 重复。
myst_title_to_header = False
myst_fence_as_directive = ["math", "mermaid"]

# 中英混合索引与浏览器分词保持一致，不依赖远端搜索服务。
html_search_language = "handbook"
mermaid_version = "11.12.1"
# 由 Sphinx 扩展统一触发渲染，避免 Mermaid 自动启动与主题切换重复渲染。
mermaid_init_config = {"startOnLoad": False, "securityLevel": "strict"}

html_theme = "sphinx_book_theme"
html_title = project
html_context = {"default_mode": "auto"}
html_theme_options = {
    "repository_url": (
        "https://github.com/Xbotics-Embodied-AI-club/"
        "Xbotics-Embodied-AI-Handbook"
    ),
    "repository_branch": "main",
    "path_to_docs": "docs",
    "use_repository_button": True,
    "use_edit_page_button": True,
    "show_toc_level": 2,
    "show_navbar_depth": 1,
}
