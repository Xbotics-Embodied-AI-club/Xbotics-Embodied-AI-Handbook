"""Run with: .venv/bin/python -m unittest discover -s docs/_ext -v"""

import base64
import hashlib
import io
import json
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
import unittest

from docutils import nodes
from markdown_it import MarkdownIt
from mdit_py_plugins.amsmath import amsmath_plugin
from mdit_py_plugins.dollarmath import dollarmath_plugin
from mdit_py_plugins.front_matter import front_matter_plugin
from sphinx.application import Sphinx

from handbook_compat import (
    HandbookSearchLanguage,
    latex_tree_to_text,
    normalize_source,
    repository_url,
    split_html_images,
)


def headings(text):
    return [token.tag for token in MarkdownIt().use(front_matter_plugin).parse(text)
            if token.type == "heading_open" and token.level == 0]


class SourceTests(unittest.TestCase):
    def test_existing_title_is_not_duplicated(self):
        source = '---\ntitle: "第 3 讲：旧标题"\nauthor: 作者\n---\n\n# 第 3 讲：完整标题\n\n## 内容\n'
        result = normalize_source(source, "part1-system-basics/03-example")
        self.assertEqual(headings(result), ["h1", "h2"])
        self.assertIn("author: 作者", result)
        self.assertNotIn("旧标题", result)
        self.assertIn("第 3 讲：完整标题", result)

    def test_metadata_title_wraps_pdf_sections(self):
        source = '---\ntitle: 第8讲 策略\nformat:\n  pdf: default\n---\n\n# 1 概念\n\n## 1.1 定义\n\n# 2 实验\n'
        result = normalize_source(source, "part3-end-to-end/08-policy")
        self.assertEqual(headings(result), ["h1", "h2", "h3", "h2"])
        self.assertNotIn("pdf:", result)
        self.assertIn("# 第8讲 策略", result)

    def test_later_h1_sections_preserve_earlier_hierarchy(self):
        source = '# 第 6 讲：感知\n\n## 正文\n\n### 小节\n\n# 参考资料\n\n## 论文\n'
        result = normalize_source(source, "part2-vision-manipulation/06-perception")
        self.assertEqual(headings(result), ["h1", "h2", "h3", "h2", "h3"])

    def test_code_examples_and_quoted_headings_unchanged(self):
        example = '````markdown\n# 示例不是标题\n```{=latex}\nraw\n```\n````\n'
        source = '# 第 17 讲：导航\n\n' + example + '\n> # 引用\n\n# 第二节\n'
        result = normalize_source(source, "part5-vln/17-vln")
        self.assertIn(example, result)
        self.assertIn("> # 引用", result)
        self.assertEqual(headings(result), ["h1", "h2"])

    def test_no_title_is_invented(self):
        source = '# 学习目标\n\n正文\n'
        self.assertEqual(normalize_source(source, "part5-vln/17-vln"), source)

    def test_equals_sign_in_math_is_not_a_setext_heading(self):
        source = '# 第 20 讲：世界模型\n\n## 理论\n\n$$\np(x)\n=\nq(x)\n$$\n\n### 下一节\n'
        self.assertEqual(normalize_source(source, "part6-agent-world-model/20-world-model"), source)

    def test_full_world_model_keeps_formulas_and_headings(self):
        path = Path(__file__).resolve().parents[1] / 'part6-agent-world-model/20-world-model.md'
        source = path.read_text()
        self.assertEqual(normalize_source(source, 'part6-agent-world-model/20-world-model'), source)

    def test_all_lecture_body_text_math_images_and_code_are_preserved(self):
        parser = (MarkdownIt().use(front_matter_plugin)
                  .use(dollarmath_plugin).use(amsmath_plugin))

        def payloads(text):
            tokens = parser.parse(text)
            return [(token.type, token.content) for i, token in enumerate(tokens)
                    if token.content and token.type != 'front_matter'
                    and not (i > 0 and tokens[i - 1].type == 'heading_open')]

        docs = Path(__file__).resolve().parents[1]
        for path in docs.glob('part[1-6]-*/*.md'):
            if path.name.startswith('00-'):
                continue
            with self.subTest(path=path.name):
                source = path.read_text()
                docname = path.relative_to(docs).with_suffix('').as_posix()
                self.assertEqual(payloads(normalize_source(source, docname)), payloads(source))

    def test_nonlecture_is_unchanged(self):
        source = '# 首页\n\n## 目录\n'
        self.assertEqual(normalize_source(source, "index"), source)

    def test_latex_fence_only(self):
        source = '# 第 9 讲：数据\n\n```{=latex}\nraw\n```\n'
        self.assertIn('```{handbook-latex}\nraw\n```', normalize_source(source, "part3-end-to-end/09-data"))

    def test_metadata_without_lecture_number(self):
        source = '---\ntitle: GRPO 后训练\n---\n\n# 1 从 PPO 到 GRPO\n'
        result = normalize_source(source, "part4-reinforcement-learning/15-grpo")
        self.assertIn("# 第 15 讲：GRPO 后训练", result)


class AssetTests(unittest.TestCase):
    def test_html_wrappers_and_captions_preserved(self):
        source = '<div align="center">\n<img src="fig/a.png" alt="A &amp; B > C" width="500"/>\n<p>图注</p></div>'
        raw = nodes.raw("", source, format="html")
        parts = split_html_images(raw)
        self.assertEqual(len(parts), 3)
        self.assertEqual(parts[0].astext(), '<div align="center">\n')
        self.assertIsInstance(parts[1], nodes.image)
        self.assertEqual(parts[1]["alt"], "A & B > C")
        self.assertEqual(parts[1]["width"], "500")
        self.assertEqual(parts[2].astext(), '\n<p>图注</p></div>')

    def test_multiple_images_preserve_remote_src(self):
        raw = nodes.raw("", '<img src="a.png"><br><img src="https://example.org/b.png">', format="html")
        images = [part for part in split_html_images(raw) if isinstance(part, nodes.image)]
        self.assertEqual([image["uri"] for image in images], ["a.png", "https://example.org/b.png"])

    def test_html_without_image_is_not_changed(self):
        raw = nodes.raw("", '<p>正文</p>', format="html")
        self.assertEqual(split_html_images(raw), [raw])

    def test_repository_links_keep_fragments_and_encode_spaces(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder).resolve()
            docs = root / "docs"
            docs.mkdir()
            target = root / "code" / "手眼 标定.py"
            target.parent.mkdir()
            target.write_text("# fixture\n")
            url = repository_url(target, root, docs, "https://github.com/o/r", "main", "说明")
            self.assertEqual(url, "https://github.com/o/r/blob/main/code/%E6%89%8B%E7%9C%BC%20%E6%A0%87%E5%AE%9A.py#%E8%AF%B4%E6%98%8E")
            self.assertEqual(url, repository_url(target, root, docs, "https://github.com/o/r", "main", "%E8%AF%B4%E6%98%8E"))
            self.assertIsNone(repository_url(root / "missing.py", root, docs, "x", "main"))
            self.assertIsNone(repository_url(root.parent, root, docs, "x", "main"))
            self.assertIsNone(repository_url(docs, root, docs, "x", "main"))
            self.assertEqual(repository_url(target.parent, root, docs, "https://github.com/o/r", "main"),
                             "https://github.com/o/r/tree/main/code")


class LatexTests(unittest.TestCase):
    def test_existing_tree_subset_keeps_annotations(self):
        source = "\\dirtree{%\n.1 \\texttt{<dataset>/}.\n.2 \\texttt{meta/} {\\color{TreeNoteColor}（说明书）}.\n.3 \\texttt{info.json}.\n}"
        result = latex_tree_to_text(source)
        self.assertEqual(result, '<dataset>/\n    meta/ （说明书）\n        info.json')

    def test_unknown_latex_is_not_silently_dropped(self):
        self.assertIsNone(latex_tree_to_text(r'\includegraphics{figure.png}'))
        self.assertIsNone(latex_tree_to_text('\\dirtree{%\n.1 \\texttt{root}.\n\\unknown\n}'))


class SearchTests(unittest.TestCase):
    def test_chinese_bigrams_and_latin(self):
        language = HandbookSearchLanguage({})
        terms = language.split("世界模型 VLA SO101")
        self.assertTrue({"世界", "界模", "模型", "世", "vla", "so101"}.issubset(terms))
        self.assertEqual(language.stem("VLA"), "vla")
        self.assertEqual(language.js_stemmer_rawcode, "")

    @unittest.skipUnless(shutil.which("node"), "Node.js is optional for query contract check")
    def test_browser_query_terms_exist_in_python_index(self):
        queries = ["世界模型", "具身智能 VLA", "PPO", "机器人", "图", "SO101", ""]
        language = HandbookSearchLanguage({})
        script = language.js_splitter_code + '\nconsole.log(JSON.stringify(' + json.dumps(queries) + '.map(splitQuery)))'
        actual = json.loads(subprocess.check_output(["node", "-e", script], text=True))
        self.assertEqual(actual[0], ["世界", "界模", "模型"])
        for query, terms in zip(queries, actual):
            self.assertTrue(set(terms).issubset(language.split(query)))


class BuildTests(unittest.TestCase):
    def test_build_copies_encoded_and_html_images_without_changing_sources(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            docs = root / "docs"
            chapter = docs / "part1-system-basics"
            chapter.mkdir(parents=True)
            (root / "README.md").write_text('# 仓库\n')
            image = root / "assets" / "图 1.png"
            image.parent.mkdir()
            image.write_bytes(base64.b64decode('iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+jRZkAAAAASUVORK5CYII='))
            (docs / "conf.py").write_text(
                "extensions = ['myst_parser', 'handbook_compat']\n"
                "root_doc = 'index'\nsource_suffix = {'.md': 'markdown'}\n"
                "myst_enable_extensions = ['html_image']\n"
                "myst_title_to_header = False\nhtml_search_language = 'handbook'\n"
                "html_theme_options = {'repository_url': 'https://github.com/o/r', 'repository_branch': 'main'}\n"
            )
            (docs / "index.md").write_text('# 首页\n\n```{toctree}\npart1-system-basics/01-example\n```\n')
            manuscript = chapter / "01-example.md"
            manuscript.write_text(
                '# 第 1 讲：世界模型\n\n'
                '![图](../../assets/%E5%9B%BE%201.png)\n\n'
                '<div><img src="../../assets/图 1.png" alt="图"><p>原图注</p></div>\n\n'
                '[仓库](../../README.md#说明)\n\n'
                '[图的原文件](../../assets/图%201.png)\n\n'
                '![仍需补齐](missing.png)\n'
            )
            before = hashlib.sha256(manuscript.read_bytes()).hexdigest()
            warnings = io.StringIO()
            app = Sphinx(str(docs), str(docs), str(root / "html"), str(root / "doctrees"),
                         "html", status=io.StringIO(), warning=warnings, freshenv=True)
            app.build(force_all=True)
            html = (root / "html/part1-system-basics/01-example.html").read_text()
            self.assertEqual(app.statuscode, 0, warnings.getvalue())
            self.assertIn('href="https://github.com/o/r/blob/main/README.md#', html)
            self.assertIn('原图注', html)
            self.assertIn('_images/', html)
            self.assertTrue(any((root / "html/_images").iterdir()))
            image_sources = re.findall(r'<img[^>]+src="([^"]+)"', html)
            self.assertEqual(sum('_images/' in src for src in image_sources), 2)
            self.assertNotIn('%E5%9B%BE%201.png [image.not_readable]', warnings.getvalue())
            self.assertIn('missing.png', warnings.getvalue())
            self.assertEqual(hashlib.sha256(manuscript.read_bytes()).hexdigest(), before)
            self.assertNotIn('ChineseStemmer', (root / "html/_static/language_data.js").read_text())


if __name__ == "__main__":
    unittest.main()
