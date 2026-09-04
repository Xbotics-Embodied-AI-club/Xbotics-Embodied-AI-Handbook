"""Website-only adaptations. Never write back to lecture, image, or PDF sources.

The source-read hook changes parsed heading/fence lines, not code examples.
The transform runs before Sphinx collects images and downloadable files.
Missing assets are deliberately left unresolved, with their original warnings.
"""

from __future__ import annotations

from html.parser import HTMLParser
from pathlib import Path
import re
from urllib.parse import quote, unquote, urlsplit

from docutils import nodes
from markdown_it import MarkdownIt
from mdit_py_plugins.amsmath import amsmath_plugin
from mdit_py_plugins.dollarmath import dollarmath_plugin
from mdit_py_plugins.front_matter import front_matter_plugin
from sphinx import addnodes
from sphinx.search import SearchLanguage
from sphinx.transforms import SphinxTransform
from sphinx.util import logging
from sphinx.util.docutils import SphinxDirective
import yaml

LOGGER = logging.getLogger(__name__)
HAN_OR_LATIN = re.compile(r"[\u3400-\u9fff]+|[a-zA-Z0-9_]+")
LECTURE = re.compile(r"part[1-6]-[^/]+/(\d{2})-[^/]+$")


class HandbookSearchLanguage(SearchLanguage):
    """Deterministic Chinese bigrams and case-insensitive Latin technical terms.

Single Han characters are also indexed for one-character queries. Chinese
queries use adjacent pairs, so browser and Python need no separate dictionary.
English terms are exact tokens, without stemming/plural expansion.
"""

    lang = "handbook"
    language_name = "Chinese and English"
    js_stemmer_code = """
window.Stemmer = function () {
  this.stemWord = function (word) { return word.toLowerCase(); };
};
"""
    js_splitter_code = r"""
function splitQuery(query) {
  const parts = query.toLowerCase().match(/[\u3400-\u9fff]+|[a-z0-9_]+/g) || [];
  return parts.flatMap(function (part) {
    if (!/^[\u3400-\u9fff]+$/.test(part) || part.length === 1) return [part];
    return Array.from({length: part.length - 1}, function (_, i) {
      return part.slice(i, i + 2);
    });
  });
}
"""

    def split(self, text: str) -> list[str]:
        result = []
        for part in HAN_OR_LATIN.findall(text.lower()):
            if "\u3400" <= part[0] <= "\u9fff":
                result.extend(part)
                result.extend(part[i : i + 2] for i in range(len(part) - 1))
            else:
                result.append(part)
        return result

    def stem(self, word: str) -> str:
        return word.lower()


def normalize_source(text: str, docname: str) -> str:
    """Normalize only structural tokens; keep fenced examples byte-for-byte."""
    parser = (MarkdownIt("commonmark").use(front_matter_plugin)
              .use(dollarmath_plugin).use(amsmath_plugin))
    tokens = parser.parse(text)
    lines = text.splitlines(keepends=True)
    edits = []
    title = ""
    front_end = 0
    lecture = LECTURE.fullmatch(docname)

    if tokens and tokens[0].type == "front_matter" and lecture:
        front = tokens[0]
        data = yaml.safe_load(front.content)
        if isinstance(data, dict):
            title = str(data.get("title", ""))
            # Preserve author/lang and other metadata, but do not render PDF
            # configuration or let a second title override the website title.
            website_data = {
                key: value for key, value in data.items()
                if key not in {"title", "format", "filters", "execute", "header-includes"}
            }
            start, front_end = front.map
            replacement = ""
            if website_data:
                replacement = "---\n" + yaml.safe_dump(
                    website_data, allow_unicode=True, sort_keys=False
                ) + "---\n"
            # Retain line numbers wherever possible for useful build warnings.
            replacement += "\n" * max(0, front_end - replacement.count("\n"))
            edits.append((start, front_end, replacement))

    headings = [
        (token, tokens[i + 1].content)
        for i, token in enumerate(tokens)
        if token.type == "heading_open" and token.level == 0
    ]
    if lecture and headings:
        number = int(lecture.group(1))
        first, first_text = headings[0]
        has_title = first.tag == "h1" and re.match(r"第\s*\d+\s*讲", first_text)
        if not has_title and not title:
            # Do not invent a title for an unknown source structure.
            return text
        shift = 0 if has_title else 1
        for i, (token, content) in enumerate(headings):
            level = int(token.tag[1])
            if has_title and i == 0:
                continue
            if level == 1:
                shift = 1
            if shift:
                start, end = token.map
                edits.append((start, end, "#" * min(6, level + 1) + " " + content + "\n"))
        if not has_title:
            if not re.match(r"第\s*\d+\s*讲", title):
                title = f"第 {number} 讲：{title}"
            edits.append((front_end, front_end, f"\n# {title}\n\n"))

    for token in tokens:
        if token.type == "fence" and token.info.strip() == "{=latex}":
            start, _ = token.map
            edits.append((start, start + 1, lines[start].replace("{=latex}", "{handbook-latex}", 1)))

    for start, end, replacement in sorted(edits, reverse=True):
        lines[start:end] = [replacement]
    return "".join(lines)


def adapt_source(app, docname, source):
    if app.builder.format == "html":
        source[0] = normalize_source(source[0], docname)


def latex_tree_to_text(source: str) -> str | None:
    """Convert the existing dirtree subset, or refuse instead of losing text."""
    lines = [line.strip() for line in source.splitlines() if line.strip()]
    if not lines or lines[0] not in {r"\dirtree{%", r"\dirtree{"} or lines[-1] != "}":
        return None
    entry = re.compile(
        r"\.(\d+)\s+\\texttt\{([^{}]*)\}"
        r"(?:\s+\{\\color\{[^{}]+\}([^{}]*)\})?\."
    )
    result = []
    previous = 0
    for line in lines[1:-1]:
        match = entry.fullmatch(line)
        if not match:
            return None
        level, label, note = int(match[1]), match[2], match[3]
        if level < 1 or level > previous + 1:
            return None
        result.append("    " * (level - 1) + label + (" " + note if note else ""))
        previous = level
    return "\n".join(result) if result else None


class WebsiteLatexDirective(SphinxDirective):
    has_content = True

    def run(self):
        source = "\n".join(self.content)
        rendered = latex_tree_to_text(source)
        if rendered is None:
            LOGGER.warning(
                "PDF-only LaTeX retained as source; no safe website conversion found",
                location=self.get_location(), type="handbook", subtype="latex",
            )
        block = nodes.literal_block(source, rendered if rendered is not None else source)
        block["language"] = "text" if rendered is not None else "latex"
        self.set_source_info(block)
        return [block]


class ImageTags(HTMLParser):
    """Locate img tags without reserializing surrounding HTML or captions."""

    def __init__(self, source):
        super().__init__(convert_charrefs=False)
        self.offsets = [0]
        for line in source.splitlines(keepends=True):
            self.offsets.append(self.offsets[-1] + len(line))
        self.images = []
        self.feed(source)

    def handle_starttag(self, tag, attrs):
        if tag != "img":
            return
        attrs = dict(attrs)
        if not attrs.get("src"):
            return
        line, column = self.getpos()
        start = self.offsets[line - 1] + column
        self.images.append((start, start + len(self.get_starttag_text()), attrs))

    handle_startendtag = handle_starttag


def split_html_images(node):
    text = node.astext()
    tags = ImageTags(text).images
    if not tags:
        return [node]
    result = []
    cursor = 0
    for start, end, attrs in tags:
        if cursor < start:
            result.append(nodes.raw("", text[cursor:start], format="html"))
        image = nodes.image(text[start:end], uri=attrs["src"], alt=attrs.get("alt") or "")
        for key in ("width", "height"):
            if attrs.get(key):
                image[key] = attrs[key]
        if attrs.get("class"):
            image["classes"] = attrs["class"].split()
        if attrs.get("align") in {"left", "center", "right"}:
            image["align"] = attrs["align"]
        result.append(image)
        cursor = end
    if cursor < len(text):
        result.append(nodes.raw("", text[cursor:], format="html"))
    for child in result:
        child.source, child.line = node.source, node.line
    return result


def repository_url(target: Path, root: Path, docs: Path, base: str, branch: str, fragment=""):
    """Link existing non-doc repository content; never guess missing targets."""
    target = target.resolve()
    if not target.is_relative_to(root) or target.is_relative_to(docs) or not target.exists():
        return None
    kind = "tree" if target.is_dir() else "blob"
    path = quote(target.relative_to(root).as_posix(), safe="/")
    url = f"{base.rstrip('/')}/{kind}/{quote(branch, safe='')}/{path}"
    return url + ("#" + quote(unquote(fragment), safe="-_.~") if fragment else "")


class HandbookTransform(SphinxTransform):
    default_priority = 700

    def apply(self, **kwargs):
        docs = Path(self.env.srcdir).resolve()
        root = docs.parent
        docname = self.env.current_document.docname

        for raw in list(self.document.findall(nodes.raw)):
            if raw.get("format") == "html" and "<img" in raw.astext().lower():
                raw.replace_self(split_html_images(raw))

        for image in list(self.document.findall(nodes.image)):
            parts = urlsplit(image["uri"])
            if parts.scheme or parts.netloc:
                continue
            decoded = unquote(parts.path)
            _, absolute = self.env.relfn2path(decoded, docname)
            target = Path(absolute).resolve()
            if not target.is_relative_to(root):
                LOGGER.warning("Image outside repository is not published", location=image,
                               type="handbook", subtype="unsafe_image")
                image.replace_self(nodes.inline("", image.get("alt", "")))
            elif target.is_file():
                image["uri"] = decoded

        options = self.config.html_theme_options
        base = options["repository_url"]
        branch = options.get("repository_branch", "main")
        references = list(self.document.findall(addnodes.pending_xref))
        references += list(self.document.findall(addnodes.download_reference))
        for reference in references:
            raw_target = reference.get("reftarget", "")
            fragment = reference.get("reftargetid") or ""
            if reference.get("refdomain") == "doc":
                # MyST uses absolute document names for Markdown outside docs/.
                if not Path(raw_target).is_absolute():
                    continue
                target = Path(raw_target + ".md")
            else:
                parts = urlsplit(raw_target)
                if parts.scheme or parts.netloc or not parts.path:
                    continue
                fragment = parts.fragment
                _, absolute = self.env.relfn2path(unquote(parts.path), docname)
                target = Path(absolute)
            url = repository_url(target, root, docs, base, branch, fragment)
            if url:
                link = nodes.reference("", "", refuri=url)
                link.source, link.line = reference.source, reference.line
                link.extend(child.deepcopy() for child in reference.children)
                reference.replace_self(link)


def enable_html_transform(app):
    if app.builder.format == "html":
        app.add_transform(HandbookTransform)


def setup(app):
    app.add_search_language(HandbookSearchLanguage)
    app.add_directive("handbook-latex", WebsiteLatexDirective)
    app.connect("builder-inited", enable_html_transform)
    app.connect("source-read", adapt_source)
    return {"version": "1.0", "parallel_read_safe": True, "parallel_write_safe": True}
