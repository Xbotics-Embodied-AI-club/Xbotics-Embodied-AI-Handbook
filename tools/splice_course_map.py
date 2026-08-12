from pathlib import Path

readme_path = Path("README.md")
frag_path = Path("docs/_course_map_fragment.md")
readme = readme_path.read_text(encoding="utf-8")
frag = frag_path.read_text(encoding="utf-8")
start = readme.index('<span id="course-map"></span>')
end = readme.index('<span id="team"></span>')
new = readme[:start] + frag + "\n---\n\n" + readme[end:]
readme_path.write_text(new, encoding="utf-8")
print(f"spliced OK bytes={len(new)}")
