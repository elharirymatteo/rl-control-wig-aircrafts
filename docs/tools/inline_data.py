"""Copy docs/data/*.json into docs/index.html, between the DATA:BEGIN and DATA:END markers.

    .venv/bin/python docs/tools/inline_data.py

Run after build_data.py or fly_altitude.py. The page reads its data from these inline blocks, so it
works without fetching data/: in the anonymized repository's HTML viewer the page runs in a
sandboxed, opaque-origin frame that cannot fetch files next to it.
"""
import re
from pathlib import Path

DOCS = Path(__file__).resolve().parents[1]
PAGE = DOCS / "index.html"
FILES = ["results.json", "throughput.json", "altitude_replay.json", "maneuvers.json"]


def main():
    blocks = []
    for name in FILES:
        text = (DOCS / "data" / name).read_text(encoding="utf-8").strip()
        text = text.replace("</", "<\\/")       # a JSON string must not close the script element
        blocks.append(f'<script type="application/json" data-src="data/{name}">{text}</script>')
    html = PAGE.read_text(encoding="utf-8")
    pat = re.compile(r"(<!-- DATA:BEGIN[^>]*-->\n).*?(<!-- DATA:END -->)", re.S)
    if len(pat.findall(html)) != 1:
        raise SystemExit("index.html: expected exactly one DATA:BEGIN/DATA:END block")
    html = pat.sub(lambda m: m.group(1) + "\n".join(blocks) + "\n" + m.group(2), html)
    PAGE.write_text(html, encoding="utf-8")
    print(f"inlined {len(FILES)} files into {PAGE.relative_to(DOCS.parent)}  {PAGE.stat().st_size // 1024} KiB")


if __name__ == "__main__":
    main()
