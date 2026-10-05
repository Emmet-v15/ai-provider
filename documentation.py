"""
The project's markdown documentation, served by the API itself.

The repo's ``.md`` files are the single source of truth: they are read from
disk on every request (so an edit shows up without a restart) and returned
either as raw markdown — for clients and agents — or rendered to HTML for a
browser.  Nothing here is copied into code, except that the cross-cutting
"Queueing & rate limits" section of API.md is also lifted into the OpenAPI
description so Swagger/ReDoc show it next to the endpoints.

Documents are addressed by a short name (``api``) or by their filename
(``API.md``, any case).  The filename alias is what keeps the files' own
relative links — ``[AGENTS.md](AGENTS.md)`` in README.md — working when
they are browsed under ``/documentation/``.
"""

from __future__ import annotations

import html
import os
import re
from dataclasses import dataclass

_BASE_DIR = os.path.dirname(os.path.abspath(__file__))


@dataclass(frozen=True)
class Doc:
    name: str
    filename: str
    title: str
    summary: str

    @property
    def path(self) -> str:
        return os.path.join(_BASE_DIR, self.filename)

    @property
    def url(self) -> str:
        return f"/documentation/{self.name}"

    def read(self) -> str:
        with open(self.path, encoding="utf-8") as f:
            return f.read()


# A fixed registry rather than a directory listing: a request can only ever
# name one of these files, never an arbitrary path.
DOCS: dict[str, Doc] = {
    d.name: d
    for d in (
        Doc("readme", "README.md", "Overview",
            "What the gateway is, its design principles and how to run it."),
        Doc("api", "API.md", "API reference",
            "Every endpoint with request/response examples, queueing and rate limits."),
        Doc("agents", "AGENTS.md", "Engineering notes",
            "Architecture, invariants, operations and troubleshooting."),
        Doc("skill", "SKILL.md", "Agent skill",
            "How an AI agent should use this API — also served raw at /SKILL.md."),
    )
}

_BY_FILENAME = {d.filename.lower(): d for d in DOCS.values()}


def find(name: str) -> Doc | None:
    """Look a document up by short name or filename, case-insensitively."""
    key = name.lower()
    return DOCS.get(key) or _BY_FILENAME.get(key)


def index() -> list[dict]:
    return [
        {"name": d.name, "title": d.title, "summary": d.summary,
         "file": d.filename, "url": d.url}
        for d in DOCS.values()
    ]


def section(markdown: str, heading: str) -> str | None:
    """The ``## heading`` section of a markdown document, heading included.

    Runs to the next ``##`` heading or ``---`` rule.
    """
    lines = markdown.splitlines()
    try:
        start = next(i for i, l in enumerate(lines) if l.strip() == f"## {heading}")
    except StopIteration:
        return None
    end = next(
        (i for i in range(start + 1, len(lines))
         if lines[i].startswith("## ") or lines[i].strip() == "---"),
        len(lines),
    )
    return "\n".join(lines[start:end]).strip()


def openapi_section(doc: str, heading: str) -> str:
    """A section for the OpenAPI description, or "" if it can't be found.

    Demoted one level (``##`` -> ``###``) so it nests under the API title.
    """
    try:
        text = section(DOCS[doc].read(), heading)
    except OSError:
        return ""
    return re.sub(r"^##", "###", text, flags=re.M) if text else ""


# ── rendering ─────────────────────────────────────────────────────────
def _slug(heading_html: str) -> str:
    """GitHub's anchor for a heading, so ``[x](#queueing--rate-limits)`` works."""
    text = html.unescape(re.sub(r"<[^>]+>", "", heading_html)).strip().lower()
    return re.sub(r"[^\w\- ]", "", text).replace(" ", "-")


def split_frontmatter(text: str) -> tuple[str | None, str]:
    """``(yaml, body)`` for a document that opens with a ``---`` block."""
    m = re.match(r"---\r?\n(.*?)\r?\n---\r?\n", text, re.S)
    return (m[1], text[m.end():]) if m else (None, text)


def _render_markdown(text: str) -> str:
    # Rendered as markdown, a skill's YAML header would come out as a rule
    # followed by a setext heading; show it as the metadata block it is.
    front, text = split_frontmatter(text)
    head = f"<pre><code>{html.escape(front)}</code></pre>" if front else ""
    return head + _render_markdown_body(text)


def _render_markdown_body(text: str) -> str:
    try:
        from markdown_it import MarkdownIt
    except ImportError:  # markdown-it-py arrives via rich; don't depend on it
        return f"<pre>{html.escape(text)}</pre>"
    out = MarkdownIt("commonmark").enable(["table", "strikethrough"]).render(text)
    return re.sub(
        r"<h([1-6])>(.*?)</h\1>",
        lambda m: f'<h{m[1]} id="{_slug(m[2])}">{m[2]}</h{m[1]}>',
        out,
    )


_PAGE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title} · AI Provider</title>
<style>
:root {{ --bg:#fff; --fg:#1f2328; --muted:#59636e; --line:#d1d9e0; --code:#f6f8fa; --link:#0969da; }}
@media (prefers-color-scheme: dark) {{
  :root {{ --bg:#0d1117; --fg:#e6edf3; --muted:#9198a1; --line:#3d444d; --code:#151b23; --link:#4493f8; }}
}}
* {{ box-sizing: border-box; }}
body {{ margin:0; background:var(--bg); color:var(--fg);
  font:16px/1.6 -apple-system,BlinkMacSystemFont,"Segoe UI",Helvetica,Arial,sans-serif; }}
main {{ max-width:960px; margin:0 auto; padding:24px 16px 64px; }}
nav {{ display:flex; flex-wrap:wrap; gap:4px 16px; padding-bottom:12px;
  border-bottom:1px solid var(--line); font-size:14px; }}
nav a[aria-current] {{ font-weight:600; color:var(--fg); }}
a {{ color:var(--link); text-decoration:none; }} a:hover {{ text-decoration:underline; }}
h1,h2 {{ border-bottom:1px solid var(--line); padding-bottom:.3em; }}
code,pre {{ font:13.6px/1.45 ui-monospace,SFMono-Regular,Consolas,monospace; background:var(--code); }}
code {{ padding:.2em .4em; border-radius:6px; }}
pre {{ padding:16px; border-radius:6px; overflow-x:auto; }} pre code {{ padding:0; }}
table {{ border-collapse:collapse; display:block; overflow-x:auto; }}
th,td {{ border:1px solid var(--line); padding:6px 13px; }}
.muted {{ color:var(--muted); }}
</style>
</head>
<body><main>
<nav>{nav}</nav>
{body}
</main></body>
</html>
"""


def _nav(current: str | None) -> str:
    links = [("Index", "/documentation", current == "index")]
    links += [(d.title, d.url, d.name == current) for d in DOCS.values()]
    links += [("Swagger UI", "/docs", False), ("ReDoc", "/redoc", False),
              ("OpenAPI JSON", "/openapi.json", False)]
    return "".join(
        f'<a href="{u}"{" aria-current=page" if cur else ""}>{html.escape(t)}</a>'
        for t, u, cur in links
    )


def render_doc(doc: Doc) -> str:
    return _PAGE.format(title=html.escape(doc.title), nav=_nav(doc.name),
                        body=_render_markdown(doc.read()))


def render_index() -> str:
    items = "".join(
        f'<li><a href="{d.url}">{html.escape(d.title)}</a> '
        f'<span class="muted">— {html.escape(d.summary)} (<code>{d.filename}</code>)</span></li>'
        for d in DOCS.values()
    )
    body = (
        "<h1>AI Provider documentation</h1>"
        f"<ul>{items}</ul>"
        "<p>Interactive endpoint reference: <a href=\"/docs\">Swagger UI</a> · "
        "<a href=\"/redoc\">ReDoc</a> · <a href=\"/openapi.json\">OpenAPI JSON</a>.</p>"
        "<p class=\"muted\">Every page is also available as raw markdown: "
        "<code>?format=md</code>, or any request that does not prefer "
        "<code>text/html</code>.</p>"
    )
    return _PAGE.format(title="Documentation", nav=_nav("index"), body=body)


def wants_html(accept: str | None, fmt: str | None) -> bool:
    """``?format=`` wins; otherwise HTML only for clients that ask for it (browsers)."""
    if fmt:
        return fmt.lower() == "html"
    return "text/html" in (accept or "")
