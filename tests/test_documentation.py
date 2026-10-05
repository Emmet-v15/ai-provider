"""
The docs served by the API — no running server or GPU needed.

TestClient without a ``with`` block skips the lifespan, so nothing is
loaded and no orphan cleanup runs.
"""

import re

import pytest
from fastapi.testclient import TestClient

import documentation
import server

client = TestClient(server.app)
BROWSER = {"accept": "text/html,application/xhtml+xml,*/*;q=0.8"}


def test_index_lists_every_document_with_a_working_url():
    r = client.get("/documentation")
    assert r.status_code == 200
    names = [d["name"] for d in r.json()["documents"]]
    assert names == list(documentation.DOCS)
    for d in r.json()["documents"]:
        assert client.get(d["url"]).status_code == 200


def test_root_is_the_index():
    assert client.get("/").json() == client.get("/documentation").json()


@pytest.mark.parametrize("doc", documentation.DOCS.values(), ids=lambda d: d.name)
def test_each_document_is_served_verbatim_as_markdown(doc):
    r = client.get(f"/documentation/{doc.name}")
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("text/markdown")
    with open(doc.path, encoding="utf-8") as f:
        assert r.text == f.read()


def test_documents_are_addressable_by_filename_too():
    assert client.get("/documentation/API.md").text == client.get("/documentation/api").text
    assert client.get("/documentation/agents.md").status_code == 200


def test_browsers_get_html_and_format_overrides():
    r = client.get("/documentation/api", headers=BROWSER)
    assert r.headers["content-type"].startswith("text/html")
    assert '<h1 id="ai-provider-api">AI Provider API</h1>' in r.text
    # In-page links like API.md's [Documentation](#documentation) need ids.
    assert 'id="queueing--rate-limits"' in r.text
    for anchor in re.findall(r'href="#([^"]+)"', r.text):
        assert f'id="{anchor}"' in r.text, f"dangling in-page link #{anchor}"
    assert "<table>" in r.text  # GFM tables render
    r = client.get("/documentation/api?format=md", headers=BROWSER)
    assert r.headers["content-type"].startswith("text/markdown")
    r = client.get("/documentation/api?format=html")
    assert r.headers["content-type"].startswith("text/html")
    assert client.get("/documentation", headers=BROWSER).headers["content-type"].startswith("text/html")


def test_unknown_document_is_404_and_paths_cannot_escape():
    r = client.get("/documentation/nope")
    assert r.status_code == 404
    assert "readme" in r.json()["detail"]
    for evil in ("..%2Fserver.py", "%2E%2E%2F.env", "server.py", "popcorn4_AGENTS.md"):
        assert client.get(f"/documentation/{evil}").status_code == 404


@pytest.mark.parametrize("doc", documentation.DOCS.values(), ids=lambda d: d.name)
def test_relative_links_between_documents_resolve_through_the_api(doc):
    """README's [API.md](API.md) must work when browsed at /documentation/readme."""
    for target in re.findall(r"\]\(([^)#:\s]+\.md)(?:#[^)]*)?\)", doc.read()):
        r = client.get(f"/documentation/{target}")
        assert r.status_code == 200, f"{doc.filename} links to {target}, which is not served"


def test_openapi_carries_the_rate_limit_guide_and_responses():
    spec = client.get("/openapi.json").json()
    desc = spec["info"]["description"]
    assert "### Queueing & rate limits" in desc
    assert "Retry-After" in desc
    assert "/documentation" in desc
    chat = spec["paths"]["/v1/chat/completions"]["post"]["responses"]
    assert "Retry-After" in chat["429"]["headers"]
    assert "409" in chat
    assert "/documentation/{name}" in spec["paths"]


def test_section_extraction_stops_at_the_next_heading():
    md = "# T\n\n## A\none\n\n## B\ntwo\n"
    assert documentation.section(md, "A") == "## A\none"
    assert documentation.section(md, "B") == "## B\ntwo"
    assert documentation.section(md, "C") is None


# ── SKILL.md ──────────────────────────────────────────────────────────

def test_skill_md_is_always_raw_markdown_even_for_browsers():
    r = client.get("/SKILL.md", headers=BROWSER)
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("text/markdown")
    assert r.text == documentation.DOCS["skill"].read()


def test_skill_md_has_valid_agent_skill_frontmatter():
    front, body = documentation.split_frontmatter(documentation.DOCS["skill"].read())
    assert front is not None, "SKILL.md must open with a --- YAML block"
    fields = dict(
        (k.strip(), v.strip()) for k, v in
        (line.split(":", 1) for line in front.splitlines() if ":" in line)
    )
    assert re.fullmatch(r"[a-z0-9-]{1,64}", fields.get("name", "")), fields.get("name")
    assert 0 < len(fields.get("description", "")) <= 1024
    assert body.lstrip().startswith("# ")


def _route_patterns():
    for route in server.app.routes:
        path = getattr(route, "path", None)
        if path:
            yield re.compile("^" + re.sub(r"\{[^}]+\}", r"[^/]+", path) + "$")


def test_every_endpoint_the_skill_mentions_exists():
    """The skill must never send an agent to an endpoint that isn't there."""
    text = documentation.DOCS["skill"].read()
    mentioned = set(re.findall(r"\b(?:GET|POST|PATCH|DELETE) (/[^\s`?]+)", text))
    mentioned |= set(re.findall(r"emperor\.empirenet:8765(/[^\s`'\"?]+)", text))
    assert len(mentioned) > 10
    patterns = list(_route_patterns())
    missing = sorted(p for p in mentioned if not any(rx.match(p) for rx in patterns))
    assert not missing, f"SKILL.md mentions endpoints the API doesn't have: {missing}"


def test_skill_frontmatter_renders_as_a_block_not_a_heading():
    r = client.get("/documentation/skill", headers=BROWSER)
    assert "<pre><code>name: ai-provider" in r.text
    assert "<h2" not in r.text.split("<h1", 1)[0], "frontmatter leaked into headings"
