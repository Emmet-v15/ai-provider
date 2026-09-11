"""
Unit tests for chat-model variant selection.

Unlike test_api.py these need no running server and no GPU: the variant
registry, the VRAM re-budgeting and the llama-server argv are all decided
before anything is spawned, which is exactly the part worth pinning down.
"""

import asyncio

import pytest

from model_manager import ModelManager
from providers import llm


def _fresh_provider() -> llm.LLMProvider:
    """A provider that is not the process-wide singleton."""
    return llm.LLMProvider()


def test_both_variants_are_registered():
    assert set(llm.CHAT_VARIANTS) == {"qwen3", "mistral"}


def test_the_weights_for_every_variant_are_on_disk():
    missing = [v.key for v in llm.CHAT_VARIANTS.values() if not v.present]
    assert not missing, f"no weights for: {missing}"


def test_only_qwen_carries_a_vision_projector():
    assert llm.CHAT_VARIANTS["qwen3"].vision is True
    assert llm.CHAT_VARIANTS["mistral"].vision is False


def test_selecting_a_variant_changes_the_active_one():
    prov = _fresh_provider()
    assert prov.variant.key == llm.DEFAULT_VARIANT
    prov.select("mistral")
    assert prov.variant.key == "mistral"


def test_an_unknown_variant_is_refused_by_name():
    prov = _fresh_provider()
    with pytest.raises(KeyError) as e:
        prov.select("gpt5")
    assert "gpt5" in str(e.value)
    assert prov.variant.key == llm.DEFAULT_VARIANT, "a failed select must not switch"


def test_a_variant_cannot_be_swapped_under_a_running_server():
    prov = _fresh_provider()

    class _Alive:
        def poll(self):
            return None

    prov._proc = _Alive()
    assert prov.is_running
    with pytest.raises(RuntimeError, match="unload"):
        prov.select("mistral")


def test_switching_variant_rebudgets_the_slot():
    """The budget must follow the weights, or a refused load becomes an OOM."""
    mgr = ModelManager()
    mgr.register("llm-chat", llm.CHAT_VARIANTS["qwen3"].vram_gb, load_fn=None)
    assert mgr.get("llm-chat").size_gb == llm.CHAT_VARIANTS["qwen3"].vram_gb

    mgr.set_size("llm-chat", llm.CHAT_VARIANTS["mistral"].vram_gb)
    assert mgr.get("llm-chat").size_gb == llm.CHAT_VARIANTS["mistral"].vram_gb
    assert (
        llm.CHAT_VARIANTS["mistral"].vram_gb != llm.CHAT_VARIANTS["qwen3"].vram_gb
    ), "the two variants should not budget identically"


def test_a_loaded_slot_cannot_be_rebudgeted():
    mgr = ModelManager()
    mgr.register("llm-chat", 18.0, load_fn=None)
    mgr.get("llm-chat").loaded = True
    with pytest.raises(RuntimeError, match="while it is loaded"):
        mgr.set_size("llm-chat", 21.0)


def _argv_for(variant_key: str, monkeypatch) -> list[str]:
    """Build the llama-server command line without spawning it."""
    captured: list[list[str]] = []

    class _FakePopen:
        def __init__(self, args, **kw):
            captured.append(args)
            self.returncode = 0

        def poll(self):
            return 0  # 'already exited' — start() bails out after building argv

    monkeypatch.setattr(llm.sp, "Popen", _FakePopen)
    prov = _fresh_provider()
    prov._variant = llm.CHAT_VARIANTS[variant_key]
    with pytest.raises(RuntimeError):
        asyncio.run(prov.start())
    return [str(a) for a in captured[-1]]


def test_each_variant_launches_with_its_own_weights_and_context(monkeypatch):
    qwen = _argv_for("qwen3", monkeypatch)
    assert "Qwen3.8-27B-UD-Q4_K_XL.gguf" in " ".join(qwen)
    assert qwen[qwen.index("-c") + 1] == str(llm.CHAT_VARIANTS["qwen3"].ctx)

    mistral = _argv_for("mistral", monkeypatch)
    assert "Dolphin-Mistral-24B-Venice-Edition-Q6_K.gguf" in " ".join(mistral)
    assert mistral[mistral.index("-c") + 1] == str(llm.CHAT_VARIANTS["mistral"].ctx)


def test_the_vision_projector_follows_the_variant(monkeypatch):
    """Handing Qwen's mmproj to a text-only Mistral would fail at load."""
    assert "--mmproj" in _argv_for("qwen3", monkeypatch)
    assert "--mmproj" not in _argv_for("mistral", monkeypatch)


def test_both_variants_share_one_port(monkeypatch):
    """They are mutually exclusive by construction — same llama-server, same port."""
    qwen = _argv_for("qwen3", monkeypatch)
    mistral = _argv_for("mistral", monkeypatch)
    assert qwen[qwen.index("--port") + 1] == mistral[mistral.index("--port") + 1]
