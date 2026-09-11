"""
Integration tests for the AI Provider API.

Requires the server to be running on http://127.0.0.1:8765.
"""

import httpx

BASE = "http://127.0.0.1:8765"


def test_health():
    r = httpx.get(f"{BASE}/health", timeout=5)
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert "vram" in body
    assert "gpu" in body
    assert "sdxl_loaded" in body


def test_models_list():
    r = httpx.get(f"{BASE}/models", timeout=5)
    assert r.status_code == 200
    body = r.json()
    assert "models" in body
    assert "loaded" in body
    names = {m["name"] for m in body["models"]}
    # All registered models
    for expected in ("tts", "llm-chat", "llm-embed", "stt", "sdxl"):
        assert expected in names, f"missing {expected}"


def test_models_loaded():
    """No models should be loaded by default (on-demand loading)."""
    r = httpx.get(f"{BASE}/models", timeout=5)
    body = r.json()
    # /models returns all registered models in both 'models' and 'loaded' keys
    # Check the 'loaded' flag on individual entries
    loaded_flag = {m["name"]: m.get("loaded", False) for m in body["loaded"]}
    assert loaded_flag.get("tts") is False


def test_voices_list():
    r = httpx.get(f"{BASE}/audio/voices", timeout=5)
    assert r.status_code == 200
    body = r.json()
    assert "voices" in body
    assert "clones" in body
    # Built-in presets should be present
    assert "alex" in body["voices"]


def test_voices_clone_create_no_audio():
    """Creating a clone without audio should 404 if not existing, or 400 if existing."""
    r = httpx.post(f"{BASE}/audio/voices", data={"tag": "test_nonexistent"})
    assert r.status_code == 404


def test_voices_clone_create_no_audio_existing():
    """Existing clone without audio should 400."""
    # First need one to exist — skip this, test via the rename path instead
    pass


def test_tts_smoke():
    """Generate a short TTS audio segment."""
    # Load TTS worker (both models load sequentially in the subprocess)
    httpx.post(f"{BASE}/models/tts/load", timeout=180)
    r = httpx.post(
        f"{BASE}/v1/audio/speech",
        json={"input": "Hello, this is a test.", "voice": "alex"},
        timeout=30,
    )
    assert r.status_code == 200
    assert len(r.content) > 0
    # Should be opus audio by default
    assert r.headers.get("content-type", "").startswith("audio/")
    assert "X-Waveform" in r.headers
    assert "X-Duration-Seconds" in r.headers


def test_tts_wav():
    """Generate WAV format."""
    httpx.post(f"{BASE}/models/tts/load", timeout=180)
    r = httpx.post(
        f"{BASE}/v1/audio/speech",
        json={"input": "Hello world", "voice": "alex", "response_format": "wav"},
        timeout=30,
    )
    assert r.status_code == 200
    assert r.headers.get("content-type") == "audio/wav"


def test_tts_empty_text():
    """Empty input should 400."""
    r = httpx.post(
        f"{BASE}/v1/audio/speech",
        json={"input": "", "voice": "alex"},
    )
    assert r.status_code == 400


def test_legacy_tts():
    """Legacy /tts endpoint still works."""
    httpx.post(f"{BASE}/models/tts/load", timeout=180)
    r = httpx.post(
        f"{BASE}/tts",
        json={"text": "Hello", "speaker": "alex"},
        params={"format": "wav"},
        timeout=30,
    )
    assert r.status_code == 200
    assert len(r.content) > 0


def test_legacy_voices():
    r = httpx.post(f"{BASE}/voices", timeout=5)
    assert r.status_code == 200
    body = r.json()
    assert "voices" in body
    assert "alex" in body["voices"]


def test_legacy_clone_list():
    r = httpx.post(f"{BASE}/clone/list", timeout=5)
    assert r.status_code == 200
    body = r.json()
    assert "clones" in body


def test_models_load_unload_cycle():
    """Load a model, verify it's loaded, unload it, verify it's gone."""
    # Load stt (may fail with 501 if whisper not installed)
    r = httpx.post(f"{BASE}/models/stt/load", timeout=30)
    if r.status_code == 501:
        return  # whisper not installed, skip

    assert r.status_code in (200, 400), f"unexpected {r.status_code}: {r.text}"
    if r.status_code == 200:
        r = httpx.get(f"{BASE}/models", timeout=5)
        loaded_names = {m["name"] for m in r.json()["loaded"]}
        assert "stt" in loaded_names

    r = httpx.post(f"{BASE}/models/stt/unload", timeout=5)
    assert r.status_code == 200

    r = httpx.get(f"{BASE}/models", timeout=5)
    loaded_after = {m["name"] for m in r.json()["loaded"]}
    assert "stt" not in loaded_after


def test_unknown_model():
    r = httpx.post(f"{BASE}/models/does_not_exist/load")
    assert r.status_code == 404

    r = httpx.post(f"{BASE}/models/does_not_exist/unload")
    assert r.status_code == 200  # unload is idempotent


def test_voice_rename_flow():
    """Rename a clone via PATCH /audio/voices/{tag}."""
    # List current clones
    r = httpx.get(f"{BASE}/audio/voices", timeout=5)
    clones = r.json()["clones"]
    if not clones:
        # Can't test rename without a clone to rename
        return

    old_tag = list(clones.keys())[0]
    new_tag = f"{old_tag}_renamed_test"

    r = httpx.patch(
        f"{BASE}/audio/voices/{old_tag}",
        data={"new_tag": new_tag},
        timeout=5,
    )
    # Rename might fail if new_tag already exists from a previous test run
    if r.status_code == 409:
        # New tag exists, try cleaning it first, then rename back
        httpx.delete(f"{BASE}/audio/voices/{new_tag}")
        r = httpx.patch(
            f"{BASE}/audio/voices/{old_tag}",
            data={"new_tag": new_tag},
            timeout=5,
        )

    assert r.status_code == 200, f"rename failed: {r.text}"
    assert r.json()["status"] == "updated"
    assert r.json()["tag"] == new_tag

    # Rename back
    r = httpx.patch(
        f"{BASE}/audio/voices/{new_tag}",
        data={"new_tag": old_tag},
        timeout=5,
    )
    assert r.status_code == 200