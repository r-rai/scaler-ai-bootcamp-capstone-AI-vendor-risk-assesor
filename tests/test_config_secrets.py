"""Settings must never reveal secrets through repr()/str() (e.g. when someone logs SETTINGS)."""
import dataclasses

from assessor.config import Settings


def test_repr_hides_secret_fields(monkeypatch):
    monkeypatch.setenv("TAVILY_API_KEY", "tvly-SENTINEL-do-not-print")
    s = Settings()
    assert s.tavily_api_key == "tvly-SENTINEL-do-not-print"  # still usable by code
    assert "SENTINEL" not in repr(s) and "SENTINEL" not in str(s)


def test_every_secret_looking_field_has_repr_off():
    """A future field named *api_key/password/secret must opt out of repr too."""
    bad = [f.name for f in dataclasses.fields(Settings)
           if any(w in f.name for w in ("api_key", "password", "secret")) and f.repr]
    assert not bad, f"secret-like fields would be printed by repr(SETTINGS): {bad}"
