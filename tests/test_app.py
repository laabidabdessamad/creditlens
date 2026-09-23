"""UI smoke test: runs the Streamlit script headlessly, routing its HTTP calls into the
FastAPI app in-process (no server needed)."""

import pytest
from fastapi.testclient import TestClient
from streamlit.testing.v1 import AppTest

from api.main import app


class _Resp:
    def __init__(self, r):
        self._r = r
        self.status_code = r.status_code
        self.ok = r.is_success

    def json(self):
        return self._r.json()

    def raise_for_status(self):
        self._r.raise_for_status()


def _path(url: str) -> str:
    """'https://host/some/path' -> '/some/path'"""
    return "/" + url.split("://", 1)[1].split("/", 1)[1]


@pytest.fixture()
def wired(monkeypatch):
    import requests

    with TestClient(app) as client:
        monkeypatch.setattr(requests, "get", lambda url, **kw: _Resp(client.get(_path(url))))
        monkeypatch.setattr(
            requests, "post", lambda url, json=None, **kw: _Resp(client.post(_path(url), json=json))
        )
        yield


def test_app_renders_and_scores(wired):
    at = AppTest.from_file("../app/streamlit_app.py", default_timeout=60).run()
    assert not at.exception
    at.button[0].click().run()
    assert not at.exception
    labels = {m.label: m.value for m in at.metric}
    assert labels["Probability of default"].endswith("%")
    assert [t.label for t in at.tabs][0] == "Assess an applicant"
