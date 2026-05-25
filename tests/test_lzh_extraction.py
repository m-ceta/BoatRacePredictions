from __future__ import annotations

from pathlib import Path

from src import live


class _FakeLhaArchive:
    def __init__(self, path: str) -> None:
        self.path = path

    def __enter__(self) -> "_FakeLhaArchive":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        return None

    def namelist(self) -> list[str]:
        return ["sample.txt"]

    def read(self, name: str) -> bytes:
        assert name == "sample.txt"
        assert Path(self.path).name == "program.lzh"
        return "番組表".encode("cp932")


class _FakeLhafileModule:
    Lhafile = _FakeLhaArchive


def test_extract_lzh_entries_prefers_lhafile(monkeypatch) -> None:
    monkeypatch.setattr(live, "lhafile", _FakeLhafileModule())
    monkeypatch.setattr(live, "find_seven_zip", lambda: None)

    extracted = live.extract_lzh_entries(b"dummy")

    assert extracted == [("sample.txt", "番組表".encode("cp932"))]


def test_extract_lzh_entries_falls_back_to_seven_zip(monkeypatch) -> None:
    monkeypatch.setattr(live, "lhafile", None)
    monkeypatch.setattr(live, "find_seven_zip", lambda: "7z.exe")
    monkeypatch.setattr(
        live,
        "extract_lzh_entries_with_seven_zip",
        lambda archive_bytes, seven_zip: [("sample.txt", b"fallback")],
    )

    extracted = live.extract_lzh_entries(b"dummy")

    assert extracted == [("sample.txt", b"fallback")]
