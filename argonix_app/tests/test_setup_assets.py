"""tools/setup_assets.py covers every data/ file the committed configs reference (no network: listing only)."""
import json
import re

import pytest

from tools import setup_assets as S

# referenced by configs but deliberately not fetched (see the script's docstring)
NOT_FETCHED = {
    "data/models/argonix_weapons_yolo26n.pt",     # licence-unknown training data: trained locally
    "data/models/argonix_altercation_gru.pt",     # trained locally
    "data/models/yolo26s-480-b16.engine",         # TensorRT engines are GPU-specific (falls back to yolo26s.pt)
    "data/models/yolo26s-960-b2.engine",          # built on the GPU host (bench_gpu.py --export); same fallback
}


def config_data_paths() -> set[str]:
    found = set()
    for p in (S.ROOT / "configs").rglob("*.json"):
        found |= set(re.findall(r'"(data/[^"]+\.[A-Za-z0-9]+)"', p.read_text(encoding="utf-8")))
    return found


def test_every_config_file_is_covered():
    produced = {S.rel(t) for s in S.steps() for t in s.targets}
    missing = config_data_paths() - produced - NOT_FETCHED
    assert not missing, f"configs reference files tools/setup_assets.py does not provide: {sorted(missing)}"


def test_groups_and_dependencies_are_consistent():
    steps = S.steps()
    names = {s.name for s in steps}
    assert {s.group for s in steps} == set(S.GROUPS)
    for s in steps:
        assert set(s.needs) <= names, s.name


def test_check_mode_changes_nothing(monkeypatch, capsys):
    ran = []
    fake = [S.Step("core", "a", [S.ROOT / "data" / "__nope__.bin"], lambda: ran.append(1))]
    monkeypatch.setattr(S, "steps", lambda: fake)
    assert S.main(["--check"]) == 1
    assert not ran and "MISS" in capsys.readouterr().out


def test_unknown_group_is_rejected():
    with pytest.raises(SystemExit):
        S.main(["--only", "nope"])


def test_keep_config_text_restores_formatting_only_rewrites(tmp_path, monkeypatch):
    (tmp_path / "configs").mkdir()
    cfg = tmp_path / "configs" / "x.json"
    cfg.write_text('{"a": {"b": 1}}\n')
    monkeypatch.setattr(S, "ROOT", tmp_path)
    with S.keep_config_text("x.json"):
        cfg.write_text('{\n  "a": {\n    "b": 1\n  }\n}')
    assert cfg.read_text() == '{"a": {"b": 1}}\n'
    with S.keep_config_text("x.json"):
        cfg.write_text('{"a": {"b": 2}}')
    assert cfg.read_text() == '{"a": {"b": 2}}'
