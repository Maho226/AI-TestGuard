"""Tests de sécurité opérationnelle (Partie VIII) — exigence T18.

Champs manquants ou supplémentaires et types incorrects : voir aussi test_prediction.py et test_robustness.py.
"""

import re
import subprocess
from pathlib import Path

import pytest

from app.preprocessing import load_data
from app.training import file_sha256, load_model, run_training_pipeline

ROOT = Path(__file__).resolve().parents[1]
SAMPLE = ROOT / "data" / "sample" / "creditcard_sample.csv"

SECRET_PATTERNS = {
    "jeton GitHub": r"gh[pousr]_[A-Za-z0-9]{36,}|github_pat_[A-Za-z0-9_]{20,}",
    "clé AWS": r"AKIA[0-9A-Z]{16}",
    "clé privée": r"-----BEGIN [A-Z ]*PRIVATE KEY-----",
    "mot de passe en clair": r"(?i)(password|passwd|mot_de_passe)\s*[:=]\s*['\"][^'\"]{4,}['\"]",
    "jeton dans une URL git": r"https://[^/\s:]+:[^@\s]{8,}@github\.com",
}


def tracked_files():
    try:
        out = subprocess.run(["git", "ls-files"], cwd=ROOT, capture_output=True, text=True, check=True).stdout
        files = [ROOT / f for f in out.splitlines()]
    except (OSError, subprocess.CalledProcessError):
        excluded = {".git", ".venv", "venv", "site-packages", "__pycache__"}
        files = [p for p in ROOT.rglob("*") if p.is_file() and not excluded & set(p.parts)]
    return [f for f in files if f.suffix in {".py", ".md", ".toml", ".yml", ".yaml", ".json", ".txt", ".ipynb", ".cfg"}]


# ---------------------------------------------------------------- secrets dans le dépôt
def test_t18_no_secrets_in_repository():
    findings = []
    for path in tracked_files():
        text = path.read_text(errors="ignore")
        for name, pattern in SECRET_PATTERNS.items():
            if re.search(pattern, text):
                findings.append(f"{path.relative_to(ROOT)} : {name}")
    assert not findings, f"Secrets potentiels détectés : {findings}"


def test_t18_secret_detector_works():
    fake = "token = 'ghp_" + "a" * 36 + "'"
    assert re.search(SECRET_PATTERNS["jeton GitHub"], fake)


def test_t18_full_dataset_is_not_versioned():
    assert "data/*.csv" in (ROOT / ".gitignore").read_text()
    assert "creditcard.csv" not in {p.name for p in tracked_files()} | {p.name for p in ROOT.glob("data/sample/*")}


# ---------------------------------------------------------------- validation des fichiers reçus
def test_t18_wrong_file_extension_is_rejected(tmp_path):
    path = tmp_path / "transactions.exe"
    path.write_bytes(SAMPLE.read_bytes())
    with pytest.raises(ValueError, match="Extension non autorisée"):
        load_data(path)


def test_t18_oversized_file_is_rejected():
    with pytest.raises(ValueError, match="trop volumineux"):
        load_data(SAMPLE, max_size_mb=0.001)


def test_t18_binary_content_is_rejected(tmp_path):
    path = tmp_path / "faux.csv"
    path.write_bytes(bytes(range(256)) * 10)
    with pytest.raises(ValueError):
        load_data(path)


# ---------------------------------------------------------------- intégrité du modèle
@pytest.fixture(scope="module")
def saved_model(tmp_path_factory):
    path = tmp_path_factory.mktemp("sec") / "model.joblib"
    run_training_pipeline(SAMPLE, path)
    return path


def test_t18_model_integrity_is_verified(saved_model):
    import json

    expected = json.loads(saved_model.with_suffix(".json").read_text())["sha256"]
    assert expected == file_sha256(saved_model)
    assert load_model(saved_model, expected_sha256=expected)["threshold"] >= 0


def test_t18_tampered_model_is_rejected(saved_model, tmp_path):
    tampered = tmp_path / "model.joblib"
    tampered.write_bytes(saved_model.read_bytes() + b"\x00")
    with pytest.raises(ValueError, match="Intégrité"):
        load_model(tampered, expected_sha256=file_sha256(saved_model))
