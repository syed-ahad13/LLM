from pathlib import Path

import pytest

from collect import polite


@pytest.fixture
def no_contact(monkeypatch, tmp_path: Path) -> Path:
    monkeypatch.delenv("URDU_BOT_CONTACT", raising=False)
    path = tmp_path / "contact.local"
    monkeypatch.setattr(polite, "CONTACT_FILE", path)
    return path


def test_contact_from_environment(no_contact: Path, monkeypatch) -> None:
    monkeypatch.setenv("URDU_BOT_CONTACT", "someone@example.org")
    assert polite.default_user_agent() == "UrduCorpusCourseBot/1.0 (NUST SEECS coursework; contact: someone@example.org)"


def test_contact_from_local_file(no_contact: Path) -> None:
    no_contact.write_text("\n  student@example.edu  \n", encoding="utf-8")
    assert polite.contact_address() == "student@example.edu"


def test_environment_wins_over_file(no_contact: Path, monkeypatch) -> None:
    no_contact.write_text("file@example.edu\n", encoding="utf-8")
    monkeypatch.setenv("URDU_BOT_CONTACT", "env@example.org")
    assert polite.contact_address() == "env@example.org"


def test_no_contact_means_no_client(no_contact: Path) -> None:
    with pytest.raises(polite.MissingContact, match="contact.local"):
        polite.PoliteClient()


def test_contact_file_is_gitignored() -> None:
    gitignore = (Path(__file__).resolve().parent.parent / ".gitignore").read_text(encoding="utf-8")
    assert "/collect/contact.local" in gitignore.splitlines()
