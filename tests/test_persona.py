from pathlib import Path
import pytest

from archygazeisnotmortal.persona import build_system_prompt, load_persona


def test_build_system_prompt_includes_persona_and_rules() -> None:
    prompt = build_system_prompt("Name: Kai\nStyle: lowercase")
    assert "PERSONA PROFILE" in prompt
    assert "Name: Kai" in prompt
    assert "Never say \"As an AI\"" in prompt


def test_load_persona_missing_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_persona(tmp_path / "missing.md")


def test_load_persona_falls_back_to_inline_content(tmp_path: Path) -> None:
    prompt = load_persona(tmp_path / "missing.md", persona_content="Name: archygaze")
    assert "Name: archygaze" in prompt

    file_path = tmp_path / "persona.md"
    file_path.write_text("Name: from_file", encoding="utf-8")
    file_prompt = load_persona(file_path, persona_content="Name: from_env")
    assert "Name: from_file" in file_prompt
    assert "Name: from_env" not in file_prompt

