"""Unit tests for Novel Theater document, casting, launch, and library use cases."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from ai.llm.template_generator import TemplateGenerator
from application.chat.conversation_library import _details, list_conversations
from application.novel_theater import (
    build_novel_system_template,
    delete_novel_document,
    extract_novel_cast,
    get_novel_document_detail,
    import_novel_document,
    list_novel_documents,
    prepare_novel_launch,
    save_novel_casting,
)
from application.session_behavior.capability import SessionBehaviorCapability
from application.session_behavior.runtime_binding import (
    after_behavior_transition_committed,
)
from config.config_manager import ConfigManager
from core.chat_history.behavior_state_store import BehaviorStateStore
from core.chat_history.sqlite_session_store import close_all_session_stores
from core.chat_history.storage import (
    chat_history_active_path,
    chat_history_session_dir,
    legacy_active_history_branch,
    open_session_store,
)
from sdk.exception.types import AppError
from test.mocks import MockLLMAdapter


def _make_state(tmp_path: Path, config_manager: ConfigManager | None = None) -> SimpleNamespace:
    cfg_mgr = config_manager or ConfigManager()
    tpl_dir = tmp_path / "templates"
    tpl_dir.mkdir(parents=True, exist_ok=True)
    hist_dir = tmp_path / "chat_history"
    hist_dir.mkdir(parents=True, exist_ok=True)
    return SimpleNamespace(
        project_root_dir=str(tmp_path),
        template_dir_path=str(tpl_dir),
        history_dir=str(hist_dir),
        config_manager=cfg_mgr,
        template_generator=TemplateGenerator(cfg_mgr),
        chat_session={},
    )


def _sample_novel_txt(tmp_path: Path) -> Path:
    path = tmp_path / "mist_harbor.txt"
    path.write_text(
        "Chapter 1 The Bell\n\nLin entered the foggy dockside station at dawn.\n"
        "\"Is anyone awake?\" she asked.\n\n"
        "Chapter 2 The Lantern\n\nMio held a brass lantern near the pier.\n"
        "\"The ferry leaves before noon,\" Mio whispered.\n",
        encoding="utf-8",
    )
    return path


def test_import_list_detail_and_delete_protection(tmp_path: Path) -> None:
    state = _make_state(tmp_path)
    novel_path = _sample_novel_txt(tmp_path)

    detail = import_novel_document(state, novel_path, title="Mist Harbor")
    doc_id = detail["manifest"]["docId"]
    assert detail["manifest"]["title"] == "Mist Harbor"
    assert len(detail["chapters"]) >= 2
    assert len(list_novel_documents(state)) == 1
    assert get_novel_document_detail(state, doc_id)["manifest"]["docId"] == doc_id

    payload = prepare_novel_launch(
        state,
        {"docId": doc_id, "casting": [], "startChapterIndex": 1},
    )
    assert payload["behaviorId"] == "novel_theater"
    assert payload["characters"] == []
    assert payload["showInitialSprite"] is False
    assert payload["memoryMode"] == "isolated"

    pure_system = build_novel_system_template(state, payload)
    assert pure_system.strip()

    with pytest.raises(AppError) as in_use_exc:
        delete_novel_document(state, doc_id)
    assert in_use_exc.value.error_code == "novel.document_in_use"

    session_dir = chat_history_session_dir(payload["historyPath"])
    (session_dir / "document-binding.json").unlink()
    close_all_session_stores()

    removed = delete_novel_document(state, doc_id)
    assert removed == {"ok": True, "docId": doc_id}
    assert list_novel_documents(state) == []


def test_extract_novel_cast_parses_json_and_matches_local_characters(
    tmp_path: Path,
) -> None:
    cfg = ConfigManager()
    cfg.config.characters = [
        SimpleNamespace(name="Lin", character_brief="Traveler"),
        SimpleNamespace(name="Mio", character_brief="Keeper"),
    ]
    cfg.get_character_by_name = lambda name: next(
        (c for c in cfg.config.characters if c.name == name), None
    )
    state = _make_state(tmp_path, cfg)
    detail = import_novel_document(state, _sample_novel_txt(tmp_path))
    doc_id = detail["manifest"]["docId"]

    adapter = MockLLMAdapter(
        responses=[
            "```json\n"
            '[{"name": "Lin", "aliases": ["Miss Lin"], "role": "protagonist", '
            '"description": "Lead investigator", "isFirstPerson": false}, '
            '{"name": "Harbor Guard", "aliases": [], "role": "supporting", '
            '"description": "Minor NPC", "isFirstPerson": false}]\n'
            "```"
        ]
    )
    result = extract_novel_cast(state, doc_id, adapter=adapter)
    assert result["docId"] == doc_id
    assert len(result["characters"]) == 2
    assert result["characters"][0]["suggestedCharacter"] == "Lin"
    assert result["characters"][1]["suggestedCharacter"] == ""
    assert [c["name"] for c in result["availableCharacters"]] == ["Lin", "Mio"]


def test_prepare_launch_bound_cast_library_and_save_casting(
    tmp_path: Path, monkeypatch
) -> None:
    cfg = ConfigManager()
    cfg.config.characters = [
        SimpleNamespace(
            name="Lin",
            character_brief="Traveler",
            personality="Calm",
            speaking_style="Direct",
            relationship="",
            sprites=[{"path": "lin.png"}],
            sprite_groups={},
        )
    ]
    cfg.get_character_by_name = lambda name: (
        cfg.config.characters[0] if name == "Lin" else None
    )
    state = _make_state(tmp_path, cfg)
    detail = import_novel_document(state, _sample_novel_txt(tmp_path))
    doc_id = detail["manifest"]["docId"]

    payload = prepare_novel_launch(
        state,
        {
            "docId": doc_id,
            "casting": [
                {
                    "sourceName": "Miss Lin",
                    "aliases": ["Lina"],
                    "targetCharacter": "Lin",
                    "role": "protagonist",
                }
            ],
        },
    )
    assert payload["characters"] == ["Lin"]
    assert payload["showInitialSprite"] is True
    assert "Miss Lin" in payload["scenario"]

    # Persist conversation record using file-backed conversation folder
    from application.chat.conversation_library import remember_conversation

    history_path = Path(payload["historyPath"])
    chat_history_active_path(history_path).write_text("[]", encoding="utf-8")
    remember_conversation(state, history_path, payload)

    conversations = list_conversations(state)
    assert len(conversations) == 1
    assert conversations[0]["kind"] == "novel"
    assert conversations[0]["novelDocId"] == doc_id
    assert conversations[0]["behaviorId"] == "novel_theater"
    assert conversations[0]["memoryMode"] == "isolated"
    conv_id = conversations[0]["id"]

    # Idle save_novel_casting updates metadata and SQLite snapshot
    updated_summary = save_novel_casting(
        state,
        conv_id,
        [{"sourceName": "Harbor Girl", "aliases": ["Hina"], "targetCharacter": "Lin"}],
    )
    assert updated_summary["castEntries"][0]["sourceName"] == "Harbor Girl"
    session_dir = chat_history_session_dir(history_path)
    close_all_session_stores()

    # Live session save_novel_casting delegates strictly via behavior-command without pre-mutating SQLite
    store = open_session_store(history_path)
    behavior_store = BehaviorStateStore(store, session_dir)
    capability = SessionBehaviorCapability(branch_id_provider=lambda: "main")
    from application.novel_theater.behavior import NovelTheaterBehavior
    from core.document import DocumentRepository
    from core.storage.layout import StorageLayout

    repo = DocumentRepository(StorageLayout(tmp_path))
    manifest = repo.get_manifest(doc_id)
    assert manifest is not None
    chapters = repo.read_chapters(doc_id)
    chunks = repo.read_chunks(doc_id)
    from application.novel_theater import build_chapter_toc_entries

    toc = build_chapter_toc_entries(chapters, chunks)

    class _DummyRunner:
        def on_transition_committed(self, **_kwargs):
            return None

        def cancel_active(self):
            return None

    behavior = NovelTheaterBehavior(
        manifest=manifest,
        chapters=chapters,
        chunks=chunks,
        toc_entries=toc,
        turn_runner=_DummyRunner(),  # type: ignore[arg-type]
    )
    snap_before = behavior_store.get_latest_state("main")
    assert snap_before is not None
    capability.attach_behavior(
        behavior,
        state_store=behavior_store,
        initial_state=snap_before.state,
        branch_id="main",
    )
    state.chat_session = {"historyPath": history_path.as_posix()}

    monkeypatch.setattr(
        "application.chat.composition._chat_snapshot",
        lambda _state: {"behavior": capability.snapshot_payload()},
    )
    monkeypatch.setattr(
        "application.chat.composition._handle_chat_command",
        lambda _state, cmd: capability.handle_behavior_command(cmd["payload"]),
    )

    save_novel_casting(
        state,
        conv_id,
        [{"sourceName": "LiveRole", "aliases": [], "targetCharacter": "Lin"}],
    )
    snap_after = behavior_store.get_latest_state(legacy_active_history_branch(history_path))
    assert snap_after is not None
    assert snap_after.state_version == snap_before.state_version + 1
    assert snap_after.state["castEntries"][0]["sourceName"] == "LiveRole"

    # Also verify _details directly when only document-binding.json exists
    detail_only = _details({"historyPath": history_path.as_posix()})
    assert detail_only["kind"] == "novel"
    assert detail_only["novelDocId"] == doc_id
    close_all_session_stores()
