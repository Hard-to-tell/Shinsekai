"""Casting normalization, launch payload assembly, and live casting updates."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
import time
from typing import Any
import uuid

from ai.llm.template.integrations.localization import resolve_template_locale
from ai.llm.template.session.novel_theater import (
    render_novel_director_prompt,
    render_novel_pure_dialog_system,
)
from application.chat.conversation_library import (
    _record,
    update_conversation_launch_fields,
)
from application.chat.history.paths import (
    _state_project_root,
    resolve_history_path_for_project,
)
from application.novel_theater.import_document import build_chapter_toc_entries
from application.session_behavior.casting import CastBinding, CastingMap
from core.chat_history.behavior_state_store import BehaviorStateStore
from core.chat_history.sqlite_session_store import borrow_session_store_for_dir
from core.chat_history.storage import (
    chat_history_active_path,
    chat_history_session_dir,
    legacy_active_history_branch,
)
from core.document import DocumentRepository, SessionDocumentBinding
from core.storage.layout import StorageLayout
from sdk.exception.types import AppError

NOVEL_THEATER_BEHAVIOR_ID = "novel_theater"
_VALID_ROLES = frozenset({"protagonist", "supporting", "antagonist"})


def resolve_chapter_index_for_chunk(
    chunk_index: int,
    chapters: Sequence[Mapping[str, Any]],
) -> tuple[int, str]:
    """Return 0-based (chapter_index, chapter_title) containing ``chunk_index``."""
    if not chapters:
        return 0, ""
    chosen_idx = 0
    chosen_title = str(chapters[0].get("title") or "")
    for idx, entry in enumerate(chapters):
        start = int(entry.get("startChunkIndex", 0) or 0)
        count = int(entry.get("chunkCount", 0) or 0)
        if count <= 0:
            continue
        if chunk_index >= start:
            chosen_idx = idx
            chosen_title = str(entry.get("title") or "")
        else:
            break
    return chosen_idx, chosen_title


def normalize_cast_bindings(
    raw_bindings: Sequence[Mapping[str, Any]],
    config_manager: Any,
) -> tuple[CastingMap, list[str], list[dict[str, Any]]]:
    """Validate and convert UI cast rows into ``CastingMap`` and metadata entries."""
    entries: list[dict[str, Any]] = []
    aliases_by_target: dict[str, list[str]] = {}
    meta_by_target: dict[str, tuple[str, bool]] = {}
    target_order: list[str] = []

    getter = getattr(config_manager, "get_character_by_name", None)
    for raw in raw_bindings:
        if not isinstance(raw, Mapping):
            continue
        source_name = str(
            raw.get("sourceName") or raw.get("name") or raw.get("source_name") or ""
        ).strip()
        if not source_name:
            continue
        raw_aliases = raw.get("aliases") or ()
        aliases = list(
            dict.fromkeys(
                clean
                for item in (raw_aliases if isinstance(raw_aliases, (list, tuple)) else ())
                if (clean := str(item or "").strip()) and clean != source_name
            )
        )
        role = str(raw.get("role") or "supporting").strip().lower()
        if role not in _VALID_ROLES:
            role = "supporting"
        silhouette = bool(raw.get("silhouetteUntilRevealed", False))
        raw_target = str(
            raw.get("targetCharacter") or raw.get("target_character") or ""
        ).strip()
        canonical_target = ""
        if raw_target:
            char_obj = getter(raw_target) if callable(getter) else None
            if char_obj is None:
                raise AppError(
                    "novel.unknown_target_character",
                    params={"character": raw_target},
                    fallback_message=f"Unknown target character: {raw_target}",
                    http_status=400,
                )
            canonical_target = str(getattr(char_obj, "name", "") or raw_target).strip()
            if canonical_target not in aliases_by_target:
                target_order.append(canonical_target)
                aliases_by_target[canonical_target] = []
                meta_by_target[canonical_target] = (role, silhouette)
            aliases_by_target[canonical_target].extend([source_name, *aliases])

        entries.append(
            {
                "sourceName": source_name,
                "aliases": aliases,
                "targetCharacter": canonical_target,
                "role": role,
                "silhouetteUntilRevealed": silhouette,
            }
        )

    bindings: list[CastBinding] = []
    for target in target_order:
        role, silhouette = meta_by_target[target]
        merged_aliases = tuple(
            dict.fromkeys(
                item for item in aliases_by_target[target] if item and item != target
            )
        )
        bindings.append(
            CastBinding(
                character_name=target,
                aliases=merged_aliases,
                role=role,
                silhouette_until_revealed=silhouette,
                revealed=not silhouette,
            )
        )
    return CastingMap(bindings), target_order, entries


def build_novel_system_template(
    state: Any,
    payload: Mapping[str, Any],
) -> str:
    """Build the system dialog-contract template supporting both bound and pure-dialogue modes."""
    config_manager = getattr(state, "config_manager", None)
    characters = list(payload.get("characters") or [])
    if not characters:
        return render_novel_pure_dialog_system(
            locale=resolve_template_locale(config_manager)
        )
    generator = getattr(state, "template_generator", None)
    media_mode = (
        "semantic"
        if str(payload.get("mediaSelectionMode") or "").strip().lower() == "semantic"
        else "indexed"
    )
    content, _ = generator.generate_chat_template(
        characters,
        str(payload.get("backgroundName") or ""),
        bool(payload.get("useEffect", True)),
        False,
        bool(payload.get("useTranslation", True)),
        False,
        False,
        True,
        False,
        media_selection_mode=media_mode,
    )
    return str(content or "")


def prepare_novel_launch(
    state: Any,
    body: Mapping[str, Any],
) -> dict[str, Any]:
    """Bind a shared document to a session directory, initialize snapshot, and build launch payload."""
    doc_id = str(body.get("docId") or "").strip().lower()
    layout = StorageLayout(_state_project_root(state))
    repo = DocumentRepository(layout)
    try:
        manifest = repo.get_manifest(doc_id)
    except ValueError as exc:
        raise AppError(
            "novel.document_not_found",
            params={"docId": doc_id},
            fallback_message="Document not found.",
            http_status=404,
        ) from exc
    if manifest is None:
        raise AppError(
            "novel.document_not_found",
            params={"docId": doc_id},
            fallback_message="Document not found.",
            http_status=404,
        )

    chapters = repo.read_chapters(doc_id)
    chunks = repo.read_chunks(doc_id)
    if not chunks:
        raise AppError(
            "novel.document_empty",
            params={"docId": doc_id},
            fallback_message="Document has no playable chunks.",
            http_status=400,
        )
    toc = build_chapter_toc_entries(chapters, chunks)

    raw_start_chapter = int(body.get("startChapterIndex", 0) or 0)
    if toc and not (0 <= raw_start_chapter < len(toc)):
        raise AppError(
            "novel.invalid_chapter",
            params={"chapterIndex": raw_start_chapter},
            fallback_message=f"Invalid starting chapter index: {raw_start_chapter}",
            http_status=400,
        )
    start_chunk_index = (
        int(toc[raw_start_chapter].get("startChunkIndex", 0) or 0)
        if toc
        else 0
    )
    start_chunk_index = max(0, min(start_chunk_index, len(chunks) - 1))
    resolved_chap_idx, _ = resolve_chapter_index_for_chunk(start_chunk_index, toc)

    config_manager = getattr(state, "config_manager", None)
    raw_casting = body.get("casting") or ()
    casting_map, bound_characters, cast_entries = normalize_cast_bindings(
        raw_casting if isinstance(raw_casting, Sequence) else (),
        config_manager,
    )

    raw_history = str(body.get("historyPath") or "").strip()
    history_root = Path(getattr(state, "history_dir", None) or layout.chat_history_dir)
    target_raw = (
        raw_history
        if raw_history
        else (history_root / f"novel-{uuid.uuid4().hex[:12]}").as_posix()
    )
    history_path = resolve_history_path_for_project(state, target_raw)
    session_dir = chat_history_session_dir(history_path)
    session_dir.mkdir(parents=True, exist_ok=True)
    active_file = chat_history_active_path(history_path)
    if not active_file.exists():
        active_file.write_text("[]", encoding="utf-8")

    DocumentRepository.write_session_binding(
        session_dir,
        SessionDocumentBinding(
            doc_id=doc_id,
            behavior_id=NOVEL_THEATER_BEHAVIOR_ID,
            bound_at_ms=int(time.time() * 1000),
            source_filename=manifest.source_filename,
        ),
    )

    initial_state = {
        "docId": doc_id,
        "title": manifest.title,
        "chunkIndex": start_chunk_index,
        "chapterIndex": resolved_chap_idx,
        "totalChunks": len(chunks),
        "totalChapters": len(toc),
        "phase": "ready",
        "casting": casting_map.to_dict(),
        "castEntries": cast_entries,
        "chapters": toc,
        "lastError": "",
    }
    with borrow_session_store_for_dir(session_dir, create_if_dir_exists=True) as store:
        assert store is not None
        BehaviorStateStore(store, session_dir).initialize_state(
            branch_id="main",
            behavior_id=NOVEL_THEATER_BEHAVIOR_ID,
            state=initial_state,
        )

    locale = resolve_template_locale(config_manager)
    director_scenario = render_novel_director_prompt(cast_entries, locale=locale)
    media_mode = (
        "semantic"
        if str(body.get("mediaSelectionMode") or "").strip().lower() == "semantic"
        else "indexed"
    )
    background_name = str(body.get("backgroundName") or "").strip()
    custom_title = str(body.get("conversationTitle") or "").strip()[:120]
    conversation_title = custom_title or manifest.title
    payload: dict[str, Any] = {
        "behaviorId": NOVEL_THEATER_BEHAVIOR_ID,
        "novelDocId": doc_id,
        "casting": casting_map.to_dict(),
        "castEntries": cast_entries,
        "characters": bound_characters,
        "backgroundName": background_name,
        "mediaSelectionMode": media_mode,
        "useChoice": False,
        "useNarration": True,
        "useStat": False,
        "useEffect": bool(body.get("useEffect", True)),
        "useTranslation": bool(body.get("useTranslation", True)),
        "memoryMode": "isolated",
        "showInitialSprite": bool(bound_characters),
        "historyPath": history_path.as_posix(),
        "conversationTitle": conversation_title,
        "templateId": "",
        "templateName": manifest.title,
        "scenario": director_scenario,
        "resetHistory": False,
        "enableMobileAccess": bool(body.get("enableMobileAccess", False)),
    }
    return payload


def save_novel_casting(
    state: Any,
    conversation_id: str,
    raw_bindings: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Persist updated cast bindings to conversation metadata, SQLite snapshot, or live runtime."""
    clean_id = str(conversation_id or "").strip()
    try:
        record = _record(state, clean_id)
    except KeyError as exc:
        raise AppError(
            "novel.conversation_not_found",
            params={"conversationId": clean_id},
            fallback_message="Conversation not found.",
            http_status=404,
        ) from exc

    casting_map, bound_characters, cast_entries = normalize_cast_bindings(
        raw_bindings,
        getattr(state, "config_manager", None),
    )
    locale = resolve_template_locale(getattr(state, "config_manager", None))
    summary = update_conversation_launch_fields(
        state,
        clean_id,
        {
            "casting": casting_map.to_dict(),
            "castEntries": cast_entries,
            "characters": bound_characters,
            "scenario": render_novel_director_prompt(cast_entries, locale=locale),
        },
    )

    history_path = resolve_history_path_for_project(state, record["historyPath"])
    session_dir = chat_history_session_dir(history_path)
    live_updated = False
    active_history = str(
        (getattr(state, "chat_session", None) or {}).get("historyPath") or ""
    ).strip()
    if active_history:
        try:
            active_session_dir = chat_history_session_dir(
                resolve_history_path_for_project(state, active_history)
            ).resolve(strict=False)
        except (OSError, ValueError):
            active_session_dir = None
        if active_session_dir == session_dir.resolve(strict=False):
            from application.chat.composition import _chat_snapshot, _handle_chat_command

            snap = _chat_snapshot(state)
            behavior_snap = snap.get("behavior") if isinstance(snap, dict) else None
            if (
                isinstance(behavior_snap, dict)
                and behavior_snap.get("behaviorId") == NOVEL_THEATER_BEHAVIOR_ID
            ):
                try:
                    _handle_chat_command(
                        state,
                        {
                            "type": "behavior-command",
                            "payload": {
                                "commandId": uuid.uuid4().hex,
                                "action": "update_casting",
                                "expectedStateVersion": int(
                                    behavior_snap.get("stateVersion") or 0
                                ),
                                "payload": {
                                    "casting": casting_map.to_dict(),
                                    "castEntries": cast_entries,
                                },
                            },
                        },
                    )
                    live_updated = True
                except RuntimeError:
                    live_updated = False

    if not live_updated:
        branch_id = legacy_active_history_branch(history_path)
        with borrow_session_store_for_dir(session_dir, create_if_dir_exists=False) as store:
            if store is not None:
                behavior_store = BehaviorStateStore(store, session_dir)
                latest = behavior_store.get_latest_state(branch_id)
                if latest is not None and latest.behavior_id == NOVEL_THEATER_BEHAVIOR_ID:
                    next_state = {
                        **latest.state,
                        "casting": casting_map.to_dict(),
                        "castEntries": cast_entries,
                    }
                    behavior_store.commit_behavior_turn(
                        branch_id=branch_id,
                        behavior_id=NOVEL_THEATER_BEHAVIOR_ID,
                        expected_state_version=latest.state_version,
                        next_state=next_state,
                        messages=(),
                    )
    return summary


__all__ = [
    "NOVEL_THEATER_BEHAVIOR_ID",
    "build_novel_system_template",
    "normalize_cast_bindings",
    "prepare_novel_launch",
    "resolve_chapter_index_for_chunk",
    "save_novel_casting",
]
