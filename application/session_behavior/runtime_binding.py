"""Runtime worker, turn admission, and OpenCC resolution helpers for SessionBehavior."""

from __future__ import annotations

from collections.abc import Callable, Mapping
import logging
from typing import Any

from application.chat.capabilities.contracts import (
    BaseTurnContextProvider,
    TurnAdmissionDecision,
    TurnAdmissionRequest,
)
from application.session_behavior.casting import CastingMap

logger = logging.getLogger(__name__)


def resolve_ui_worker(runtime_components: Any, app_runtime: Any) -> Any | None:
    """Resolve the active PresentationWorker across runtime components and AppRuntime."""
    ui_worker = getattr(runtime_components, "ui_worker", None)
    if ui_worker is not None:
        return ui_worker
    workflow = getattr(runtime_components, "workflow", None)
    if workflow is not None:
        try:
            from application.runtime.workflow import get_chat_workflow_handles

            handles = get_chat_workflow_handles(workflow)
            if getattr(handles, "ui_worker", None) is not None:
                return handles.ui_worker
        except Exception:
            pass
    if app_runtime is not None:
        return getattr(app_runtime, "ui_worker", None)
    return None


def resolve_character_media_handler(
    runtime_components: Any,
    app_runtime: Any,
) -> Any | None:
    """Locate CharacterMediaHandler across runtime worker handles and AppRuntime."""
    candidates: list[Any] = []
    direct = getattr(runtime_components, "dialog_media_worker", None)
    if direct is not None:
        candidates.append(direct)

    workflow = getattr(runtime_components, "workflow", None)
    if workflow is not None:
        try:
            from application.runtime.workflow import get_chat_workflow_handles

            handles = get_chat_workflow_handles(workflow)
            if getattr(handles, "dialog_media_worker", None) is not None:
                candidates.append(handles.dialog_media_worker)
        except Exception:
            pass

    if app_runtime is not None:
        for attr in ("dialog_media_worker", "tts_worker"):
            worker = getattr(app_runtime, attr, None)
            if worker is not None:
                candidates.append(worker)

    for worker in candidates:
        dispatcher = getattr(worker, "dialog_media_dispatcher", None) or getattr(
            worker, "handler_chain", None
        )
        if dispatcher is None:
            continue
        handlers = getattr(dispatcher, "handlers", None)
        if not handlers:
            handlers = getattr(dispatcher, "_handlers", ())
        for handler in handlers or ():
            if hasattr(handler, "casting_map_provider"):
                return handler
    return None


def resolve_opencc(runtime_components: Any, app_runtime: Any) -> Any | None:
    """Resolve the active OpenCC instance used for presentation alias matching."""
    cc = getattr(runtime_components, "opencc", None)
    if cc is not None:
        return cc
    cc = getattr(app_runtime, "opencc", None) if app_runtime is not None else None
    if cc is not None:
        return cc
    try:
        from application.runtime.context import try_get_app_runtime

        rt = try_get_app_runtime()
        return getattr(rt, "opencc", None) if rt is not None else None
    except Exception:
        return None


class BehaviorTurnContextProvider(BaseTurnContextProvider):
    """Enforce active SessionBehavior free-text policy at LLMWorker admission."""

    capability_id = "session_behavior"

    def __init__(self, capability: Any) -> None:
        self._capability = capability

    def admit_turn(
        self,
        request: TurnAdmissionRequest,
    ) -> TurnAdmissionDecision | None:
        source = str(request.source or "user").strip() or "user"
        if source == "user" and not self._capability.is_free_text_allowed():
            return TurnAdmissionDecision(admitted=False, record_user_message=False)
        return None


def sync_casting_media_policy(
    llm_manager: Any | None,
    casting: CastingMap | None,
    *,
    media_handler: Any | None,
    opencc_provider: Callable[[], Any | None],
    active_casting_provider: Callable[[], CastingMap | None],
) -> None:
    """Propagate active casting rules to LLMManager and CharacterMediaHandler."""
    if llm_manager is not None and hasattr(llm_manager, "set_speaker_media_policy"):
        if casting is not None:
            def _speaker_requires_media(
                speaker_name: str,
                _casting: CastingMap = casting,
            ) -> bool:
                return _casting.speaker_requires_media(speaker_name, cc=opencc_provider())

            llm_manager.set_speaker_media_policy(
                _speaker_requires_media,
                casting.bound_speaker_names(),
            )
        else:
            llm_manager.set_speaker_media_policy(None, ())
    if media_handler is not None and hasattr(media_handler, "casting_map_provider"):
        media_handler.casting_map_provider = (
            active_casting_provider if casting is not None else None
        )


def auto_attach_bound_behavior(capability: Any, context: Any) -> bool:
    """Attach any persisted SessionBehavior bound to the session history directory."""
    if getattr(capability, "behavior", None) is not None:
        return True
    try:
        from application.novel_theater.bootstrap import attach_novel_theater_if_bound

        return bool(attach_novel_theater_if_bound(capability, context))
    except Exception:
        logger.warning("Failed to auto-attach bound session behavior", exc_info=True)
        return False


def notify_behavior_session_ready(capability: Any) -> None:
    """Notify bound SessionBehavior when streaming/session startup completes."""
    if getattr(capability, "behavior", None) is None:
        return
    try:
        from application.novel_theater.bootstrap import kickoff_novel_theater_if_fresh

        kickoff_novel_theater_if_fresh(capability)
    except Exception:
        logger.warning("Failed to run SessionBehavior on_session_ready hook", exc_info=True)


def after_behavior_transition_committed(
    behavior: Any,
    *,
    branch_id: str,
    state_version: int,
    appended_turn_seqs: tuple[int, ...] = (),
) -> None:
    """Trigger post-commit behavior work only after SQLite commit and replace emit."""
    callback = getattr(behavior, "on_transition_committed", None)
    if callable(callback):
        callback(
            branch_id=branch_id,
            state_version=state_version,
            appended_turn_seqs=appended_turn_seqs,
        )


__all__ = [
    "BehaviorTurnContextProvider",
    "after_behavior_transition_committed",
    "auto_attach_bound_behavior",
    "notify_behavior_session_ready",
    "resolve_character_media_handler",
    "resolve_opencc",
    "resolve_ui_worker",
    "sync_casting_media_policy",
]
