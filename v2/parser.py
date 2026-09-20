"""Parse Claude Code JSONL conversation files into structured data."""

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class ParsedMessage:
    uuid: Optional[str]
    parent_uuid: Optional[str]
    role: str  # user, assistant, system
    content: Optional[str]
    thinking: Optional[str]
    tool_calls: Optional[list]  # list of tool_use blocks
    tool_results: Optional[list]  # list of tool_result blocks
    is_sidechain: bool
    timestamp: Optional[datetime]
    sequence_num: int


@dataclass
class ParsedConversation:
    session_id: Optional[str]
    project: Optional[str]
    model: Optional[str]
    started_at: Optional[datetime]
    last_message_at: Optional[datetime]
    message_count: int
    messages: list[ParsedMessage] = field(default_factory=list)


def sanitize_null_bytes(text: str) -> str:
    """Remove null bytes that PostgreSQL can't store."""
    return text.replace("\x00", "") if text else text


def sanitize_deep(obj):
    """Strip null bytes from every string inside a nested structure.

    The scalar sanitizer above only covers the columns we flatten to text. Tool
    inputs go to JSONB whole and unflattened, so a null byte nested anywhere in
    one reaches Postgres verbatim and the whole INSERT dies with "unsupported
    Unicode escape sequence" — which fails the CONVERSATION, not the message, and
    retries identically every minute forever. Found 2026-09-20: one session had
    been failing every minute since 2026-09-15 because a tool_use input carried
    the source of a sanitizer whose own regex literal contained \\x00.
    """
    if isinstance(obj, str):
        return obj.replace("\x00", "")
    if isinstance(obj, dict):
        return {k: sanitize_deep(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [sanitize_deep(v) for v in obj]
    return obj


def extract_text_content(content) -> str:
    """Extract text from a message content field.

    Content can be a string (user messages) or a list of blocks (assistant messages).
    """
    if isinstance(content, str):
        return sanitize_null_bytes(content)

    if isinstance(content, list):
        texts = []
        for block in content:
            if isinstance(block, dict):
                if block.get("type") == "text":
                    texts.append(block.get("text", ""))
            elif isinstance(block, str):
                texts.append(block)
        return sanitize_null_bytes("\n".join(texts)) if texts else None

    return None


def extract_thinking(content) -> Optional[str]:
    """Extract thinking blocks from assistant content."""
    if not isinstance(content, list):
        return None

    thinking_parts = []
    for block in content:
        if isinstance(block, dict) and block.get("type") == "thinking":
            thinking_parts.append(block.get("thinking", ""))

    return sanitize_null_bytes("\n".join(thinking_parts)) if thinking_parts else None


def extract_tool_calls(content) -> Optional[list]:
    """Extract tool_use blocks from assistant content."""
    if not isinstance(content, list):
        return None

    tools = []
    for block in content:
        if isinstance(block, dict) and block.get("type") in ("tool_use", "server_tool_use"):
            tools.append({
                "type": block.get("type"),
                "id": block.get("id"),
                "name": block.get("name"),
                # sanitize_deep, not sanitize_null_bytes: input is arbitrary nested
                # JSON and goes to JSONB whole. See sanitize_deep's docstring.
                "input": sanitize_deep(block.get("input")),
            })

    return tools if tools else None


def extract_tool_results(content) -> Optional[list]:
    """Extract tool_result blocks from content."""
    if not isinstance(content, list):
        return None

    results = []
    for block in content:
        if isinstance(block, dict) and block.get("type") in ("tool_result", "advisor_tool_result"):
            result_content = block.get("content", "")
            if isinstance(result_content, list):
                result_content = "\n".join(
                    b.get("text", "") for b in result_content
                    if isinstance(b, dict) and b.get("type") == "text"
                )
            results.append({
                "type": block.get("type"),
                "tool_use_id": block.get("tool_use_id"),
                "content": sanitize_null_bytes(str(result_content)[:10000]),  # cap size
            })

    return results if results else None


def parse_timestamp(ts_str: Optional[str]) -> Optional[datetime]:
    """Parse ISO timestamp string to datetime."""
    if not ts_str:
        return None
    try:
        return datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
    except (ValueError, AttributeError):
        return None


def parse_conversation(path: Path) -> ParsedConversation:
    """Parse a Claude Code JSONL file into a ParsedConversation.

    Reads the entire file, extracts user/assistant/system messages,
    and returns structured data ready for DB insertion.
    """
    path = Path(path)
    session_id = None
    project = None
    model = None
    messages = []
    first_timestamp = None
    last_timestamp = None
    sequence = 0

    if not path.exists() or path.stat().st_size == 0:
        return ParsedConversation(
            session_id=None, project=None, model=None,
            started_at=None, last_message_at=None,
            message_count=0, messages=[]
        )

    # Derive project from the parent directory name
    # ~/.claude/projects/-home-rrobinson-corpus-isaac-workspace-corpus/abc123.jsonl
    # project = "-home-rrobinson-corpus-isaac-workspace-corpus"
    project = path.parent.name
    if project == "subagents":
        project = path.parent.parent.name  # go up one more for subagent files

    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue

            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                logger.warning(f"Malformed JSON at {path}:{line_num}, skipping")
                continue

            rec_type = rec.get("type")

            # Extract session_id from queue-operation records
            if rec_type == "queue-operation" and rec.get("operation") == "enqueue":
                if not session_id:
                    session_id = rec.get("sessionId")
                ts = parse_timestamp(rec.get("timestamp"))
                if ts and not first_timestamp:
                    first_timestamp = ts

            # Extract session_id from regular messages too
            if not session_id and "sessionId" in rec:
                session_id = rec["sessionId"]

            # Process user and assistant messages
            if rec_type in ("user", "assistant", "system"):
                msg_data = rec.get("message", {})
                role = msg_data.get("role", rec_type)
                content_raw = msg_data.get("content", rec.get("content"))

                # Extract model from assistant messages
                if role == "assistant" and not model:
                    model = msg_data.get("model")

                text_content = extract_text_content(content_raw)
                thinking = extract_thinking(content_raw) if role == "assistant" else None
                tool_calls = extract_tool_calls(content_raw) if role == "assistant" else None
                tool_results = extract_tool_results(content_raw) if role == "user" else None

                ts = parse_timestamp(rec.get("timestamp"))
                if ts:
                    if not first_timestamp:
                        first_timestamp = ts
                    last_timestamp = ts

                msg = ParsedMessage(
                    uuid=rec.get("uuid"),
                    parent_uuid=rec.get("parentUuid"),
                    role=role,
                    content=text_content,
                    thinking=thinking,
                    tool_calls=tool_calls,
                    tool_results=tool_results,
                    is_sidechain=rec.get("isSidechain", False),
                    timestamp=ts,
                    sequence_num=sequence,
                )
                messages.append(msg)
                sequence += 1

    # Fall back to filename as session_id if not found in records
    if not session_id:
        session_id = path.stem  # filename without extension

    return ParsedConversation(
        session_id=session_id,
        project=project,
        model=model,
        started_at=first_timestamp,
        last_message_at=last_timestamp,
        message_count=len(messages),
        messages=messages,
    )
