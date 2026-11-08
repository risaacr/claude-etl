"""Unit tests for _plan_message_write — the decision of how much of a stored
conversation to keep when a transcript is re-ingested. No database needed.

Why this exists (measured 2026-11-08): the cron re-ingests a live transcript
whenever its BYTES change, but Claude Code appends many records the parser
ignores (queue-operation, last-prompt, ai-title, mode, permission-mode,
attachment, api-request...). Those passes had the same message count, the old
append check refused anything that did not GROW, and a 27k-message session was
DELETEd and reinserted 517 times in a week with byte-identical content.

The plan must keep exactly the rows it has positively verified (uuid AND
content hash, row by row from the head) and nothing else.

Run:  cd ~/projects/claude-etl && python3 -m pytest v2/tests/test_write_plan.py -v
"""

from v2.db import _content_hash, _plan_message_write
from v2.parser import ParsedMessage


def _msg(seq, uuid, content):
    return ParsedMessage(
        uuid=uuid, parent_uuid=None, role="user", content=content, thinking=None,
        tool_calls=None, tool_results=None, is_sidechain=False, timestamp=None,
        sequence_num=seq,
    )


def _incoming(n, mutate=None, uuid_of=None):
    """n messages; `mutate` maps seq -> new content, `uuid_of` maps seq -> new uuid."""
    mutate, uuid_of = mutate or {}, uuid_of or {}
    return [_msg(i, uuid_of.get(i, f"u{i}"), mutate.get(i, f"c{i}")) for i in range(n)]


def _stored(n):
    """What a clean ingest of _incoming(n) leaves in the messages table."""
    return [(i, f"u{i}", _content_hash(f"c{i}")) for i in range(n)]


def test_nothing_stored_is_an_insert():
    assert _plan_message_write([], _incoming(3)) == ("insert", 0)


def test_identical_messages_write_nothing():
    # THE live case: file bytes changed, messages did not.
    assert _plan_message_write(_stored(5), _incoming(5)) == ("unchanged", 5)


def test_growth_with_intact_prefix_is_an_append():
    assert _plan_message_write(_stored(5), _incoming(8)) == ("append", 5)


def test_mutated_last_message_same_count_rewrites_only_it():
    assert _plan_message_write(_stored(5), _incoming(5, mutate={4: "partial->final"})) == ("suffix", 4)


def test_mutated_last_message_plus_growth_rewrites_from_it():
    assert _plan_message_write(_stored(5), _incoming(9, mutate={4: "x"})) == ("suffix", 4)


def test_mid_conversation_mutation_rewrites_from_that_row():
    assert _plan_message_write(_stored(10), _incoming(10, mutate={6: "x"})) == ("suffix", 6)


def test_uuid_change_with_same_content_is_a_divergence():
    assert _plan_message_write(_stored(5), _incoming(5, uuid_of={3: "other"})) == ("suffix", 3)


def test_head_mismatch_is_a_full_replace():
    # ⛔ Compaction rewrites from the head: the file can be LONGER and share
    # nothing at the same sequence numbers.
    assert _plan_message_write(_stored(5), _incoming(9, uuid_of={0: "compacted"})) == ("replace", 0)
    assert _plan_message_write(_stored(5), _incoming(9, mutate={0: "summary"})) == ("replace", 0)


def test_head_content_mismatch_same_count_is_a_full_replace():
    assert _plan_message_write(_stored(5), _incoming(5, mutate={0: "x"})) == ("replace", 0)


def test_shrunk_file_with_intact_prefix_drops_only_the_tail():
    assert _plan_message_write(_stored(8), _incoming(5)) == ("suffix", 5)


def test_missing_uuid_never_reads_as_a_match():
    stored = _stored(5)
    stored[2] = (2, None, _content_hash("c2"))
    incoming = _incoming(5, uuid_of={2: None})
    assert _plan_message_write(stored, incoming) == ("suffix", 2)


def test_missing_uuid_at_head_is_a_full_replace():
    stored = _stored(3)
    stored[0] = (0, None, _content_hash("c0"))
    assert _plan_message_write(stored, _incoming(3, uuid_of={0: None})) == ("replace", 0)


def test_null_content_on_both_sides_matches_on_uuid():
    # Tool-only messages have content None and content_hash None. uuid is the
    # identity for them, exactly as the old boundary check treated them.
    stored = [(0, "u0", _content_hash("c0")), (1, "u1", None)]
    incoming = [_msg(0, "u0", "c0"), _msg(1, "u1", None)]
    assert _plan_message_write(stored, incoming) == ("unchanged", 2)


def test_stored_order_does_not_matter():
    assert _plan_message_write(list(reversed(_stored(5))), _incoming(5)) == ("unchanged", 5)


def test_duplicate_stored_sequence_is_a_full_replace():
    # A duplicate below the cut would survive a 'DELETE WHERE seq >= keep'.
    stored = _stored(5) + [(2, "u2", _content_hash("c2"))]
    assert _plan_message_write(stored, _incoming(5)) == ("replace", 0)


def test_null_stored_sequence_is_a_full_replace():
    stored = _stored(5) + [(None, "ux", None)]
    assert _plan_message_write(stored, _incoming(5)) == ("replace", 0)


def test_gap_in_stored_sequence_cuts_at_the_gap():
    stored = [r for r in _stored(6) if r[0] != 3]
    assert _plan_message_write(stored, _incoming(6)) == ("suffix", 3)


def test_stored_range_not_starting_at_zero_is_a_full_replace():
    stored = [(i + 1, f"u{i}", _content_hash(f"c{i}")) for i in range(4)]
    assert _plan_message_write(stored, _incoming(4)) == ("replace", 0)


def test_empty_incoming_with_rows_stored_is_a_full_replace():
    assert _plan_message_write(_stored(3), []) == ("replace", 0)
