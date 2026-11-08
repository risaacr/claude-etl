"""Tests for database operations. Requires a live conversationsdb_v2 connection."""

import os
import pytest
from pathlib import Path

# Skip all tests if DB is not available
pytestmark = pytest.mark.skipif(
    not os.environ.get("CONVERSATIONS_DB_PASSWORD"),
    reason="CONVERSATIONS_DB_PASSWORD not set — skipping DB tests"
)

from v2.parser import parse_conversation
from v2.db import get_connection, ensure_schema, upsert_conversation

FIXTURE_DIR = Path(__file__).parent
SMALL_FIXTURE = FIXTURE_DIR / "fixture-small.jsonl"


@pytest.fixture
def db_conn():
    """Get a DB connection and ensure schema exists."""
    conn = get_connection()
    ensure_schema(conn)
    yield conn
    conn.close()


class TestSchema:
    def test_tables_exist(self, db_conn):
        with db_conn.cursor() as cur:
            cur.execute(
                "SELECT table_name FROM information_schema.tables "
                "WHERE table_schema = 'public' AND table_name IN ('conversations', 'messages')"
            )
            tables = {row[0] for row in cur.fetchall()}
        assert "conversations" in tables
        assert "messages" in tables

    def test_indexes_exist(self, db_conn):
        with db_conn.cursor() as cur:
            cur.execute(
                "SELECT indexname FROM pg_indexes WHERE tablename IN ('conversations', 'messages')"
            )
            indexes = {row[0] for row in cur.fetchall()}
        assert "idx_messages_content_trgm" in indexes
        assert "idx_conversations_unique" in indexes


class TestUpsert:
    def test_insert_new_conversation(self, db_conn):
        parsed = parse_conversation(SMALL_FIXTURE)
        conv_id = upsert_conversation(
            db_conn, parsed,
            user_name="test_user", machine="test_machine",
            source_file="test/fixture-small.jsonl"
        )
        assert conv_id is not None

        # Verify in DB
        with db_conn.cursor() as cur:
            cur.execute("SELECT message_count FROM conversations WHERE id = %s", (conv_id,))
            row = cur.fetchone()
            assert row[0] == parsed.message_count

        # Cleanup
        with db_conn.cursor() as cur:
            cur.execute("DELETE FROM conversations WHERE id = %s", (conv_id,))
        db_conn.commit()

    def test_idempotent_upsert(self, db_conn):
        parsed = parse_conversation(SMALL_FIXTURE)

        # Insert twice
        conv_id1 = upsert_conversation(
            db_conn, parsed,
            user_name="test_user", machine="test_machine",
            source_file="test/fixture-small.jsonl"
        )
        conv_id2 = upsert_conversation(
            db_conn, parsed,
            user_name="test_user", machine="test_machine",
            source_file="test/fixture-small.jsonl"
        )

        assert conv_id1 == conv_id2  # Same conversation, same ID

        # Verify only one conversation exists
        with db_conn.cursor() as cur:
            cur.execute(
                "SELECT COUNT(*) FROM conversations WHERE session_id = %s AND machine = %s",
                (parsed.session_id, "test_machine")
            )
            assert cur.fetchone()[0] == 1

            # Verify correct message count (not doubled)
            cur.execute(
                "SELECT COUNT(*) FROM messages WHERE conversation_id = %s",
                (conv_id1,)
            )
            assert cur.fetchone()[0] == parsed.message_count

        # Cleanup
        with db_conn.cursor() as cur:
            cur.execute("DELETE FROM conversations WHERE id = %s", (conv_id1,))
        db_conn.commit()

    def test_text_search_works(self, db_conn):
        parsed = parse_conversation(SMALL_FIXTURE)
        conv_id = upsert_conversation(
            db_conn, parsed,
            user_name="test_user", machine="test_machine",
            source_file="test/fixture-small.jsonl"
        )

        # Trigram search
        with db_conn.cursor() as cur:
            cur.execute(
                "SELECT COUNT(*) FROM messages WHERE conversation_id = %s AND content IS NOT NULL",
                (conv_id,)
            )
            has_content = cur.fetchone()[0] > 0
            assert has_content

        # Cleanup
        with db_conn.cursor() as cur:
            cur.execute("DELETE FROM conversations WHERE id = %s", (conv_id,))
        db_conn.commit()




class TestAppendOnlyFastPath:
    """The sync cron re-ingests a live transcript every minute. Before 2026-08-28
    each pass DELETEd and re-INSERTed every message of that session, so WAL scaled
    with (session size x times saved) — measured at 25-33 GB/day off a 2.5 GB
    table, filling the backup volume. These tests pin the two properties that make
    appending safe: take the fast path on a real append, REFUSE it on anything not
    provably a prefix.

    ⚠️ Every test gets its OWN session_id and cleans up in a finally. They shared
    the fixture's session_id at first, so one failure skipped its cleanup and the
    next three died on SourceFileCollision instead of their own assertion — four
    reds, one of them real. A cascading suite cannot tell you which check fired.
    """

    MACHINE = "test_append_machine"

    @staticmethod
    def _file(tmp_path, n_lines=None, name="t.jsonl"):
        """A transcript, optionally truncated to its first n_lines — i.e. the same
        session mid-flight."""
        lines = SMALL_FIXTURE.read_text().splitlines(keepends=True)
        p = tmp_path / name
        p.write_text("".join(lines if n_lines is None else lines[:n_lines]))
        return p

    def _ingest(self, db_conn, path, session_id):
        parsed = parse_conversation(path)
        parsed.session_id = session_id
        conv_id = upsert_conversation(
            db_conn, parsed, user_name="test_user", machine=self.MACHINE,
            source_file=str(path),
        )
        return parsed, conv_id

    @staticmethod
    def _rows(cur, conv_id):
        """sequence_num -> xmin. xmin is the load-bearing part: counting rows cannot
        tell an append from a full rewrite, because both end with the right number of
        correct rows. xmin changes if and only if the row was written again."""
        cur.execute(
            "SELECT sequence_num, xmin::text FROM messages WHERE conversation_id = %s",
            (conv_id,),
        )
        return dict(cur.fetchall())

    def _drop(self, db_conn, session_id):
        with db_conn.cursor() as cur:
            cur.execute(
                "DELETE FROM conversations WHERE session_id = %s AND machine = %s",
                (session_id, self.MACHINE),
            )
        db_conn.commit()

    def test_append_does_not_rewrite_existing_rows(self, db_conn, tmp_path):
        sid = "append-grow-test"
        try:
            f = self._file(tmp_path, 20, "grow.jsonl")
            first, conv_id = self._ingest(db_conn, f, sid)
            assert first.message_count > 0, "prefix parsed to nothing — test is vacuous"
            with db_conn.cursor() as cur:
                before = self._rows(cur, conv_id)

            f.write_text(SMALL_FIXTURE.read_text())   # the file grows, as a live one does
            full, conv_id2 = self._ingest(db_conn, f, sid)
            assert conv_id2 == conv_id
            assert full.message_count > first.message_count, \
                "full fixture no longer than its prefix — test is vacuous"

            with db_conn.cursor() as cur:
                after = self._rows(cur, conv_id)
            assert len(after) == full.message_count          # correctness
            assert {s: v for s, v in after.items() if s in before} == before, \
                "existing message rows were rewritten — the DELETE path ran"
        finally:
            self._drop(db_conn, sid)

    def test_compaction_is_not_mistaken_for_an_append(self, db_conn, tmp_path):
        """⛔ Why a tail-only check is unsafe: a rewritten transcript can be LONGER
        than what is stored while sharing none of its messages."""
        sid = "append-compaction-test"
        try:
            f = self._file(tmp_path, 20, "compact.jsonl")
            self._ingest(db_conn, f, sid)
            with db_conn.cursor() as cur:
                cur.execute(
                    "UPDATE messages SET uuid = 'stale-' || sequence_num::text "
                    "WHERE conversation_id = (SELECT id FROM conversations "
                    "WHERE session_id = %s AND machine = %s)", (sid, self.MACHINE),
                )
            db_conn.commit()

            f.write_text(SMALL_FIXTURE.read_text())
            full, conv_id = self._ingest(db_conn, f, sid)

            with db_conn.cursor() as cur:
                cur.execute(
                    "SELECT COUNT(*) FILTER (WHERE uuid LIKE 'stale-%%'), COUNT(*) "
                    "FROM messages WHERE conversation_id = %s", (conv_id,),
                )
                stale, total = cur.fetchone()
            assert stale == 0, "divergent rows survived — fast path taken on a non-prefix"
            assert total == full.message_count
        finally:
            self._drop(db_conn, sid)

    def test_edited_content_falls_back_to_replace(self, db_conn, tmp_path):
        """Same uuids, changed content. A uuid-only check would call this a prefix.
        (Since 2026-11-08 the edited row and everything after it are rewritten,
        not the whole conversation; the assertion is that nothing stale survives.)"""
        sid = "append-edit-test"
        try:
            f = self._file(tmp_path, 20, "edit.jsonl")
            self._ingest(db_conn, f, sid)
            with db_conn.cursor() as cur:
                cur.execute(
                    "UPDATE messages SET content_hash = 'deadbeef' WHERE conversation_id = "
                    "(SELECT id FROM conversations WHERE session_id = %s AND machine = %s) "
                    "AND sequence_num = (SELECT MAX(sequence_num) FROM messages "
                    "WHERE conversation_id = (SELECT id FROM conversations "
                    "WHERE session_id = %s AND machine = %s))",
                    (sid, self.MACHINE, sid, self.MACHINE),
                )
                assert cur.rowcount == 1, "fault injection did not land — test is vacuous"
            db_conn.commit()

            f.write_text(SMALL_FIXTURE.read_text())
            full, conv_id = self._ingest(db_conn, f, sid)
            with db_conn.cursor() as cur:
                cur.execute(
                    "SELECT COUNT(*) FILTER (WHERE content_hash = 'deadbeef'), COUNT(*) "
                    "FROM messages WHERE conversation_id = %s", (conv_id,),
                )
                edited, total = cur.fetchone()
            assert edited == 0, "edited row survived — content was not checked"
            assert total == full.message_count
        finally:
            self._drop(db_conn, sid)

    def test_reingest_of_unchanged_file_is_idempotent_and_writes_nothing(self, db_conn, tmp_path):
        """Idempotency still holds (no doubled rows), and since 2026-11-08 an
        unchanged file no longer rewrites anything: every row is verified."""
        sid = "append-unchanged-test"
        try:
            f = self._file(tmp_path, 20, "same.jsonl")
            parsed, conv_id = self._ingest(db_conn, f, sid)
            with db_conn.cursor() as cur:
                before = self._rows(cur, conv_id)
            self._ingest(db_conn, f, sid)
            with db_conn.cursor() as cur:
                cur.execute("SELECT COUNT(*) FROM messages WHERE conversation_id = %s", (conv_id,))
                assert cur.fetchone()[0] == parsed.message_count
                assert self._rows(cur, conv_id) == before, "unchanged file rewrote rows"
        finally:
            self._drop(db_conn, sid)


class TestSuffixRewrite:
    """Narrowed rewrite: keep the longest stored prefix verified row by row
    (uuid + content_hash), rewrite only what follows it.

    Measured 2026-11-08 on the live DB: every full rewrite in the log was a pass
    where the MESSAGE COUNT WAS UNCHANGED, because the transcript had only grown
    by records the parser ignores. One 27k-message session was rewritten in full
    517 times; that was ~2 GB/h of WAL. These tests assert on xmin per row, which
    changes if and only if the row was written again, so 'kept' and 'rewritten
    with identical data' cannot be confused.
    """

    MACHINE = "test_suffix_machine"

    @staticmethod
    def _records():
        import json
        return [json.loads(l) for l in SMALL_FIXTURE.read_text().splitlines() if l.strip()]

    @staticmethod
    def _message_lines(records):
        """Indices of records the parser turns into messages, in sequence order."""
        return [i for i, r in enumerate(records) if r.get("type") in ("user", "assistant", "system")]

    @staticmethod
    def _write(path, records):
        import json
        path.write_text("".join(json.dumps(r) + "\n" for r in records))
        return path

    def _ingest(self, db_conn, path, session_id, source_file=None):
        parsed = parse_conversation(path)
        parsed.session_id = session_id
        conv_id = upsert_conversation(
            db_conn, parsed, user_name="test_user", machine=self.MACHINE,
            source_file=source_file or str(path),
        )
        return parsed, conv_id

    @staticmethod
    def _rows(db_conn, conv_id):
        """sequence_num -> (xmin, uuid, content_hash)."""
        with db_conn.cursor() as cur:
            cur.execute(
                "SELECT sequence_num, xmin::text, uuid, content_hash FROM messages "
                "WHERE conversation_id = %s", (conv_id,),
            )
            return {r[0]: r[1:] for r in cur.fetchall()}

    @staticmethod
    def _assert_matches_parse(rows, parsed):
        from v2.db import _content_hash
        assert sorted(rows) == list(range(parsed.message_count)), "rows are not exactly seq 0..n-1"
        for m in parsed.messages:
            _, uuid_, h = rows[m.sequence_num]
            assert (uuid_, h) == (m.uuid, _content_hash(m.content)), f"seq {m.sequence_num} stale"

    def _drop(self, db_conn, session_id):
        db_conn.rollback()
        with db_conn.cursor() as cur:
            cur.execute(
                "DELETE FROM conversations WHERE session_id = %s AND machine = %s",
                (session_id, self.MACHINE),
            )
        db_conn.commit()

    @staticmethod
    def _rewritten(before, after):
        return {s for s in before if s in after and before[s][0] != after[s][0]}

    def test_non_message_growth_rewrites_nothing(self, db_conn, tmp_path):
        """THE live case: the file grew by ignored records only."""
        sid = "suffix-ignored-growth-test"
        try:
            recs = self._records()
            f = self._write(tmp_path / "a.jsonl", recs)
            parsed, conv_id = self._ingest(db_conn, f, sid)
            before = self._rows(db_conn, conv_id)
            assert len(before) == parsed.message_count > 0

            noise = [{"type": "queue-operation", "operation": "enqueue", "sessionId": "x"},
                     {"type": "last-prompt"}, {"type": "ai-title"},
                     {"type": "attachment", "uuid": "att-1"}]
            self._write(f, recs + noise)
            again, _ = self._ingest(db_conn, f, sid)
            assert again.message_count == parsed.message_count, "noise parsed as messages — vacuous"

            after = self._rows(db_conn, conv_id)
            assert after == before, "rows were rewritten although no message changed"
        finally:
            self._drop(db_conn, sid)

    def test_append_writes_only_new_rows(self, db_conn, tmp_path):
        sid = "suffix-append-test"
        try:
            recs = self._records()
            cut = self._message_lines(recs)[10]
            f = self._write(tmp_path / "a.jsonl", recs[:cut])
            first, conv_id = self._ingest(db_conn, f, sid)
            before = self._rows(db_conn, conv_id)

            self._write(f, recs)
            full, _ = self._ingest(db_conn, f, sid)
            assert full.message_count > first.message_count
            after = self._rows(db_conn, conv_id)
            assert self._rewritten(before, after) == set()
            self._assert_matches_parse(after, full)
        finally:
            self._drop(db_conn, sid)

    def test_tail_mutation_with_growth_rewrites_only_from_the_last_row(self, db_conn, tmp_path):
        """The stored last message was a partial version; the file now holds its
        final version plus more messages."""
        sid = "suffix-tail-mutation-grow-test"
        try:
            recs = self._records()
            idx = self._message_lines(recs)
            cut = idx[10]
            partial = [dict(r) for r in recs[:cut]]
            last = idx[9]
            partial[last] = dict(partial[last], message={"role": "user", "content": "PARTIAL"})
            f = self._write(tmp_path / "a.jsonl", partial)
            first, conv_id = self._ingest(db_conn, f, sid)
            before = self._rows(db_conn, conv_id)
            assert first.messages[9].content == "PARTIAL", "mutation did not land — vacuous"

            self._write(f, recs)
            full, _ = self._ingest(db_conn, f, sid)
            after = self._rows(db_conn, conv_id)
            assert self._rewritten(before, after) == {9}, "only the mutated last row may be rewritten"
            self._assert_matches_parse(after, full)
        finally:
            self._drop(db_conn, sid)

    def test_same_count_mutated_last_message_rewrites_only_it(self, db_conn, tmp_path):
        sid = "suffix-same-count-test"
        try:
            recs = self._records()
            idx = self._message_lines(recs)
            f = self._write(tmp_path / "a.jsonl", recs)
            parsed, conv_id = self._ingest(db_conn, f, sid)
            before = self._rows(db_conn, conv_id)

            mutated = [dict(r) for r in recs]
            mutated[idx[-1]] = dict(mutated[idx[-1]], message={"role": "assistant", "content": "FINAL"})
            self._write(f, mutated)
            again, _ = self._ingest(db_conn, f, sid)
            assert again.message_count == parsed.message_count
            after = self._rows(db_conn, conv_id)
            assert self._rewritten(before, after) == {parsed.message_count - 1}
            self._assert_matches_parse(after, again)
        finally:
            self._drop(db_conn, sid)

    def test_compaction_head_rewrite_is_a_full_replace(self, db_conn, tmp_path):
        """⛔ A head mismatch must rewrite everything, even though every later
        row would match on its own."""
        sid = "suffix-compaction-head-test"
        try:
            recs = self._records()
            idx = self._message_lines(recs)
            f = self._write(tmp_path / "a.jsonl", recs[: idx[10]])
            _, conv_id = self._ingest(db_conn, f, sid)
            before = self._rows(db_conn, conv_id)

            compacted = [dict(r) for r in recs]
            compacted[idx[0]] = dict(compacted[idx[0]], uuid="compact-summary-uuid")
            self._write(f, compacted)
            full, _ = self._ingest(db_conn, f, sid)
            after = self._rows(db_conn, conv_id)
            assert self._rewritten(before, after) == set(before), "head mismatch did not rewrite every row"
            self._assert_matches_parse(after, full)
        finally:
            self._drop(db_conn, sid)

    def test_duplicate_stored_sequence_is_a_full_replace(self, db_conn, tmp_path):
        sid = "suffix-duplicate-seq-test"
        try:
            f = self._write(tmp_path / "a.jsonl", self._records())
            parsed, conv_id = self._ingest(db_conn, f, sid)
            with db_conn.cursor() as cur:
                cur.execute(
                    "UPDATE messages SET sequence_num = 1 WHERE conversation_id = %s AND sequence_num = 5",
                    (conv_id,),
                )
                assert cur.rowcount == 1
            db_conn.commit()
            again, _ = self._ingest(db_conn, f, sid)
            self._assert_matches_parse(self._rows(db_conn, conv_id), again)
        finally:
            self._drop(db_conn, sid)

    def test_source_guard_runs_before_the_suffix_delete(self, db_conn, tmp_path):
        """An unrelated file claiming the session must be refused, and nothing
        deleted, even when its messages share a prefix (the suffix path)."""
        from v2.db import SourceFileCollision
        sid = "suffix-source-guard-test"
        try:
            recs = self._records()
            idx = self._message_lines(recs)
            a = self._write(tmp_path / "a.jsonl", recs)
            _, conv_id = self._ingest(db_conn, a, sid)
            before = self._rows(db_conn, conv_id)

            other = [dict(r) for r in recs]
            other[idx[-1]] = dict(other[idx[-1]], message={"role": "assistant", "content": "OTHER"})
            b = self._write(tmp_path / "b.jsonl", other)
            with pytest.raises(SourceFileCollision):
                self._ingest(db_conn, b, sid)
            db_conn.rollback()
            assert self._rows(db_conn, conv_id) == before, "rows touched despite the guard"
        finally:
            self._drop(db_conn, sid)

    def test_source_guard_blocks_unchanged_and_append_paths_too(self, db_conn, tmp_path):
        """A collision is refused even when nothing would be deleted."""
        from v2.db import SourceFileCollision
        sid = "suffix-source-guard-nodelete-test"
        try:
            recs = self._records()
            idx = self._message_lines(recs)
            a = self._write(tmp_path / "a.jsonl", recs[: idx[10]])
            _, conv_id = self._ingest(db_conn, a, sid)
            before = self._rows(db_conn, conv_id)
            for name, content in (("same.jsonl", recs[: idx[10]]), ("grown.jsonl", recs)):
                b = self._write(tmp_path / name, content)
                with pytest.raises(SourceFileCollision):
                    self._ingest(db_conn, b, sid)
                db_conn.rollback()
            assert self._rows(db_conn, conv_id) == before
        finally:
            self._drop(db_conn, sid)

    def test_no_delete_is_issued_before_the_source_guard(self, db_conn, tmp_path):
        """Ordering, not just outcome: a DELETE issued before the guard raises is
        undone by the rollback, so the row-level tests above cannot see it. The
        guard must run before any DELETE reaches the server."""
        from v2.db import SourceFileCollision
        sid = "suffix-guard-order-test"
        try:
            recs = self._records()
            idx = self._message_lines(recs)
            a = self._write(tmp_path / "a.jsonl", recs)
            self._ingest(db_conn, a, sid)

            issued = []

            class RecordingCursor:
                def __init__(self, real):
                    self._real = real
                def __enter__(self):
                    self._real.__enter__()
                    return self
                def __exit__(self, *exc):
                    return self._real.__exit__(*exc)
                def execute(self, sql, *args, **kw):
                    issued.append(sql)
                    return self._real.execute(sql, *args, **kw)
                def __getattr__(self, name):
                    return getattr(self._real, name)

            class RecordingConn:
                def __init__(self, real):
                    self._real = real
                def cursor(self, *a, **kw):
                    return RecordingCursor(self._real.cursor(*a, **kw))
                def __getattr__(self, name):
                    return getattr(self._real, name)

            for variant in ("replace", "suffix"):
                other = [dict(r) for r in recs]
                if variant == "replace":
                    other[idx[0]] = dict(other[idx[0]], uuid="other-head")
                else:
                    other[idx[-1]] = dict(other[idx[-1]], message={"role": "assistant", "content": "X"})
                b = self._write(tmp_path / f"{variant}.jsonl", other)
                parsed = parse_conversation(b)
                parsed.session_id = sid
                issued.clear()
                with pytest.raises(SourceFileCollision):
                    upsert_conversation(RecordingConn(db_conn), parsed, user_name="test_user",
                                        machine=self.MACHINE, source_file=str(b))
                db_conn.rollback()
                assert issued, "recorder saw nothing — test is vacuous"
                assert not any("DELETE" in s.upper() for s in issued), \
                    f"{variant}: DELETE issued before the source guard: {issued}"
        finally:
            self._drop(db_conn, sid)
