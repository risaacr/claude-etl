#!/bin/bash
# Claude ETL hourly cron job
# Location: /home/rrobinson/projects/claude-etl/cron-ingest.sh (server)
# Crontab: 0 * * * * /home/rrobinson/projects/claude-etl/cron-ingest.sh
#
# Pipeline:
#   1. code_ingest.py - Ingest Claude Code conversations (synced via Obsidian)
#   2. web_ingest.py  - Ingest Claude Web exports (if any in inbox)
#   3. chatgpt_ingest.py - Ingest ChatGPT exports
#   4. normalize.py   - Transform raw data into normalized messages
#   5. embed.py       - Generate embeddings for new messages
#   6. chunk_conversations.py - Create topic chunks with summaries

set -e

# Server paths
SCRIPT_DIR="/home/rrobinson/projects/claude-etl"
LOG_DIR="$HOME/corpus/isaac-workspace-corpus/var/log"
LOG_FILE="$LOG_DIR/claude-etl-ingest.log"
LOCK_FILE="/tmp/claude-etl.lock"

# Ensure log directory exists
mkdir -p "$LOG_DIR"

# Prevent concurrent runs (chunk_conversations.py takes minutes per conversation)
if [ -f "$LOCK_FILE" ]; then
    pid=$(cat "$LOCK_FILE")
    if kill -0 "$pid" 2>/dev/null; then
        echo "$(date -Iseconds) SKIPPED: previous run still active (PID $pid)" >> "$LOG_FILE"
        exit 0
    fi
    rm -f "$LOCK_FILE"
fi
echo $$ > "$LOCK_FILE"
trap 'rm -f "$LOCK_FILE"' EXIT

cd "$SCRIPT_DIR"

# Run full pipeline with timestamp
{
    echo "=== $(date -Iseconds) ==="

    echo "[1/6] Code ingestion..."
    ./venv/bin/python3 code_ingest.py 2>&1 || echo "  code_ingest failed"

    echo "[2/6] Web ingestion..."
    ./venv/bin/python3 web_ingest.py 2>&1 || echo "  web_ingest failed (no new exports?)"

    echo "[3/6] ChatGPT ingestion..."
    ./venv/bin/python3 chatgpt_ingest.py --source inbox 2>&1 || echo "  chatgpt_ingest failed (no new exports?)"

    echo "[4/6] Normalization..."
    ./venv/bin/python3 normalize.py 2>&1 || echo "  normalize failed"

    echo "[5/6] Embeddings (limit 1000)..."
    ./venv/bin/python3 embed.py --limit 1000 2>&1 || echo "  embed failed"

    echo "[6/6] Topic chunking (limit 50)..."
    ./venv/bin/python3 chunk_conversations.py --source both --limit 50 2>&1 || echo "  chunk failed"

    echo "Done."
    echo ""
} >> "$LOG_FILE"

# Keep log file from growing forever (keep last 2000 lines)
tail -n 2000 "$LOG_FILE" > "$LOG_FILE.tmp" && mv "$LOG_FILE.tmp" "$LOG_FILE"
