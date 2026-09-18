#!/usr/bin/env python3
"""Analyze a docket's comments through the Anthropic Message Batches API.

Why this exists as a separate script instead of a flag on pipeline.py: the
Batches API is submit-then-collect, which does not fit pipeline.py's synchronous
worker pool. Rather than restructure that loop (and diverge from upstream), this
writes its results into the pipeline's own `.analysis_checkpoint.jsonl`. A normal

    python pipeline.py --regulation <slug>

run afterwards then finds every comment already analyzed, makes zero LLM calls,
and builds the parquet and the report exactly as it always does. The checkpoint
is keyed on normalized comment text, same as pipeline._checkpoint_key, so the two
agree on what "already done" means.

Batch pricing is 50% of standard. Cost for a docket this size is dominated by the
fixed prompt prefix (system prompt + tool schema), which is identical on every
request, so the prefix is marked for prompt caching as well.

Usage:
    python batch_analyze.py --regulation <slug> --estimate      # price it, no spend
    python batch_analyze.py --regulation <slug>                 # submit, wait, collect
    python batch_analyze.py --regulation <slug> --collect-only  # resume a submitted batch
"""
import argparse
import json
import logging
import os
import sys
import time

import anthropic
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), '.env'))

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, REPO_ROOT)

BATCH_STATE_FILE = '.batch_state.json'
CHECKPOINT_FILE = '.analysis_checkpoint.jsonl'
TOOL_NAME = 'record_analysis'

# The Batches API caps a single batch at 100k requests / 256MB. A docket of a few
# thousand comments fits in one, but chunking keeps any single failure cheap to
# redo and lets long dockets stream through.
CHUNK_SIZE = 2000

# Standard per-MTok rates; batch is half. Kept here only to price a run up front —
# the authoritative number is the usage reported after collection.
PRICES = {
    'claude-opus-5':   (5.00, 25.00),
    'claude-sonnet-5': (2.00, 10.00),
    'claude-haiku-4-5': (1.00, 5.00),
}


def _checkpoint_key(text: str) -> str:
    """Must match pipeline._checkpoint_key exactly, or the handoff silently misses."""
    return (text or '').strip().lower()


def build_tool(analyzer):
    """The config-derived Pydantic model, exposed as the one tool Claude must call.

    Reusing analyzer.result_model means the batch path and the live path validate
    against the same schema — a field added to analyzer_config.yaml shows up in
    both without touching this file.
    """
    return {
        'name': TOOL_NAME,
        'description': 'Record the structured analysis of this public comment.',
        'input_schema': analyzer.result_model.model_json_schema(),
        'cache_control': {'type': 'ephemeral'},
    }


def build_request(comment, system_prompt, tool, model, max_tokens):
    """One batch request. custom_id carries the comment id back on collection."""
    text = comment['text']
    parts = []
    if comment.get('organization'):
        parts.append(f"Submitting organization: {comment['organization']}")
    if comment.get('submitter'):
        parts.append(f"Submitter: {comment['submitter']}")
    parts.append(f"Comment text:\n\n{text}")

    return {
        'custom_id': comment['_batch_id'],
        'params': {
            'model': model,
            'max_tokens': max_tokens,
            'system': [{'type': 'text', 'text': system_prompt,
                        'cache_control': {'type': 'ephemeral'}}],
            'tools': [tool],
            'tool_choice': {'type': 'tool', 'name': TOOL_NAME},
            'messages': [{'role': 'user', 'content': '\n'.join(parts)}],
        },
    }


def load_unique_comments(csv_file, limit=None):
    """Load and dedup with the pipeline's own functions, so ids and text match."""
    from pipeline import read_comments_from_csv, create_dedup_table

    comments = read_comments_from_csv(csv_file, limit=limit)
    unique, _dup_map = create_dedup_table(comments)
    logger.info(f"{len(comments)} comments -> {len(unique)} unique texts to analyze")
    return comments, unique


def already_done() -> set:
    """Text keys the checkpoint already covers — never pay for these twice."""
    done = set()
    if os.path.exists(CHECKPOINT_FILE):
        with open(CHECKPOINT_FILE) as f:
            for line in f:
                try:
                    e = json.loads(line)
                except json.JSONDecodeError:
                    continue
                # An entry with no analysis is a recorded failure, not a result.
                if e.get('text_key') and e.get('analysis'):
                    done.add(e['text_key'])
    return done


def estimate(client, model, system_prompt, tool, pending, max_tokens):
    """Price the run before spending anything.

    Token-counts a real sample rather than guessing from character counts: the
    fixed prefix (system prompt + tool schema) dominates, and it is measured
    exactly. Output is estimated from max_tokens-capped typical completions.
    """
    sample = pending[: min(25, len(pending))]
    counts = []
    for c in sample:
        req = build_request(c, system_prompt, tool, model, max_tokens)['params']
        r = client.messages.count_tokens(
            model=model, system=req['system'], tools=req['tools'],
            messages=req['messages'])
        counts.append(r.input_tokens)

    mean_in = sum(counts) / len(counts)
    total_in = mean_in * len(pending)
    # Structured output for this schema runs ~250-350 tokens; 300 is the midpoint.
    est_out_each = 300
    total_out = est_out_each * len(pending)

    in_rate, out_rate = PRICES.get(model, (0, 0))
    std = total_in / 1e6 * in_rate + total_out / 1e6 * out_rate
    batch = std / 2

    print()
    print(f"  model                 {model}")
    print(f"  comments to analyze   {len(pending):,}")
    print(f"  mean input tokens     {mean_in:,.0f}  (measured on {len(sample)} real comments)")
    print(f"  est. output tokens    {est_out_each} each")
    print(f"  total input           {total_in/1e6:,.2f}M tokens")
    print(f"  total output          {total_out/1e6:,.2f}M tokens")
    print(f"  standard price        ${std:,.2f}")
    print(f"  BATCH price (50%)     ${batch:,.2f}   <- what this run costs")
    print(f"  (prompt caching may reduce the input side further)")
    print()
    return batch


def submit(client, requests_):
    """Submit in chunks; return the batch ids."""
    ids = []
    for i in range(0, len(requests_), CHUNK_SIZE):
        chunk = requests_[i:i + CHUNK_SIZE]
        b = client.messages.batches.create(requests=chunk)
        ids.append(b.id)
        logger.info(f"Submitted batch {b.id} ({len(chunk)} requests)")
    return ids


def wait_for(client, batch_ids, poll_seconds=30):
    """Poll until every batch has ended."""
    pending = list(batch_ids)
    while pending:
        still = []
        for bid in pending:
            b = client.messages.batches.retrieve(bid)
            if b.processing_status == 'ended':
                c = b.request_counts
                logger.info(f"{bid} ended — succeeded={c.succeeded} errored={c.errored} "
                            f"canceled={c.canceled} expired={c.expired}")
            else:
                still.append(bid)
        pending = still
        if pending:
            logger.info(f"{len(pending)} batch(es) still processing; sleeping {poll_seconds}s")
            time.sleep(poll_seconds)


def collect(client, batch_ids, by_batch_id, model, analyzer):
    """Read results and append them to the pipeline's checkpoint.

    Results arrive in arbitrary order, so everything is keyed by custom_id.
    Each tool input is validated against the same Pydantic model the live path
    uses; anything that fails validation is recorded as an error rather than
    written as if it were a clean result.
    """
    ok = err = 0
    usage_in = usage_out = cache_read = cache_write = 0

    with open(CHECKPOINT_FILE, 'a') as out:
        for bid in batch_ids:
            for result in client.messages.batches.results(bid):
                comment = by_batch_id.get(result.custom_id)
                if comment is None:
                    logger.warning(f"Result for unknown custom_id {result.custom_id}")
                    continue

                entry = {'text_key': _checkpoint_key(comment['text']),
                         'id': comment['id'], 'analysis': None,
                         'analysis_error': None, 'model_used': model}

                if result.result.type != 'succeeded':
                    entry['analysis_error'] = f"batch result: {result.result.type}"
                    err += 1
                else:
                    msg = result.result.message
                    u = msg.usage
                    usage_in += u.input_tokens
                    usage_out += u.output_tokens
                    cache_read += getattr(u, 'cache_read_input_tokens', 0) or 0
                    cache_write += getattr(u, 'cache_creation_input_tokens', 0) or 0

                    block = next((b for b in msg.content
                                  if b.type == 'tool_use' and b.name == TOOL_NAME), None)
                    if block is None:
                        entry['analysis_error'] = f"no tool_use block (stop_reason={msg.stop_reason})"
                        err += 1
                    else:
                        try:
                            entry['analysis'] = analyzer.result_model(**block.input).model_dump()
                            ok += 1
                        except Exception as e:
                            entry['analysis_error'] = f"schema validation failed: {e}"
                            err += 1

                out.write(json.dumps(entry) + '\n')

    in_rate, out_rate = PRICES.get(model, (0, 0))
    billed_in = usage_in + cache_write * 1.25 + cache_read * 0.1
    spend = (billed_in / 1e6 * in_rate + usage_out / 1e6 * out_rate) / 2

    logger.info(f"Collected {ok} analyses, {err} errors")
    logger.info(f"Tokens — input {usage_in:,} | output {usage_out:,} | "
                f"cache read {cache_read:,} | cache write {cache_write:,}")
    logger.info(f"Actual batch spend: ~${spend:,.2f}")
    return ok, err


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--regulation', required=True,
                   help='Slug under regulations/<slug>/; the script chdirs into it.')
    p.add_argument('--model', default='claude-sonnet-5',
                   help='Anthropic model id (default: claude-sonnet-5)')
    p.add_argument('--max-tokens', type=int, default=2048)
    p.add_argument('--limit', type=int, help='Only load the first N comments (testing)')
    p.add_argument('--estimate', action='store_true',
                   help='Price the run and exit without submitting anything.')
    p.add_argument('--collect-only', action='store_true',
                   help='Skip submission; collect batches recorded in .batch_state.json')
    p.add_argument('--yes', action='store_true', help='Skip the spend confirmation prompt.')
    args = p.parse_args()

    reg_dir = os.path.join(REPO_ROOT, 'regulations', args.regulation)
    if not os.path.isdir(reg_dir):
        sys.exit(f"No such regulation directory: {reg_dir}")
    os.chdir(reg_dir)

    if not os.getenv('ANTHROPIC_API_KEY'):
        sys.exit("ANTHROPIC_API_KEY not set (put it in the repo-root .env)")

    from comment_analyzer import CommentAnalyzer
    # CommentAnalyzer's constructor insists on OPENAI_API_KEY because the live
    # path goes through LiteLLM's OpenAI provider. This path never calls LiteLLM,
    # so satisfy the check without implying a real OpenAI key is configured.
    os.environ.setdefault('OPENAI_API_KEY', 'unused-by-batch-path')
    analyzer = CommentAnalyzer(model=args.model, config_file='analyzer_config.yaml')

    system_prompt = analyzer.get_system_prompt()
    tool = build_tool(analyzer)
    client = anthropic.Anthropic()

    if not os.path.exists('source.csv'):
        sys.exit("source.csv not found. Fetch comments first:\n"
                 f"  python fetch_comments_api.py --regulation {args.regulation}")

    _all, unique = load_unique_comments('source.csv', limit=args.limit)
    for c in unique:
        c['_batch_id'] = c['id'].replace('#', '_')[:64]
    by_batch_id = {c['_batch_id']: c for c in unique}

    done = already_done()
    pending = [c for c in unique if _checkpoint_key(c['text']) not in done]
    logger.info(f"{len(done)} already in checkpoint; {len(pending)} to analyze")

    if args.collect_only:
        if not os.path.exists(BATCH_STATE_FILE):
            sys.exit(f"No {BATCH_STATE_FILE} to collect from.")
        state = json.load(open(BATCH_STATE_FILE))
        wait_for(client, state['batch_ids'])
        collect(client, state['batch_ids'], by_batch_id, state['model'], analyzer)
        return

    if not pending:
        logger.info("Nothing to do — every comment is already analyzed.")
        return

    cost = estimate(client, args.model, system_prompt, tool, pending, args.max_tokens)
    if args.estimate:
        return

    if not args.yes:
        if input(f"Submit {len(pending):,} comments (~${cost:,.2f})? [y/N] ").strip().lower() != 'y':
            sys.exit("Aborted.")

    requests_ = [build_request(c, system_prompt, tool, args.model, args.max_tokens)
                 for c in pending]
    batch_ids = submit(client, requests_)
    json.dump({'batch_ids': batch_ids, 'model': args.model, 'submitted': time.time()},
              open(BATCH_STATE_FILE, 'w'), indent=2)
    logger.info(f"Recorded batch ids in {BATCH_STATE_FILE} — "
                f"safe to Ctrl-C and resume with --collect-only")

    wait_for(client, batch_ids)
    collect(client, batch_ids, by_batch_id, args.model, analyzer)

    print(f"\nDone. Now build the report:\n"
          f"  python pipeline.py --regulation {args.regulation}\n")


if __name__ == '__main__':
    main()
