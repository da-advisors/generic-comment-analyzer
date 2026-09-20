#!/usr/bin/env python3
"""Backfill one newly-added `fields:` entry across an already-analysed corpus.

Adding a field to analyzer_config.yaml changes the schema but not the comment
text, and the pipeline's reuse cache is keyed on text — so every existing row is
reused and the new field is silently never populated. Re-running the pipeline
with --reprocess would fix that, but at a cost beyond the new field:

  * it rebuilds every analysis dict, discarding `verified_stance` and
    `verification_reasoning` on every row the second pass has already settled,
    so the verification has to be paid for again;
  * it re-derives every other field under a changed system prompt, so existing
    classifications can move for reasons unrelated to the field being added, and
    the published headline shifts with no way to attribute the change.

This asks only for the new field, for the rows that lack it, and merges the
answer into the existing analysis. Everything already settled stays settled.

The field's options and prompt are read from the config, so this file has no
per-field knowledge and the config remains the single source of truth.

Usage:
    python extract_field.py --regulation <slug> --field procedural [--estimate]
"""
import argparse
import concurrent.futures as cf
import json
import os
import sys
import threading

import anthropic
import pandas as pd
import yaml
from dotenv import load_dotenv

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(REPO_ROOT, '.env'))

_local = threading.local()


def client():
    if not hasattr(_local, 'c'):
        _local.c = anthropic.Anthropic()
    return _local.c


def field_spec(config, name):
    for f in config.get('fields', []):
        if f.get('name') == name:
            return f
    raise SystemExit(f"No field named {name!r} in analyzer_config.yaml")


def build_tool(spec):
    """Schema for this one field, from its declared type and options."""
    name, ftype = spec['name'], spec.get('type', 'text')
    opts = spec.get('options') or []
    if ftype == 'multi_enum':
        prop = {'type': 'array', 'items': {'type': 'string', 'enum': opts},
                'description': 'Every option that applies; empty if none do.'}
    elif ftype == 'single_enum':
        prop = {'type': 'string', 'enum': opts}
    elif ftype == 'enum_or_empty':
        prop = {'type': 'string', 'enum': list(opts) + ['']}
    else:
        prop = {'type': 'string'}
    return {
        'name': f'record_{name}',
        'description': f"Record the {spec.get('label', name)} for this comment.",
        'input_schema': {'type': 'object', 'properties': {name: prop},
                         'required': [name], 'additionalProperties': False},
    }


def system_prompt(config, spec):
    """Enough context to judge the field, and nothing else.

    Deliberately not the full analysis prompt: this pass must not re-litigate
    stance or entity type, and a prompt that described them would invite the
    model to weigh them.
    """
    parts = [f"You are analyzing public comments on a proposed rule: "
             f"{config.get('regulation_name', '')}.",
             (config.get('regulation_description') or '').strip(), '',
             f"Record one field: {spec.get('label', spec['name'])}.", '',
             (spec.get('prompt') or '').strip()]
    if spec.get('options'):
        parts += ['', 'Options (exact matches only):']
        parts += [f"- {o}" for o in spec['options']]
    return '\n'.join(p for p in parts if p is not None)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--regulation', required=True)
    ap.add_argument('--field', required=True)
    ap.add_argument('--model', default=os.getenv('LLM_MODEL', 'anthropic/claude-sonnet-5'))
    ap.add_argument('--workers', type=int, default=12)
    ap.add_argument('--estimate', action='store_true', help='Price it and exit.')
    ap.add_argument('--parquet', default='full_run.parquet')
    args = ap.parse_args()

    reg = os.path.join(REPO_ROOT, 'regulations', args.regulation)
    os.chdir(reg)
    config = yaml.safe_load(open('analyzer_config.yaml'))
    spec = field_spec(config, args.field)
    tool = build_tool(spec)
    sysmsg = system_prompt(config, spec)
    model = args.model.split('/', 1)[-1]

    df = pd.read_parquet(args.parquet)
    todo = [(i, r) for i, r in df.iterrows()
            if isinstance(r['analysis'], dict) and args.field not in r['analysis']]
    print(f"{len(df):,} rows; {len(todo):,} missing `{args.field}`")
    if not todo:
        print("Nothing to do.")
        return

    c = client()
    if args.estimate:
        sample = todo[:20]
        tot = 0
        for _, r in sample:
            tot += c.messages.count_tokens(
                model=model, system=sysmsg, tools=[tool],
                messages=[{'role': 'user', 'content': f"Comment:\n\n{(r['text'] or '')[:6000]}"}]
            ).input_tokens
        mean = tot / len(sample)
        # Sonnet 5 rates; this schema's output is one short array.
        cost = (mean * len(todo) / 1e6 * 2.0) + (60 * len(todo) / 1e6 * 10.0)
        print(f"  mean input {mean:,.0f} tok (measured on {len(sample)})")
        print(f"  estimated  ${cost:,.2f}")
        return

    def one(item):
        i, r = item
        try:
            msg = c.messages.create(
                model=model, max_tokens=400, system=sysmsg, tools=[tool],
                tool_choice={'type': 'tool', 'name': tool['name']},
                messages=[{'role': 'user', 'content': f"Comment:\n\n{(r['text'] or '')[:6000]}"}])
            blk = next(b for b in msg.content if b.type == 'tool_use')
            return i, blk.input.get(args.field), msg.usage, None
        except Exception as e:
            return i, None, None, str(e)

    # Persist answers as they arrive, keyed by comment id. The merge and the
    # parquet write happen after every call has been paid for, so a failure
    # there must not cost the run.
    ckpt = f'.{args.field}_backfill.json'
    prior = {}
    if os.path.exists(ckpt):
        try:
            prior = json.load(open(ckpt))
            print(f"resuming: {len(prior):,} answers already recorded in {ckpt}")
        except (json.JSONDecodeError, OSError):
            prior = {}
    todo = [(i, r) for i, r in todo if str(r['id']) not in prior]
    print(f"{len(todo):,} still to call")

    results, errors, uin, uout = {}, [], 0, 0
    ck_lock = threading.Lock()
    # seed from the checkpoint by row index
    id_to_idx = {str(r['id']): i for i, r in df.iterrows()}
    for cid, val in prior.items():
        if cid in id_to_idx:
            results[id_to_idx[cid]] = val

    with cf.ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(one, t) for t in todo]
        for n, f in enumerate(cf.as_completed(futs), 1):
            i, val, usage, err = f.result()
            if err:
                errors.append((i, err))
            else:
                results[i] = val
                uin += usage.input_tokens
                uout += usage.output_tokens
                with ck_lock:
                    prior[str(df.iloc[i]['id'])] = val
                    if len(prior) % 50 == 0:
                        json.dump(prior, open(ckpt, 'w'))
            if n % 200 == 0:
                print(f"  {n}/{len(todo)}", flush=True)

    # Merge into the existing analysis dicts, leaving every other key untouched.
    #
    # Every row gets the key, and a multi_enum always gets a list -- never None.
    # Arrow types the `analysis` struct once for the whole column, so a single
    # None where the other rows hold a list fails the write with "cannot mix list
    # and non-list values" AFTER all the model calls have been paid for.
    if not results:
        print("\nNo call succeeded — leaving the parquet untouched.")
        if errors:
            print(f"{len(errors):,} error(s); first: {errors[0][1][:200]}")
        raise SystemExit(1)

    ftype = spec.get('type', 'text')
    def normalise(v):
        if ftype == 'multi_enum':
            return [str(x) for x in v] if isinstance(v, (list, tuple)) else []
        return '' if v is None else str(v)

    # Only rows with an actual answer get the key. Writing an empty value for a
    # row whose call failed produces a result indistinguishable from a genuine
    # "none of these apply" -- the row then looks done, is skipped on the next
    # run, and the absence is reported downstream as a finding.
    #
    # Arrow types the struct once per column, so every row that DOES get the key
    # must get the same shape; rows without it are left alone and Arrow nulls
    # them, which is recoverable because null is distinguishable from empty.
    col = list(df['analysis'])
    filled = 0
    for i, val in results.items():
        if not isinstance(col[i], dict):
            continue
        a = dict(col[i])
        a[args.field] = normalise(val)
        col[i] = a
        filled += 1
    df['analysis'] = col

    if errors:
        print(f"\n{len(errors):,} row(s) failed and were left without the field; "
              f"re-run to pick them up.")

    # Write beside the target and rename, so a failed write cannot leave a
    # half-written parquet where a complete one used to be.
    tmp = args.parquet + '.tmp'
    df.to_parquet(tmp, index=False)
    os.replace(tmp, args.parquet)

    json.dump(prior, open(ckpt, 'w'))

    cost = uin / 1e6 * 2.0 + uout / 1e6 * 10.0
    print(f"\nfilled {filled:,}, errors {len(errors)}")
    print(f"tokens {uin:,} in / {uout:,} out   ~${cost:,.2f}")
    if errors:
        for i, e in errors[:3]:
            print("  ERR", df.iloc[i]['id'], str(e)[:110])


if __name__ == '__main__':
    main()
