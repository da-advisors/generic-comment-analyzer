#!/usr/bin/env python3
"""OCR attachments that the local extraction path could not read.

`attachment_utils.process_attachments` extracts text from PDFs with PyMuPDF and
falls back to an OpenAI vision model for scanned pages. Without an OpenAI key
that fallback is unavailable, and a scanned submission ends up analyzed on its
body text alone -- for a "See attached file(s)" comment, that means analyzed on
nothing. This fills the gap with Claude, which reads PDFs and images natively.

Writes the same `<file>.extracted.txt` sidecars the pipeline already consults,
so a subsequent pipeline run picks the text up with no further change. Because
that changes the comment's text, its analysis is correctly re-run: the text key
the reuse cache is built on no longer matches.

Usage:
    python ocr_attachments.py --regulation census-2030-residence          # all unreadable
    python ocr_attachments.py --regulation census-2030-residence --dry-run
"""
import argparse
import base64
import glob
import mimetypes
import os
import re
import sys

import anthropic
from dotenv import load_dotenv

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(REPO_ROOT, '.env'))

MODEL = os.getenv('OCR_MODEL', 'claude-sonnet-5')
MIN_USEFUL_CHARS = 200

PROMPT = """Transcribe all text in this document, in reading order.

Rules:
- Output the text only. No preamble, no commentary, no markdown fences.
- Preserve paragraph breaks. Keep headings, letterhead, signature blocks, and
  lists as they appear.
- For a table, write it as readable rows; do not invent structure that is not there.
- If a page is a photograph, scan or screenshot of a letter, transcribe the letter.
- Describe an image only when it carries meaning no text conveys, and mark it
  [image: ...].
- If a page is genuinely blank or illegible, write [illegible page] and continue.
- Do not summarize, correct, complete or translate anything. Transcribe verbatim,
  including typos."""


def is_pdf(p):
    return p.lower().endswith('.pdf')


def is_image(p):
    return p.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.webp'))


def attachment_groups(comment_dir):
    """Group files by the source attachment they came from.

    regulations.gov often serves the same attachment twice, once as a PDF and
    once as an image (`attachment_1_attachment_2.pdf`, `attachment_2_attachment_2.jpg`).
    The trailing `attachment_N` is the source document; transcribing both halves
    would duplicate the text and double the cost, so one file per group is read
    and the PDF is preferred because it carries all its pages.
    """
    groups = {}
    for p in sorted(glob.glob(os.path.join(comment_dir, '*'))):
        if p.endswith('.extracted.txt') or not (is_pdf(p) or is_image(p)):
            continue
        m = re.search(r'(attachment_\d+)(?:\.[^.]+)?$', os.path.basename(p))
        groups.setdefault(m.group(1) if m else os.path.basename(p), []).append(p)
    return {k: sorted(v, key=lambda p: (not is_pdf(p), p)) for k, v in groups.items()}


def existing_text(comment_dir):
    total = 0
    for f in glob.glob(os.path.join(comment_dir, '*.extracted.txt')):
        try:
            total += len(open(f, encoding='utf-8', errors='replace').read().strip())
        except OSError:
            pass
    return total


def transcribe(client, path):
    data = base64.standard_b64encode(open(path, 'rb').read()).decode()
    if is_pdf(path):
        block = {'type': 'document',
                 'source': {'type': 'base64', 'media_type': 'application/pdf', 'data': data}}
    else:
        media = mimetypes.guess_type(path)[0] or 'image/jpeg'
        block = {'type': 'image',
                 'source': {'type': 'base64', 'media_type': media, 'data': data}}

    # Streaming: a long scanned submission can run past the non-streaming timeout.
    with client.messages.stream(
        model=MODEL, max_tokens=16000,
        messages=[{'role': 'user', 'content': [block, {'type': 'text', 'text': PROMPT}]}],
    ) as stream:
        msg = stream.get_final_message()

    text = ''.join(b.text for b in msg.content if b.type == 'text').strip()
    return text, msg.usage


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--regulation', required=True)
    ap.add_argument('--dry-run', action='store_true', help='List what would be read, read nothing.')
    ap.add_argument('--only', nargs='*', help='Limit to these comment IDs.')
    ap.add_argument('--force', action='store_true',
                    help='Re-OCR even where usable text already exists.')
    args = ap.parse_args()

    reg = os.path.join(REPO_ROOT, 'regulations', args.regulation)
    att_root = os.path.join(reg, 'attachments')
    if not os.path.isdir(att_root):
        sys.exit(f"No attachments directory at {att_root}")

    targets = []
    for cid in sorted(os.listdir(att_root)):
        d = os.path.join(att_root, cid)
        if not os.path.isdir(d):
            continue
        if args.only and cid not in args.only:
            continue
        if not args.force and existing_text(d) >= MIN_USEFUL_CHARS:
            continue
        groups = attachment_groups(d)
        if groups:
            targets.append((cid, groups))

    if not targets:
        print("Nothing to OCR — every attachment already has usable text.")
        return

    print(f"{len(targets)} comment(s) with unreadable attachments:")
    for cid, groups in targets:
        picks = [os.path.basename(v[0]) for v in groups.values()]
        skipped = sum(len(v) - 1 for v in groups.values())
        print(f"  {cid}: {len(groups)} document(s) -> {', '.join(picks)}"
              + (f"   (+{skipped} duplicate format(s) skipped)" if skipped else ""))
    if args.dry_run:
        return

    client = anthropic.Anthropic()
    tot_in = tot_out = 0
    for cid, groups in targets:
        print(f"\n{cid}")
        for name, files in groups.items():
            path = files[0]
            try:
                text, usage = transcribe(client, path)
            except Exception as e:
                print(f"  {os.path.basename(path)}: FAILED — {e}")
                continue
            tot_in += usage.input_tokens
            tot_out += usage.output_tokens
            if len(text) < 20:
                print(f"  {os.path.basename(path)}: no text recovered")
                continue
            out = path + '.extracted.txt'
            with open(out, 'w', encoding='utf-8') as f:
                f.write(text)
            print(f"  {os.path.basename(path)}: {len(text):,} chars -> {os.path.basename(out)}")

    # Sonnet 5 rates; a PDF page bills as image tokens, hence the large input side.
    cost = tot_in / 1e6 * 2.0 + tot_out / 1e6 * 10.0
    print(f"\ntokens: {tot_in:,} in / {tot_out:,} out   ~${cost:,.2f}")
    print(f"\nRe-run the pipeline to pick the new text up:\n"
          f"  python pipeline.py --regulation {args.regulation}")


if __name__ == '__main__':
    main()
