#!/usr/bin/env python3
"""Compare two bench_batch_quality runs (fp16 vs fp8) per-request.

Reads the [REQ i] lines from two output files, verifies each request used the
same input prompt (psum), and reports:
  - first-token agreement rate
  - position-wise token match rate over the generated sequence
  - mean divergence position
  - full-sequence exact-match count

Usage: compare_fp8_runs.py <fp16.txt> <fp8.txt> [max_new]
"""
import re
import sys

REQ_RE = re.compile(
    r"\[REQ (\d+)\] psum=(\d+) plen=(\d+) ttft=([\d.]+) tpot=([\d.]+) tok=(.*)")


def parse(path):
    out = {}
    for line in open(path, encoding="utf-8", errors="replace"):
        m = REQ_RE.match(line.strip())
        if not m:
            continue
        i = int(m.group(1))
        out[i] = {
            "psum": int(m.group(2)),
            "plen": int(m.group(3)),
            "toks": [int(x) for x in m.group(6).split()],
        }
    return out


def main():
    if len(sys.argv) < 3:
        print("usage: compare_fp8_runs.py <fp16.txt> <fp8.txt> [max_new]")
        return 1
    a = parse(sys.argv[1])
    b = parse(sys.argv[2])
    max_new = int(sys.argv[3]) if len(sys.argv) > 3 else 0

    common = sorted(set(a) & set(b))
    if not common:
        print("no matching [REQ] lines between the two files")
        return 1

    print(f"# requests compared: {len(common)}")
    n_first_agree = 0
    n_exact = 0
    pos_matches = 0
    pos_total = 0
    div_sum = 0
    n_div = 0
    psum_bad = 0
    max_out = 0

    for i in common:
        pa, pb = a[i], b[i]
        if pa["psum"] != pb["psum"]:
            psum_bad += 1
            continue
        ta, tb = pa["toks"], pb["toks"]
        L = min(len(ta), len(tb))
        max_out = max(max_out, len(ta), len(tb))
        if ta and tb and ta[0] == tb[0]:
            n_first_agree += 1
        if ta == tb:
            n_exact += 1
        d = 0
        while d < L and ta[d] == tb[d]:
            d += 1
        pos_matches += d
        pos_total += L
        if d < L:
            div_sum += d
            n_div += 1

    n = len(common)
    n_psum_ok = n - psum_bad
    print(f"# input-set (psum) mismatch : {psum_bad}/{n}  (0 = same inputs verified)")
    print(f"# first-token agreement     : {n_first_agree}/{n_psum_ok}  "
          f"({100.0*n_first_agree/n_psum_ok:.1f}%)")
    if pos_total:
        print(f"# position-wise match rate   : {pos_matches}/{pos_total}  "
              f"({100.0*pos_matches/pos_total:.1f}%)")
    if n_div:
        print(f"# mean divergence position   : {div_sum/n_div:.1f} "
              f"(of max_new={max_out if max_new else max_out})")
    print(f"# full-sequence exact match  : {n_exact}/{n_psum_ok}  "
          f"({100.0*n_exact/n_psum_ok:.1f}%)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
