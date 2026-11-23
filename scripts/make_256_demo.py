#!/usr/bin/env python3
"""Generate a deterministic 256-factory onboarding payload.

Writes demo/create_version_256.json -- a valid POST /versions body with
the new upper bound of 256 factories.  Each factory reports only the
integer sufficient statistics (sample count, vector sums, outer-product
sums); one in 32 factories carries a non-zero zero-point correction.

Also prints the exact summary-operation counts the training will report:
2K-1 = 511 exact-integer operations (K-1 merges into the total plus K
leave-one-fold subtractions), versus K(K-1) = 65280 for the old
per-fold re-merge approach.
"""
import json
import os
import random

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "demo")

K = 256
A, B, INTERCEPT, NOISE = 0.020, 0.011, 1500.0, 1.0


def simulate(fid, n, s_mu, p_mu, correction, seed):
    rng = random.Random(seed)
    rows = []
    for _ in range(n):
        s = int(round(rng.gauss(s_mu, 25)))
        p = int(round(rng.gauss(p_mu, 12)))
        y = (A * s + B * (p + correction) + INTERCEPT
             + rng.gauss(0, NOISE))
        rows.append((s, p, int(round(y))))
    sums = [sum(r[i] for r in rows) for i in range(3)]
    outer = [[sum(r[i] * r[j] for r in rows) for j in range(3)]
             for i in range(3)]
    return {
        "factory_id": fid,
        "n": n,
        "sums": sums,
        "outer": outer,
        "correction": correction,
    }


def main():
    os.makedirs(OUT, exist_ok=True)
    factories = []
    for i in range(K):
        fid = f"F{i+1:03d}"
        correction = 40 if i % 32 == 7 else 0
        body = simulate(
            fid,
            n=120 + (i % 5) * 24,                 # 120..216 samples
            s_mu=1_000_000 + 40 * (i % 64),
            p_mu=500_000 + 10 * (i % 32),
            correction=correction,
            seed=900_000 + i,
        )
        factories.append(body)

    path = os.path.join(OUT, "create_version_256.json")
    with open(path, "w") as fh:
        json.dump({"factories": factories}, fh)
    size_kb = os.path.getsize(path) / 1024
    print(f"wrote {path} ({K} factories, {size_kb:.0f} KiB)")
    print(f"summary ops at train time: {2 * K - 1} "
          f"(K-1 merges + K fold subtractions) vs legacy {K * (K - 1)}")
    print("matrix factorisations: at most", K,
          "(one per fold, reused for all three lambdas)")


if __name__ == "__main__":
    main()
