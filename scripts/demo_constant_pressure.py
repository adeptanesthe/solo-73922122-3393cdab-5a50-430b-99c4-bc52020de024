#!/usr/bin/env python3
"""API demo: successive zero-point corrections make one fold's pressure constant.

Runs against a live service (default http://localhost:8000) using only the
standard library.  It onboards the four demo factories, then walks every
version's per-fold diagnostics and performs correction updates until the
fold holding F3 out trains on a pool whose *corrected* pressure is a
single constant -- that fold then drops the pressure feature on its own,
while every other fold keeps both features.

Steps:
  1. POST /versions                -> version 1 (F1/F2/F4 share one constant
                                       raw pressure 500000 but carry different
                                       corrections; F3 pressure varies)
  2. POST /versions/1/correction   -> F3 0 -> 500 (fixes its zero-point drift)
  3. POST /versions/2/correction   -> F2 -40 -> 0
  4. POST /versions/3/correction   -> F4 10 -> 0
     Now F1/F2/F4 corrected pressures are all 500000: in the fold that
     holds F3 out, pressure has zero pooled variance and is ignored in
     that fold only.  Each response carries the summary-op counts.
"""
import json
import os
import sys
import urllib.error
import urllib.request

BASE = os.environ.get("BASE_URL", "http://localhost:8000")
HERE = os.path.dirname(os.path.abspath(__file__))


def call(method, path, payload=None):
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(
        BASE + path, data=data, method=method,
        headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            return resp.status, json.loads(resp.read())
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read())


def show_folds(view):
    t = view["training"]
    print(f"  version {view['version']}: summary_ops={t['summary_ops']} "
          f"(legacy re-merge would be {t['legacy_summary_ops']}), "
          f"factorizations={t['matrix_factorizations']}")
    for f in t["folds"]:
        tag = ("DROPS " + ",".join(f["dropped_features"])) \
            if f["dropped_features"] else "keeps speed+pressure"
        print(f"    fold hold-out {f['factory_id']}: {tag}")


def main():
    with open(os.path.join(HERE, "..", "demo", "create_version.json")) as fh:
        onboard = json.load(fh)

    print("=== 1. onboard 4 factories: POST /versions")
    status, view = call("POST", "/versions", onboard)
    assert status == 201, view
    show_folds(view)

    # F3 drift fix first (the main zero-point correction story).
    print("=== 2. correct F3 zero-point drift 0 -> 500")
    status, view = call("POST", f"/versions/{view['version']}/correction",
                        {"factory_id": "F3", "correction": 500})
    assert status == 200, view
    show_folds(view)

    # Align the three constant-raw-pressure factories onto one corrected value.
    for fid, new_corr in (("F2", 0), ("F4", 0)):
        print(f"=== correction: {fid} -> {new_corr}")
        status, view = call("POST", f"/versions/{view['version']}/correction",
                            {"factory_id": fid, "correction": new_corr})
        assert status == 200, view
        show_folds(view)

    folds = {f["factory_id"]: f for f in view["training"]["folds"]}
    assert folds["F3"]["dropped_features"] == ["pressure"], folds["F3"]
    assert all(folds[f]["dropped_features"] == []
               for f in ("F1", "F2", "F4"))
    print("OK: only the fold holding the varying-pressure factory F3 out now "
          "sees constant corrected pressure and ignores that feature.")
    print("final corrections:", view["corrections"])


if __name__ == "__main__":
    try:
        main()
    except AssertionError as exc:
        print("demo failed:", exc, file=sys.stderr)
        sys.exit(1)
