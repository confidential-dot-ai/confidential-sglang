"""Read the scorecard's server counters from Prometheus instead of the workers.

On a cluster the workers' /metrics are not reachable from outside (exec into
pods is off, and metrics leave only through the collector's remote write), so
the scorecard reads the same series from the Prometheus they are written to.
Each role is a PromQL label matcher list that selects its pods, for example

  --prom-prefill 'environment="candidate",job="inference-workers",role="prefill"'

A snapshot taken at time T waits --prom-lag-s (remote write and scrape delay),
then queries every sglang:* and smg_* series of each role at T, and builds the
same Snapshot live_metrics builds from /metrics text, so the window and gate
logic are unchanged. A window must not span a worker restart: the old pod's
series stay visible for Prometheus's lookback (5 min) and would be summed.
"""

from __future__ import annotations

import json
import os
import time
import urllib.parse
import urllib.request

import live_metrics as lm

SERIES = "sglang:.*|smg_.*"


def _escape(v: str) -> str:
    return v.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")


def exposition(result: list) -> str:
    """Prometheus API vector samples as /metrics text lines."""
    lines = []
    for item in result:
        labels = dict(item["metric"])
        name = labels.pop("__name__")
        body = ",".join(f'{k}="{_escape(v)}"' for k, v in sorted(labels.items()))
        lines.append(f"{name}{{{body}}} {item['value'][1]}")
    return "\n".join(lines)


class PromScraper:
    def __init__(self, url, prefill="", decode="", worker="", router="", token_env="", lag_s=60.0, timeout=30.0):
        if worker and (prefill or decode):
            raise ValueError("--prom-worker (colocated) cannot be combined with --prom-prefill/--prom-decode")
        if not worker and not (prefill and decode):
            raise ValueError("pass --prom-prefill and --prom-decode (PD) or --prom-worker (colocated)")
        self.url, self.lag_s, self.timeout = url.rstrip("/"), lag_s, timeout
        self.prefill, self.decode = prefill or worker, decode or worker
        self.colocated = bool(worker)
        self.router = router
        self.token = os.environ.get(token_env, "") if token_env else ""

    def _query(self, matchers: str, at: float) -> lm.Scrape:
        sel = f'{{__name__=~"{SERIES}"' + (f",{matchers}" if matchers else "") + "}"
        q = urllib.parse.urlencode({"query": sel, "time": f"{at:.3f}"})
        req = urllib.request.Request(f"{self.url}/api/v1/query?{q}")
        if self.token:
            req.add_header("authorization", f"Bearer {self.token}")
        with urllib.request.urlopen(req, timeout=self.timeout) as r:
            body = json.loads(r.read())
        if body.get("status") != "success":
            raise OSError(f"prometheus query failed: {body.get('error', body)}")
        return lm.parse(exposition(body["data"]["result"]))

    def snapshot(self) -> lm.Snapshot:
        t = time.time()
        time.sleep(self.lag_s)
        return self.snapshot_at(t)

    def snapshot_at(self, t: float) -> lm.Snapshot:
        """The series at a past time t (the caller has waited the lag since t)."""
        p = self._query(self.prefill, t)
        d = p if self.colocated else self._query(self.decode, t)
        r = self._query(self.router, t) if self.router else None
        return lm.Snapshot(t, p, d, r)


def add_args(p):
    p.add_argument("--prom-url", default="", help="read server counters from this Prometheus (query API base URL)")
    p.add_argument("--prom-prefill", default="", help="label matchers selecting the prefill workers")
    p.add_argument("--prom-decode", default="", help="label matchers selecting the decode workers")
    p.add_argument("--prom-worker", default="", help="label matchers selecting colocated (non-PD) workers")
    p.add_argument("--prom-router", default="", help="label matchers selecting the SGLang router")
    p.add_argument("--prom-token-env", default="", help="environment variable with a bearer token for Prometheus")
    p.add_argument("--prom-lag-s", type=float, default=60, help="remote-write delay to wait before each query")


def from_args(args) -> PromScraper:
    return PromScraper(
        args.prom_url, prefill=args.prom_prefill, decode=args.prom_decode, worker=args.prom_worker,
        router=args.prom_router, token_env=args.prom_token_env, lag_s=args.prom_lag_s,
    )
