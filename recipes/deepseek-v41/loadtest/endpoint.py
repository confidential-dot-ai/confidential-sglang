"""The endpoint every stress tool talks to: a local router, or a cluster's gateway.

A cluster gateway needs an API key. It is read from an environment variable
(--api-key-env, default STRESS_API_KEY) and never from the command line, so it
stays out of shell history, `ps` and the result files. --ca-cert trusts a
private CA (for example a TEErminator or C8s router certificate).
"""

from __future__ import annotations

import json
import os
import resource
import ssl
import urllib.request

DEFAULT_KEY_ENV = "STRESS_API_KEY"


def add_endpoint_args(p, url_default="http://127.0.0.1:8040", model_default=""):
    p.add_argument("--url", default=url_default, help="base URL: router, or the cluster gateway")
    p.add_argument("--model", default=model_default, help="model id (default: the first one /v1/models lists)")
    p.add_argument(
        "--api-key-env",
        default=DEFAULT_KEY_ENV,
        help=f"environment variable holding the gateway API key (default {DEFAULT_KEY_ENV}; unset = no key)",
    )
    p.add_argument("--ca-cert", default="", help="PEM bundle to trust for an https --url")


def api_key(args) -> str:
    return os.environ.get(args.api_key_env or DEFAULT_KEY_ENV, "")


def headers(args) -> dict:
    h = {"content-type": "application/json"}
    key = api_key(args)
    if key:
        h["authorization"] = f"Bearer {key}"
    return h


def ssl_context(args):
    """None for plain http; for https the system trust store plus --ca-cert."""
    if not args.url.startswith("https://"):
        return None
    ctx = ssl.create_default_context()
    if args.ca_cert:
        ctx.load_verify_locations(cafile=args.ca_cert)
    return ctx


def client_session(args, limit=0, total_timeout=900.0):
    import aiohttp

    conn = aiohttp.TCPConnector(limit=limit, ssl=ssl_context(args) or True)
    return aiohttp.ClientSession(
        connector=conn,
        headers=headers(args),
        timeout=aiohttp.ClientTimeout(total=total_timeout),
    )


def get_json(args, path, timeout=30.0):
    req = urllib.request.Request(args.url.rstrip("/") + path, headers=headers(args))
    with urllib.request.urlopen(req, timeout=timeout, context=ssl_context(args)) as r:
        return json.loads(r.read())


def resolve_model(args) -> str:
    """Fill args.model from /v1/models when it was not given."""
    if not args.model:
        args.model = get_json(args, "/v1/models")["data"][0]["id"]
    return args.model


def raise_nofile(needed: int) -> int:
    """Raise this process's open-file soft limit to the hard limit and return it.

    Each in-flight request holds a socket. Containers often start with a soft
    limit of 1024, and past it new connections fail in the client with EMFILE,
    which would look like the server refusing them."""
    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    want = hard if hard != resource.RLIM_INFINITY else max(soft, needed + 256)
    if want > soft:
        try:
            resource.setrlimit(resource.RLIMIT_NOFILE, (want, hard))
            soft = want
        except (ValueError, OSError):
            pass
    if soft < needed + 64:
        print(f"warning: open-file limit {soft} < {needed} concurrent requests; raise `ulimit -n`", flush=True)
    return soft


def describe(args) -> dict:
    """What a result file records about the endpoint (never the key)."""
    return {"url": args.url, "model": args.model, "api_key": bool(api_key(args)), "ca_cert": args.ca_cert or None}
