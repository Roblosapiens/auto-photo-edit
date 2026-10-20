"""Keep the photo-processing pipeline offline.

Import and call enable() before importing torch / open_clip / ultralytics / geocalib (every entry
script in src/ does). It works per process - nothing system-wide is changed:

1. Library switches: Hugging Face Hub offline (no update checks, no telemetry), Ultralytics offline (no
   DNS "is online" probe, which also disables its anonymous analytics and any automatic downloads).
2. A network guard: any outbound connection or DNS lookup from any library raises OSError and is
   recorded in `attempts`, so a dependency we missed cannot phone home silently.

Network access is only allowed when explicitly requested, e.g. tools/update_models.py, by setting
AUTOPHOTOEDIT_ALLOW_NETWORK=1. AUTOPHOTOEDIT_NETWORK_AUDIT=1 records attempts without blocking them
(used by tests/test_offline.py to show what would otherwise connect).
"""
import os
import socket
import sys

ALLOW_ENV = "AUTOPHOTOEDIT_ALLOW_NETWORK"
AUDIT_ENV = "AUTOPHOTOEDIT_NETWORK_AUDIT"
OFFLINE_ENV = {
    "HF_HUB_OFFLINE": "1",            # huggingface_hub (open_clip, timm): cache only, no update checks
    "TRANSFORMERS_OFFLINE": "1",
    "HF_HUB_DISABLE_TELEMETRY": "1",
    "YOLO_OFFLINE": "1",              # ultralytics: no online probe -> no analytics, no auto-downloads
    "DO_NOT_TRACK": "1",              # honoured by some tools as a generic opt-out
}
attempts = []  # (kind, target) for every blocked or audited network attempt
_enabled = False


def _record(kind, target, block):
    attempts.append((kind, str(target)[:120]))
    print(f"[offline] {'blocked' if block else 'audited'} {kind}: {str(target)[:120]}", file=sys.stderr)
    if block:
        raise OSError(f"network access disabled (offline mode): {kind} {target}")


def _guard(block):
    orig_getaddrinfo, orig_connect, orig_connect_ex = socket.getaddrinfo, socket.socket.connect, socket.socket.connect_ex
    orig_create_connection = socket.create_connection

    def getaddrinfo(host, *a, **k):
        if host not in (None, "localhost", "127.0.0.1", "::1"):
            _record("DNS lookup", host, block)
        return orig_getaddrinfo(host, *a, **k)

    def connect(self, address):
        if not _is_local(address):
            _record("connect", address, block)
        return orig_connect(self, address)

    def connect_ex(self, address):
        if not _is_local(address):
            _record("connect", address, block)
        return orig_connect_ex(self, address)

    def create_connection(address, *a, **k):
        if not _is_local(address):
            _record("connect", address, block)
        return orig_create_connection(address, *a, **k)

    socket.getaddrinfo = getaddrinfo
    socket.socket.connect = connect
    socket.socket.connect_ex = connect_ex
    socket.create_connection = create_connection


def _is_local(address):
    host = address[0] if isinstance(address, tuple) and address else address
    return host in ("localhost", "127.0.0.1", "::1") or not isinstance(address, tuple)


def enable():
    """Switch this process to offline mode (idempotent). Returns True if offline mode is active."""
    global _enabled
    if _enabled:
        return True
    if os.environ.get(ALLOW_ENV) == "1":
        return False
    if os.environ.get(AUDIT_ENV) == "1":  # observe the libraries' normal behaviour: no switches, no blocking
        _guard(block=False)
        _enabled = True
        return False
    for k, v in OFFLINE_ENV.items():
        os.environ[k] = v
    _guard(block=True)
    _enabled = True
    return True
