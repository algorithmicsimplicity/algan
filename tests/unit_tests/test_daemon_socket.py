r"""The daemon's trigger socket: who may poke it, and where it can be found.

Three things are pinned here, all of them ways a daemon used to be reachable
by the wrong process or unreachable by the right one:

* every verb needs the token from the state file (a bare ``quit\\n`` from any
  local process used to stop somebody else's daemon);
* a held port is not fatal -- the daemon binds an ephemeral one and publishes
  it, instead of exiting and leaving every later run to spawn another daemon
  that fails the same way;
* the state file says which Algan the daemon *is*, since ``$ALGAN_HOME``
  registers one daemon for every virtualenv on the machine.
"""

from __future__ import annotations

import os
import queue
import socket
import threading

import pytest

from algan import daemon as d
from algan import daemon_client as dc


def _free_port():
    """A port nothing is listening on (bound and released)."""
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    sock.close()
    return port


@pytest.fixture
def state(tmp_path, monkeypatch):
    monkeypatch.setenv("ALGAN_HOME", str(tmp_path))
    return d._StateFile(_free_port())


@pytest.fixture
def served(state):
    """A live trigger socket, with the queue its verbs land in."""
    events = queue.Queue()
    server = d._start_socket(
        events, state._payload["port"], state, threading.Event(), d._SourceDigest({})
    )
    assert server is not None
    try:
        yield events, state
    finally:
        server.shutdown()
        server.server_close()


def _send(port, line):
    with socket.create_connection(("127.0.0.1", port), 5) as sock:
        sock.sendall(line.encode() + b"\n")
        return sock.recv(256).decode("utf-8", "replace").strip()


# --------------------------------------------------------------------------
# The token
# --------------------------------------------------------------------------


@pytest.mark.parametrize("verb", ["render", "ping", "quit"])
def test_every_verb_needs_the_token(served, verb):
    events, state = served
    assert "bad token" in _send(state._payload["port"], verb)
    assert events.empty(), "an untokened verb must not reach the daemon"


@pytest.mark.parametrize(("verb", "expected"), [("render", "render"), ("quit", "quit")])
def test_a_tokened_verb_is_queued(served, verb, expected):
    events, state = served
    assert _send(state._payload["port"], f"{verb} {state.token}") == "ok"
    assert events.get_nowait()[0] == expected


def test_ping_answers_with_the_token(served):
    events, state = served
    assert _send(state._payload["port"], f"ping {state.token}") == "pong"
    assert events.empty()


def test_a_wrong_token_is_refused(served):
    events, state = served
    assert "bad token" in _send(state._payload["port"], "quit not-the-token")
    assert events.empty()


def test_an_unknown_verb_still_says_what_is_accepted(served):
    _, state = served
    assert "expected run" in _send(state._payload["port"], f"frobnicate {state.token}")


# --------------------------------------------------------------------------
# The port
# --------------------------------------------------------------------------


def test_a_held_port_falls_back_to_an_ephemeral_one(state):
    """The bug: the daemon exited, so every later run spawned another one."""
    holder = socket.socket()
    holder.bind(("127.0.0.1", 0))
    holder.listen(1)
    held = holder.getsockname()[1]
    events = queue.Queue()
    try:
        server = d._start_socket(
            events, held, state, threading.Event(), d._SourceDigest({})
        )
        assert server is not None, "a taken port must not stop the daemon"
        try:
            bound = server.server_address[1]
            assert bound != held
            # The state file is how clients find it, so it must carry the port
            # actually bound rather than the one that was wanted.
            assert state._payload["port"] == bound
            assert _send(bound, f"ping {state.token}") == "pong"
        finally:
            server.shutdown()
            server.server_close()
    finally:
        holder.close()


def test_an_explicit_port_is_allowed_to_fail(state):
    """``--port`` is an instruction, not a preference: it fails loudly."""
    holder = socket.socket()
    holder.bind(("127.0.0.1", 0))
    holder.listen(1)
    held = holder.getsockname()[1]
    try:
        server = d._start_socket(
            queue.Queue(),
            held,
            state,
            threading.Event(),
            d._SourceDigest({}),
            allow_fallback=False,
        )
        assert server is None
    finally:
        holder.close()


# --------------------------------------------------------------------------
# The state file
# --------------------------------------------------------------------------


def test_the_state_file_says_which_algan_this_is(state):
    payload = state._payload
    identity = dc.interpreter_identity()
    for field in ("python", "prefix", "algan_path", "algan_version"):
        assert payload[field] == identity[field]
    assert dc.describe_interpreter_mismatch(payload) is None


def test_the_state_file_is_written_owner_only(state):
    state.write()
    assert os.path.isfile(state.path)
    if os.name != "nt":
        assert os.stat(state.path).st_mode & 0o777 == 0o600


# --------------------------------------------------------------------------
# Tracebacks
# --------------------------------------------------------------------------


def test_the_daemons_own_frames_are_stripped_from_a_script_traceback(tmp_path):
    """A script's error must read as it would in its own process."""
    import runpy

    script = tmp_path / "boom.py"
    script.write_text("raise ValueError('boom')\n", encoding="utf-8")
    with pytest.raises(ValueError) as raised:
        runpy.run_path(str(script), run_name="__main__")
    # In the daemon the top frame is ``execute`` in daemon.py, which is
    # plumbing; here it is pytest's own call, which is not. Start one frame in,
    # where the daemon's own frames end and runpy's begin.
    stripped = d.strip_plumbing_frames(raised.tb.tb_next)
    frames = []
    while stripped is not None:
        frames.append(stripped.tb_frame.f_code.co_filename)
        stripped = stripped.tb_next
    assert frames, "the script's own frame must survive"
    assert frames[0] == str(script)
    assert not any("runpy" in name for name in frames)
    assert d._is_plumbing_frame(d.__file__), "the daemon's own frames go too"


def test_an_all_plumbing_traceback_is_left_alone():
    """An empty traceback would be worse than an honest one."""
    with pytest.raises(ValueError) as raised:
        raise ValueError("from the daemon itself")
    tb = raised.value.__traceback__
    assert d.strip_plumbing_frames(tb) is tb
