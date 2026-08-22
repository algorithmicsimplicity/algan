"""Minimal OpenAI-compatible SSE endpoint that drives one tool-call round trip.

Turn 1 (no tool result in history) -> emit a tool_calls delta for the client's
own `bash` tool. Turn 2 (a role:"tool" message present) -> emit plain text.
That exercises the full OpenCode loop with no external credential.
"""

import json
import os
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

LOG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "requests.jsonl")


def sse(obj):
    return f"data: {json.dumps(obj)}\n\n".encode()


def chunk(delta, finish=None):
    return {
        "id": "chatcmpl-mock",
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": "mock-1",
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
    }


class H(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def do_POST(self):
        body = self.rfile.read(int(self.headers.get("content-length", 0)))
        req = json.loads(body or b"{}")

        with open(LOG, "a") as f:
            f.write(
                json.dumps(
                    {
                        "path": self.path,
                        "tools": [
                            t.get("function", {}).get("name")
                            for t in req.get("tools", [])
                        ],
                        "roles": [m.get("role") for m in req.get("messages", [])],
                    }
                )
                + "\n"
            )

        saw_tool_result = any(m.get("role") == "tool" for m in req.get("messages", []))
        names = [t.get("function", {}).get("name") for t in req.get("tools", [])]
        tool = "bash" if "bash" in names else (names[0] if names else "bash")

        self.send_response(200)
        self.send_header("content-type", "text/event-stream")
        self.send_header("cache-control", "no-cache")
        self.end_headers()

        if saw_tool_result:
            self.wfile.write(sse(chunk({"role": "assistant", "content": ""})))
            self.wfile.write(
                sse(
                    chunk(
                        {"content": "LOOP_VERIFIED: tool ran and its result came back."}
                    )
                )
            )
            self.wfile.write(sse(chunk({}, "stop")))
        else:
            self.wfile.write(
                sse(
                    chunk(
                        {
                            "role": "assistant",
                            "tool_calls": [
                                {
                                    "index": 0,
                                    "id": "call_1",
                                    "type": "function",
                                    "function": {
                                        "name": tool,
                                        "arguments": json.dumps(
                                            {
                                                "command": "echo OPENCODE_TOOL_EXECUTED",
                                                "description": "loop probe",
                                            }
                                        ),
                                    },
                                }
                            ],
                        }
                    )
                )
            )
            self.wfile.write(sse(chunk({}, "tool_calls")))
        self.wfile.write(b"data: [DONE]\n\n")
        self.wfile.flush()


ThreadingHTTPServer(("127.0.0.1", 8791), H).serve_forever()
