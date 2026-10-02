/* Deterministic regressions for the real viewer's asynchronous client. */
"use strict";
const assert = require("node:assert/strict");
const test = require("node:test");
const { client } = require("./client_harness.cjs");

function attributes(label) {
  return { label, at: 0, attributes: [{ name: "location", value: [0, 0, 0] }] };
}

for (const pending of [false, true]) {
  test(`seeking refreshes selected attributes with an ${pending ? "in-flight" : "completed"} request`, async () => {
    const c = client();
    const old = c.run("state.selected = 'a'; showAttributes()");
    if (!pending) {
      c.requests[0].reply(attributes("old")); await old;
    }
    const seek = c.run("seek(5)");
    c.images[0].onload(); await seek;
    const refresh = c.requests.filter((r) => r.url.includes("/api/attrs"));
    assert.equal(refresh.length, 2);
    assert.match(refresh[1].url, /frame=5/);
    refresh[1].reply(attributes("new"));
    await new Promise((resolve) => setImmediate(resolve));
    if (pending) {
      refresh[0].reply(attributes("old")); await old;
    }
    assert.match(c.elements.get("attrs").innerHTML, /new at t=/);
  });
}

test("stopping playback refreshes the selected object's current attributes", async () => {
  const c = client();
  c.run("state.selected = 'a'; state.frame = 5; state.playing = true; stop()");
  assert.match(c.requests[0].url, /frame=5/);
  c.requests[0].reply(attributes("stopped"));
  await new Promise((resolve) => setImmediate(resolve));
  assert.match(c.elements.get("attrs").innerHTML, /stopped at t=/);
});

for (const polled of [false, true]) {
  test(`${polled ? "a polled" : "a requested"} resolution change refreshes selected attributes at the new epoch`, async () => {
    const c = client();
    const data = {
      runtime: 2.5, fps: 4, total_frames: 10, width: 32, height: 32,
      epoch: 0, cached: [], resolution_name: "PREVIEW", resolution_options: [],
    };
    c.run(`adoptState(${JSON.stringify(data)}); transcriptLoaded = true; el('tree').append(document.createElement('div'))`);
    const old = c.run("state.selected = 'a'; showAttributes()");
    const change = c.run(polled ? "refreshState()" : "changeResolution('LOW')");
    assert.match(c.requests[1].url, polled ? /\/api\/state/ : /\/api\/resolution/);
    c.requests[1].reply({ ...data, epoch: 1, resolution_name: "LOW" });
    await new Promise((resolve) => setImmediate(resolve));
    c.images[0].onload(); await change;
    const refresh = c.requests.filter((r) => r.url.includes("/api/attrs"));
    assert.equal(refresh.length, 2);
    refresh[1].reply(attributes("new resolution"));
    await new Promise((resolve) => setImmediate(resolve));
    refresh[0].reply(attributes("old resolution")); await old;
    assert.match(c.elements.get("attrs").innerHTML, /new resolution at t=/);
  });
}

test("a frame from an old resolution cannot replace the new bitmap", async () => {
  const c = client();
  const old = c.run("showFrame(0)");
  c.run("state.epoch++; state.images.clear()");
  const current = c.run("showFrame(0)");
  c.images[1].onload(); await current;
  c.images[0].onload(); await old;
  assert.deepEqual(c.draws, [c.images[1]]);
});

test("an old frame failure cannot erase the replacement cache entry", async () => {
  const c = client();
  const old = c.run("frameImage(0)");
  const failed = assert.rejects(old);
  c.run("state.epoch++; state.images.clear()");
  const current = c.run("frameImage(0)");
  c.images[0].onerror(); await failed;
  assert.equal(c.run("state.images.get(0)"), current);
  c.images[1].onload(); await current;
});

test("an obsolete frame failure does not put a successfully drawn frame back into loading", async () => {
  const c = client();
  const old = c.run("showFrame(0)");
  const current = c.run("showFrame(1)");
  c.images[1].onload(); await current;
  c.images[0].onerror(); await old;
  assert.equal(c.elements.get("status").textContent, "");
});

test("the newest pixel inspection owns the fragment panel, including repeat clicks", async () => {
  const c = client();
  const old = c.run("inspect(2, 3)");
  const current = c.run("inspect(2, 3)");
  c.requests[1].reply({ label: "new" }); await current;
  c.requests[0].reply({ label: "old" }); await old;
  assert.equal(c.elements.get("fragments").innerHTML, "new");
});

test("changing frame or resolution invalidates an in-flight pixel inspection", async () => {
  for (const change of ["state.frame++", "state.epoch++"]) {
    const c = client();
    const old = c.run("inspect(2, 3)");
    c.run(change + "; el('fragments').innerHTML = 'current'");
    c.requests[0].reply({ label: "old" }); await old;
    assert.equal(c.elements.get("fragments").innerHTML, "current");
  }
});

test("a stale inspection error cannot replace a newer answer", async () => {
  const c = client();
  const old = c.run("inspect(2, 3)");
  const current = c.run("inspect(4, 5)");
  c.requests[1].reply({ label: "new" }); await current;
  c.requests[0].reject(new Error("old failure")); await old;
  assert.equal(c.elements.get("fragments").innerHTML, "new");
});

test("the latest selection owns the attributes panel, including repeat requests", async () => {
  const c = client();
  const old = c.run("state.selected = 'a'; showAttributes()");
  const current = c.run("showAttributes()");
  c.requests[1].reply(attributes("new")); await current;
  c.requests[0].reply(attributes("old")); await old;
  assert.match(c.elements.get("attrs").innerHTML, /new at t=/);
});

test("changing selection, frame or resolution invalidates in-flight attributes", async () => {
  for (const change of ["state.selected = 'b'", "state.frame++", "state.epoch++"]) {
    const c = client();
    const old = c.run("state.selected = 'a'; showAttributes()");
    c.run(change + "; el('attrs').innerHTML = 'current'");
    c.requests[0].reply(attributes("old")); await old;
    assert.equal(c.elements.get("attrs").innerHTML, "current");
  }
});

test("a stale attribute error cannot replace a newer answer", async () => {
  const c = client();
  const old = c.run("state.selected = 'a'; showAttributes()");
  const current = c.run("state.selected = 'b'; showAttributes()");
  c.requests[1].reply(attributes("new")); await current;
  c.requests[0].reject(new Error("old failure")); await old;
  assert.match(c.elements.get("attrs").innerHTML, /new at t=/);
});


test("a failed hierarchy expansion can be retried and does not issue concurrent duplicate loads", async () => {
  const c = client();
  const row = c.run("nodeRow({ node: 'parent', label: 'Parent', kind: 'mob', spawned: true, has_children: true })");
  const arrow = row.children[0].children[0];
  const children = row.children[1];
  const failed = arrow.onclick();
  const failure = assert.doesNotReject(failed);
  c.requests[0].reject(new Error("temporarily unavailable"));
  await failure;
  assert.match(children.children[0].textContent, /retry/);
  await arrow.onclick(); // collapse
  const retry = arrow.onclick();
  assert.equal(c.requests.length, 2);
  await arrow.onclick(); // collapse while loading
  await arrow.onclick(); // expand while the same load is pending
  assert.equal(c.requests.length, 2);
  c.requests[1].reply({ children: [] });
  await retry;
  await arrow.onclick();
  await arrow.onclick();
  assert.equal(c.requests.length, 2, "successful results stay cached");
});
