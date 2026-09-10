/* Exercise the real viewer client with a deterministic DOM/network boundary.
 * Run directly: node --test tests/viewer/test_viewer_async.cjs
 */
"use strict";

const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const test = require("node:test");
const vm = require("node:vm");

function client() {
  const elements = new Map();
  const images = [];
  const draws = [];
  const requests = [];
  function element() {
    return {
      style: {}, children: [], width: 32, height: 32,
      clientWidth: 320, clientHeight: 320,
      classList: { add() {}, remove() {} },
      addEventListener() {}, append(...items) { this.children.push(...items); },
      replaceChildren(...items) { this.children = items; },
      getBoundingClientRect() { return { left: 0, top: 0, width: 32, height: 32 }; },
      getContext() {
        return { drawImage(image) { draws.push(image); }, getImageData() { return { data: [0, 0, 0, 255] }; } };
      },
      set textContent(value) { this.innerHTML = String(value); },
      get textContent() { return this.innerHTML; },
      innerHTML: "",
    };
  }
  const sandbox = {
    document: {
      getElementById(id) {
        if (!elements.has(id)) elements.set(id, element());
        return elements.get(id);
      },
      querySelectorAll() { return []; }, createElement: element,
    },
    Image: class {
      constructor() { this.width = 32; this.height = 32; images.push(this); }
    },
    fetch(url) {
      return new Promise((resolve, reject) => {
        requests.push({ url, reject, reply(data) { resolve({ ok: true, json: async () => data }); } });
      });
    },
    window: { addEventListener() {} }, location: { search: "" }, URLSearchParams,
    performance: { now() { return 0; } }, requestAnimationFrame() {}, setTimeout,
  };
  const context = vm.createContext(sandbox);
  const filename = path.join(__dirname, "../../algan/viewer/static/viewer.js");
  const source = fs.readFileSync(filename, "utf8");
  const bootstrap = source.indexOf("(function start() {");
  assert.notEqual(bootstrap, -1, "test must remove only the page's automatic bootstrap");
  vm.runInContext(source.slice(0, bootstrap), context, { filename });
  const run = (code) => vm.runInContext(code, context);
  run("state.totalFrames = 10; renderFragments = (data) => { el('fragments').innerHTML = data.label; };");
  return { run, elements, images, draws, requests };
}

function attributes(label) {
  return { label, at: 0, attributes: [{ name: "location", value: [0, 0, 0] }] };
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
