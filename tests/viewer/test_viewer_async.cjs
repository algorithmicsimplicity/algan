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
    const classes = new Set();
    const attributes = new Map();
    return {
      style: {}, children: [], width: 32, height: 32,
      clientWidth: 320, clientHeight: 320,
      scrollTop: 0, hidden: false,
      classList: { add(name) { classes.add(name); }, remove(name) { classes.delete(name); }, contains(name) { return classes.has(name); } },
      setAttribute(name, value) { attributes.set(name, value); },
      removeAttribute(name) { attributes.delete(name); },
      getAttribute(name) { return attributes.get(name); },
      focus() { sandbox.document.activeElement = this; },
      addEventListener() {}, append(...items) { this.children.push(...items); },
      replaceChildren(...items) { this.children = items; },
      getBoundingClientRect() { return this.rect || { left: 0, top: 0, bottom: 320, width: 320, height: 320 }; },
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
      createTextNode(text) { return { textContent: text }; },
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

function transcript(c, blocks) {
  c.run(`renderTranscript(${JSON.stringify({ blocks })})`);
}
function line(words, timing = "aligned") {
  return { text: words.map(w => w[0]).join(" "), after: "", timing,
    words: words.map(([text, start, end], i) => ({ text, start, end, before: i ? " " : "" })) };
}

test("tabs preserve fragment content and support keyboard navigation", () => {
  const c = client();
  c.run("el('fragments').textContent = 'keep'; selectInspectorTab('transcript')");
  assert.equal(c.elements.get("fragments-panel").hidden, true);
  assert.equal(c.elements.get("transcript-panel").hidden, false);
  assert.equal(c.elements.get("transcript-tab").getAttribute("aria-selected"), "true");
  c.elements.get("transcript-tab").onkeydown({ key: "Home", preventDefault() {} });
  assert.equal(c.elements.get("fragments-panel").hidden, false);
  assert.equal(c.elements.get("fragments").textContent, "keep");
  assert.equal(c.run("document.activeElement === el('fragments-tab')"), true);
});

test("word highlighting respects silence, boundaries, and backward seeks", () => {
  const c = client();
  transcript(c, [line([["first", 1, 2], ["second", 3, 4]])]);
  for (const [time, expected] of [[0, ""], [1, "first"], [2, ""], [3.5, "second"], [4, ""], [1.5, "first"]]) {
    c.run(`syncTranscript(${time})`);
    assert.equal(c.run("[...transcriptState.active].map(n => n.textContent).join(',')"), expected);
  }
});

test("clicking a word seeks at its onset even when rounding would go backward", () => {
  const c = client();
  transcript(c, [line([["word", 1.01, 1.5]])]);
  c.run("state.fps = 30; seek = index => { state.frame = index; }; transcriptState.entries[0].node.onclick()");
  assert.equal(c.run("state.frame"), 31);
});

test("all frame changes update transcript before a frame finishes loading", () => {
  const c = client();
  transcript(c, [line([["first", 0, .1], ["second", .1, .2]])]);
  c.run("state.fps = 30; showFrame(4)");
  assert.equal(c.run("[...transcriptState.active][0].textContent"), "second");
});

test("play mode uses the same transcript clock as the playhead", async () => {
  const c = client();
  transcript(c, [line([["first", 0, .5], ["second", .5, 1]])]);
  await c.run("frameImage = async () => ({width: 32, height: 32}); state.fps = 10; state.playing = true; state.playStartedAt = -600; tick()");
  assert.equal(c.run("state.frame"), 6);
  assert.equal(c.run("[...transcriptState.active][0].textContent"), "second");
});

test("overlapping speech is time indexed independently from authoring order", () => {
  const c = client();
  transcript(c, [line([["later", 2, 3]]), line([["long", 0, 5]])]);
  c.run("syncTranscript(2.5)");
  assert.equal(c.run("transcriptState.active.size"), 2);
  c.run("syncTranscript(3.5)");
  assert.equal(c.run("[...transcriptState.active][0].textContent"), "long");
});

test("scrolling follows the time position only inside the visible transcript", () => {
  const c = client();
  transcript(c, [line([["first", 0, 1], ["last", 5, 6]])]);
  c.run("transcriptState.entries[1].node.rect = {top: 900, bottom: 925}; syncTranscript(5.5)");
  assert.equal(c.elements.get("transcript").scrollTop, 0, "hidden tabs do not scroll");
  c.run("state.fps = 10; state.frame = 55; selectInspectorTab('transcript')");
  assert.ok(c.elements.get("transcript").scrollTop > 0);
  assert.equal(c.elements.get("stage").scrollTop, 0);
  const top = c.elements.get("transcript").scrollTop;
  c.run("syncTranscript(5.6)");
  assert.equal(c.elements.get("transcript").scrollTop, top, "same word does not jitter");
});

test("transcript text is inserted as text, not HTML", () => {
  const c = client();
  transcript(c, [line([["<script>oops</script>", 0, 1]])]);
  assert.equal(c.run("transcriptState.entries[0].node.textContent"), "<script>oops</script>");
});

test("untimed text and empty scenes have no fake clickable words", () => {
  const c = client();
  transcript(c, [line([["silent", null, null]], "unavailable")]);
  assert.equal(c.run("transcriptState.entries.length"), 0);
  transcript(c, []);
  assert.match(c.elements.get("transcript").children[0].textContent, /no Speech blocks/);
});

test("missing alignment is disclosed", () => {
  const c = client();
  transcript(c, [line([["estimate", 0, 1]], "estimated")]);
  assert.match(c.elements.get("transcript-note").textContent, /estimated/);
  assert.match(c.run("transcriptState.entries[0].node.title"), /estimated/);
});

test("transcript load is deduplicated, retried after failure, then cached", async () => {
  const c = client();
  const first = c.run("loadTranscript()");
  await c.run("loadTranscript()");
  assert.equal(c.requests.length, 1);
  c.requests[0].reject(new Error("offline")); await first;
  assert.match(c.elements.get("transcript").textContent, /Retrying/);
  const retry = c.run("loadTranscript()");
  c.requests[1].reply({ blocks: [] }); await retry;
  await c.run("loadTranscript()");
  assert.equal(c.requests.length, 2);
});

test("bootstrap accepts the server's runtime field", () => {
  const c = client();
  c.run("syncResolution = () => {}; adoptState({fps: 30, total_frames: 90, runtime: 3, width: 640, height: 360})");
  assert.equal(c.run("state.duration"), 3);
  assert.match(c.elements.get("meta").textContent, /3.00s/);
});
