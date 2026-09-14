"use strict";
const assert = require("node:assert/strict");
const test = require("node:test");
const { client } = require("./client_harness.cjs");

const data = { blocks: [
  { text: "First,\n\tsecond!", timing: "aligned", words: [
    { text: "First,", offset: 0, end_offset: 6, start: 1.05, end: 1.4 },
    { text: "second!", offset: 8, end_offset: 15, start: 2, end: 2.5 },
  ] },
  { text: "later", timing: "estimated", words: [
    { text: "later", offset: 0, end_offset: 5, start: 6, end: 7 },
  ] },
] };
function loaded(payload = data) {
  const c = client();
  c.run(`transcriptView.setData(${JSON.stringify(payload)}); state.fps = 10; state.totalFrames = 100;`);
  return c;
}
function current(c) {
  return JSON.parse(c.run("JSON.stringify([...transcriptView.active].map(e => e.textContent))"));
}
function textContent(node) {
  return node.children.length ? node.children.map(textContent).join("") : node.textContent;
}

test("tab switching preserves fragments and uses accessible keyboard navigation", () => {
  const c = loaded();
  c.run("el('fragments').innerHTML = 'inspection stays here'");
  const fragments = c.elements.get("fragments-tab");
  const transcript = c.elements.get("transcript-tab");
  transcript.onclick();
  assert.equal(transcript.attributes["aria-selected"], "true");
  assert.equal(c.elements.get("fragments-panel").hidden, true);
  assert.equal(c.elements.get("transcript-panel").hidden, false);
  let prevented = false;
  transcript.onkeydown({ key: "ArrowLeft", preventDefault() { prevented = true; } });
  assert.equal(prevented, true);
  assert.equal(fragments.attributes["aria-selected"], "true");
  assert.equal(c.run("document.activeElement === el('fragments-tab')"), true);
  assert.equal(c.elements.get("fragments").innerHTML, "inspection stays here");
  fragments.onkeydown({ key: "End", preventDefault() {} });
  assert.equal(c.run("transcriptView.tab"), "transcript");
});

test("highlight follows exact half-open intervals and clears in pauses", () => {
  const c = loaded();
  for (const [time, expected] of [
    [0, []], [1.05, ["First,"]], [1.4, []], [2, ["second!"]],
    [2.5, []], [6.5, ["later"]], [9, []], [1.2, ["First,"]],
  ]) {
    c.run(`transcriptView.update(${time})`);
    assert.deepEqual(current(c), expected, `t=${time}`);
  }
});

test("the playhead updates transcript before the frame finishes rendering", async () => {
  const c = loaded();
  const pending = c.run("showFrame(20)");
  assert.deepEqual(current(c), ["second!"]);
  assert.equal(c.draws.length, 0);
  c.images[0].onload();
  await pending;
});

test("clicking a word seeks via the player at its actual fps, and stops playback", async () => {
  const c = loaded();
  const pending = c.run("state.playing = true; transcriptView.cues[0].element.onclick()");
  assert.equal(c.run("state.frame"), 11, "first frame at or after 1.05 s at 10 fps");
  assert.equal(c.run("state.playing"), false);
  assert.match(c.requests[0].url, /\/api\/prefetch\?frame=11/);
  c.images[0].onload();
  await pending;
});

test("scrubbing backwards and time/frame input paths use the same transcript clock", async () => {
  const c = loaded();
  for (const [frame, expected] of [[65, ["later"]], [20, ["second!"]], [0, []]]) {
    const pending = c.run(`seek(${frame})`);
    assert.deepEqual(current(c), expected);
    c.images.at(-1).onload();
    await pending;
  }
});

test("scrolling is confined to the transcript, and opening a hidden tab catches up", () => {
  const c = loaded();
  const panel = c.elements.get("transcript-panel");
  panel.rect = { top: 0, bottom: 320, height: 320 };
  c.run("transcriptView.cues[2].element.rect = { top: 800, bottom: 832, height: 32 }; transcriptView.update(6.5)");
  assert.equal(panel.scrolls, undefined, "hidden tabs do not scroll");
  c.elements.get("transcript-tab").onclick();
  assert.equal(panel.scrolls, 1);
  assert.equal(panel.scrollTop, 656);
  c.run("transcriptView.update(6.6)");
  assert.equal(panel.scrolls, 1, "do not restart scrolling for every frame in a word");
  c.run("transcriptView.cues[0].element.rect = { top: -600, bottom: -568, height: 32 }; transcriptView.update(1.2)");
  assert.equal(panel.scrollTop, 0);
  assert.equal(c.elements.get("stage").scrolls, undefined);
});

test("overlapping narration highlights every audible word even in authoring order", () => {
  const c = loaded({ blocks: [
    { text: "late", timing: "aligned", words: [{ text: "late", offset: 0, end_offset: 4, start: 2, end: 3 }] },
    { text: "long", timing: "aligned", words: [{ text: "long", offset: 0, end_offset: 4, start: 0, end: 8 }] },
  ] });
  c.run("transcriptView.update(2.5)");
  assert.deepEqual(current(c), ["late", "long"]);
  c.run("transcriptView.update(4)");
  assert.deepEqual(current(c), ["long"]);
});

test("original punctuation, Unicode offsets and HTML-looking narration are literal text", () => {
  const c = client();
  // Offsets are Python code points, including the leading emoji.
  const literal = "🙂  <b>hello</b>\n\tworld!";
  const chars = Array.from(literal);
  const payload = { blocks: [{ text: literal, timing: "aligned", words: [
    { text: "🙂", offset: 0, end_offset: 1, start: null, end: null },
    { text: "<b>hello</b>", offset: 3, end_offset: 3 + "<b>hello</b>".length, start: 0, end: 1 },
    { text: "world!", offset: chars.length - 6, end_offset: chars.length, start: 1, end: 2 },
  ] }] };
  c.run(`transcriptView.setData(${JSON.stringify(payload)})`);
  assert.equal(textContent(c.elements.get("transcript")), literal);
  assert.equal(c.run("transcriptView.cues.length"), 2);
});

test("empty and untimed transcripts are explicit; estimated timing is labeled", () => {
  let c = loaded({ blocks: [] });
  assert.match(c.elements.get("transcript-status").textContent, /No Speech blocks/);
  c = loaded({ blocks: [{ text: "silent", timing: "unavailable", words: [
    { text: "silent", offset: 0, end_offset: 6, start: null, end: null },
  ] }] });
  assert.equal(textContent(c.elements.get("transcript")), "silent");
  assert.equal(c.run("transcriptView.cues.length"), 0);
  assert.match(c.elements.get("transcript-status").textContent, /without playable audio/);
  c = loaded();
  assert.match(c.elements.get("transcript-status").textContent, /estimated/);
});

test("transcript load failures retry, concurrent loads coalesce, successful loads stay cached", async () => {
  const c = client();
  const first = c.run("loadTranscript()");
  await c.run("loadTranscript()");
  assert.equal(c.requests.length, 1);
  c.requests[0].reject(new Error("offline"));
  await first;
  assert.match(c.elements.get("transcript-status").textContent, /retrying/);
  const retry = c.run("loadTranscript()");
  c.requests[1].reply(data);
  await retry;
  await c.run("loadTranscript()");
  assert.equal(c.requests.length, 2);
});
