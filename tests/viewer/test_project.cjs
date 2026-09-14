"use strict";
const assert = require("node:assert/strict");
const test = require("node:test");
const { client } = require("./client_harness.cjs");

function scene(id = 0, version = id) {
  return { scene_id: id, scene_version: version,
    scenes: [{id: 0, name: "0_intro"}, {id: 2, name: "2_outro"}],
    runtime: id + 1, fps: id ? 8 : 4, total_frames: id ? 24 : 4,
    width: 24, height: 16, epoch: 0, cached: [],
    resolution_name: "PREVIEW", resolution_options: [] };
}
function adopt(c, data = scene()) {
  c.run(`adoptState(${JSON.stringify(data)})`);
}

test("the page consumes the server runtime field and builds stable scene tabs", () => {
  const c = client(); adopt(c);
  assert.match(c.elements.get("meta").textContent, /1.00s/);
  const tabs = c.elements.get("scene-tabs");
  assert.equal(tabs.hidden, false);
  assert.equal(tabs.children.length, 2);
  assert.equal(tabs.children[1].textContent, "2_outro");
  assert.equal(tabs.children[0].attributes["aria-selected"], "true");
  assert.equal(tabs.children[1].tabIndex, -1);
});

test("selecting a tab resets playhead and inspection and uses its stable ID", async () => {
  const c = client(); adopt(c);
  c.run("state.frame=3; state.playing=true; state.selected=123; state.pixel={x:4,y:5}");
  const switching = c.elements.get("scene-tabs").children[1].onclick();
  assert.match(c.requests[0].url, /\/api\/scene\?id=2&/);
  assert.equal(c.run("state.playing"), false);
  assert.equal(c.run("state.frame"), 0);
  assert.equal(c.run("state.selected"), null);
  assert.equal(c.elements.get("scene-panel").inert, true);
  c.requests[0].reply(scene(2, 1)); await switching;
  assert.equal(c.run("state.sceneId"), 2);
  assert.equal(c.run("state.fps"), 8);
  assert.equal(c.elements.get("scene-panel").inert, false);
  assert.match(c.images[0].src, /&s=1$/);
  assert.ok(c.requests.slice(1).every(r => r.url.endsWith("&s=1")));
});

test("old bitmaps cannot overwrite a new scene even when frame and epoch match", async () => {
  const c = client(); adopt(c);
  const old = c.run("showFrame(0)");
  adopt(c, scene(2, 1));
  const current = c.run("showFrame(0)");
  c.images[1].onload(); await current;
  c.images[0].onload(); await old;
  assert.deepEqual(c.draws, [c.images[1]]);
});

test("stale hierarchy and transcript responses cannot repopulate the new scene", async () => {
  const c = client(); adopt(c);
  const oldTree = c.run("loadHierarchy()");
  const oldText = c.run("loadTranscript()");
  adopt(c, scene(2, 1));
  const currentTree = c.run("loadHierarchy()");
  const currentText = c.run("loadTranscript()");
  c.requests[2].reply({roots: [{node: 2, label:"NEW", kind:"mob"}]}); await currentTree;
  c.requests[3].reply({blocks: []}); await currentText;
  c.requests[0].reply({roots: [{node: 1, label:"OLD", kind:"mob"}]}); await oldTree;
  c.requests[1].reply({blocks:[{text:"old", timing:"unavailable", words:[]}]}); await oldText;
  assert.equal(c.elements.get("tree").children[0].children[0].children[1].textContent, "NEW");
  assert.equal(c.elements.get("transcript").children.length, 0);
  assert.equal(c.run("transcriptLoaded"), true);
});

test("stale fragments and attributes are ignored across scene switches", async () => {
  const c = client(); adopt(c);
  const fragments = c.run("inspect(2,3)");
  const attributes = c.run("state.selected=1; showAttributes()");
  adopt(c, scene(2, 1));
  c.requests[0].reply({label:"OLD FRAGMENTS"}); await fragments;
  c.requests[1].reply({label:"OLD ATTRIBUTES", attributes:[]}); await attributes;
  assert.doesNotMatch(c.elements.get("attrs").innerHTML, /OLD/);
  assert.doesNotMatch(c.elements.get("fragments").innerHTML, /OLD/);
});

test("a state poll already in flight cannot undo a clicked selection", async () => {
  const c = client(); adopt(c);
  const poll = c.run("refreshState()");
  const switching = c.run("changeScene(2)");
  c.requests[1].reply(scene(2, 1)); await switching;
  c.requests[0].reply(scene()); await poll;
  assert.equal(c.run("state.sceneId"), 2);
});

test("a stale resolution response cannot resize another scene", async () => {
  const c = client(); adopt(c);
  const resolution = c.run("changeResolution('HD')");
  adopt(c, scene(2, 1));
  c.requests[0].reply({...scene(), width:1920, height:1080, epoch:1});
  await resolution;
  assert.equal(c.run("state.width"), 24);
  assert.equal(c.run("state.epoch"), 0);
});

test("scene tabs support arrow keys and do not submit overlapping switches", async () => {
  const c = client(); adopt(c);
  const tabs = c.elements.get("scene-tabs");
  tabs.children[0].onkeydown({key:"ArrowRight", preventDefault(){}});
  assert.match(c.requests[0].url, /id=2/);
  assert.equal(c.requests.length, 1);
  c.run("changeScene(0)");
  assert.equal(c.requests.length, 1);
  c.requests[0].reply(scene(2, 1));
  await new Promise(resolve => setImmediate(resolve));
  assert.equal(tabs.children[1].attributes["aria-selected"], "true");
});

test("single-scene viewer requests remain unscoped and need no project catalog", () => {
  const c = client();
  const data = scene(); delete data.scenes; delete data.scene_id; delete data.scene_version;
  adopt(c, data);
  assert.equal(c.run("state.sceneVersion"), null);
  assert.doesNotMatch(c.run("api('/api/hierarchy')"), /&s=/);
});


test("a completed switch restores focus to the selected scene tab", async () => {
  const c = client(); adopt(c);
  const tabs = c.elements.get("scene-tabs");
  let focused = null;
  for (const button of tabs.children) button.focus = () => { focused = button; };
  const switching = c.run("changeScene(2)");
  c.requests[0].reply(scene(2, 1)); await switching;
  assert.equal(focused, tabs.children[1]);
});
