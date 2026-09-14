"use strict";
const assert = require("node:assert/strict");
const test = require("node:test");
const { client } = require("./client_harness.cjs");
const flush = () => new Promise(resolve => setImmediate(resolve));

function setup() {
  const c = client();
  c.run("state.hasAudio=true; state.fps=10; state.totalFrames=20; state.duration=2");
  return c;
}
function frame(c, index) {
  const image = c.images.find(image => image.src.includes(`/frame/${index}.png?`));
  assert.ok(image, `frame ${index} requested`);
  image.onload();
}
async function start(c) {
  const playing = c.run("play()");
  const request = c.requests.find(request => request.url.startsWith("/audio.wav"));
  request.replyAudio({ duration: 2 });
  frame(c, c.run("state.frame"));
  await playing;
  return c.audioContexts[0];
}

test("audio is lazy, permission is requested in Play, and sound waits for the first picture", async () => {
  const c = setup();
  assert.equal(c.audioContexts.length, 0);
  assert.equal(c.requests.length, 0);
  const playing = c.run("play()");
  const context = c.audioContexts[0];
  assert.equal(context.resumes, 1);
  assert.match(c.requests[0].url, /^\/audio.wav\?t=/);
  c.requests[0].replyAudio({ duration: 2 });
  await flush();
  assert.equal(context.sources.length, 0);
  frame(c, 0); await playing;
  assert.equal(context.sources.length, 1);
  assert.equal(context.sources[0].offset, 0);
});

test("silent scenes play without creating an audio context or requesting a mix", async () => {
  const c = setup(); c.run("state.hasAudio=false");
  const playing = c.run("play()");
  frame(c, 0); await playing;
  assert.equal(c.requests.length, 0);
  assert.equal(c.audioContexts.length, 0);
  frame(c, 2); c.advance(0.2); await c.run("tick()");
  assert.equal(c.run("state.frame"), 2);
});

test("Stop stops and disconnects sound; replay uses the cached mix and displayed time", async () => {
  const c = setup(); const context = await start(c);
  frame(c, 2); c.advance(0.2); await c.run("tick()");
  c.run("stop()");
  assert.equal(context.sources[0].stopped, true);
  assert.equal(context.sources[0].disconnected, true);
  await c.run("play()");
  assert.equal(context.sources[1].offset, 0.2);
  assert.equal(c.requests.filter(r => r.url.startsWith("/audio.wav")).length, 1);
});

test("scrubber/transcript seeking stops audio and the next Play starts at the selected time", async () => {
  const c = setup(); const context = await start(c);
  frame(c, 3);
  await c.run("seek(3)");
  assert.equal(context.sources[0].stopped, true);
  assert.equal(c.run("state.playing"), false);
  assert.equal(c.run("state.frame"), 3);
  await c.run("play()");
  assert.equal(context.sources[1].offset, 0.3);
});

test("buffering freezes audio before the pending image resolves and resumes the same clock", async () => {
  const c = setup(); const context = await start(c);
  c.advance(0.21);
  const ticking = c.run("tick()");
  assert.equal(context.sources[0].stopped, true);
  const frozen = c.run("audio.time()");
  c.advance(5);
  assert.equal(c.run("audio.time()"), frozen);
  frame(c, 2); await ticking;
  assert.equal(context.sources[1].offset, frozen);
  assert.equal(c.run("audio.time()"), frozen);
});

test("Stop while buffering prevents late frames from restarting audio or another loop", async () => {
  const c = setup(); const context = await start(c);
  c.advance(0.2); const ticking = c.run("tick()");
  c.run("stop()");
  const loops = c.animationFrames.length;
  frame(c, 2); await ticking;
  assert.equal(context.sources.length, 1);
  assert.equal(c.animationFrames.length, loops);
});

test("Stop while loading and an immediate new Play start exactly one source", async () => {
  const c = setup();
  const old = c.run("play()");
  c.run("stop()");
  const current = c.run("play()");
  c.requests[0].replyAudio({duration:2}); frame(c, 0);
  await Promise.all([old, current]);
  assert.equal(c.audioContexts[0].sources.length, 1);
  assert.equal(c.animationFrames.length, 1);
});

test("a scene switch discards pending audio and prevents the old scene from starting", async () => {
  const c = setup(); c.run("state.sceneVersion=1");
  const old = c.run("play()");
  assert.match(c.requests[0].url, /&s=1$/);
  c.run("clearScene(); state.sceneReady=true; state.hasAudio=true; state.sceneVersion=2");
  const current = c.run("play()");
  assert.match(c.requests[1].url, /&s=2$/);
  c.requests[1].replyAudio({duration:2, label:"new"});
  c.images[1].onload(); await current;
  c.requests[0].replyAudio({duration:2, label:"old"});
  c.images[0].onload(); await old;
  assert.equal(c.run("audio.buffer.label"), "new");
  assert.equal(c.audioContexts[0].sources.length, 1);
});

test("scene changes and page exit stop playing sources immediately", async () => {
  for (const action of [c => c.run("clearScene()"), c => c.listeners.get("pagehide")()]) {
    const c = setup(); const context = await start(c);
    action(c);
    assert.equal(context.sources[0].stopped, true);
    assert.equal(c.run("state.playing"), false);
  }
});

test("the last fractional frame of audio plays before stopping; Play then restarts at zero", async () => {
  const c = setup(); c.run("state.totalFrames=10; state.duration=1.03");
  const context = await start(c);
  c.run("frameImage(9)"); frame(c, 9);
  c.advance(1.01); await c.run("tick()");
  assert.equal(c.run("state.playing"), true);
  assert.equal(c.run("state.frame"), 9);
  c.advance(0.03); await c.run("tick()");
  assert.equal(context.sources[0].stopped, true);
  assert.equal(c.run("state.playing"), false);
  await c.run("play()");
  assert.equal(context.sources[1].offset, 0);
});

test("changing resolution stops audio without invalidating its scene mix", async () => {
  const c = setup(); const context = await start(c);
  const mix = c.run("audio.buffer");
  const change = c.run("changeResolution('HD')");
  assert.equal(context.sources[0].stopped, true);
  const request = c.requests.find(request => request.url.startsWith("/api/resolution"));
  request.reject(new Error("test")); await change;
  assert.equal(c.run("audio.buffer"), mix);
});

test("audio errors stop playback visibly and a later Play can retry", async () => {
  const c = setup(); const playing = c.run("play()");
  c.requests[0].reject(new Error("mix failed"));
  frame(c, 0); await playing;
  assert.equal(c.run("state.playing"), false);
  assert.match(c.elements.get("status").textContent, /mix failed/);
  const retry = c.run("play()");
  c.requests[1].replyAudio({duration:2}); await retry;
  assert.equal(c.audioContexts[0].sources.length, 1);
});

test("audio decode and autoplay failures are reported rather than playing silent video", async () => {
  for (const failure of ["decodeAudioData", "resume"]) {
    const c = setup();
    c.run(`audio.context = new window.AudioContext();
      audio.context.${failure} = () => Promise.reject(new Error('blocked ${failure}'))`);
    const playing = c.run("play()");
    c.requests[0].replyAudio({duration:2}); frame(c, 0); await playing;
    assert.equal(c.run("state.playing"), false);
    assert.match(c.elements.get("status").textContent, /blocked/);
    assert.equal(c.audioContexts[0].sources.length, 0);
  }
});

test("a late first picture cannot erase an audio failure or revive stopped playback", async () => {
  const c = setup(); const playing = c.run("play()");
  c.requests[0].reject(new Error("mix failed"));
  await playing;
  frame(c, 0); await flush();
  assert.match(c.elements.get("status").textContent, /mix failed/);
  assert.equal(c.audioContexts[0].sources.length, 0);
  assert.equal(c.draws.length, 0);
});
