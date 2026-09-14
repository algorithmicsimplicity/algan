/* Exercise the real viewer client with a deterministic DOM/network boundary.
 * Run directly: node --test tests/viewer/test_viewer_async.cjs
 */
"use strict";

const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const vm = require("node:vm");

function client() {
  const elements = new Map();
  const images = [];
  const draws = [];
  const requests = [];
  const audioContexts = [];
  const animationFrames = [];
  const listeners = new Map();
  let now = 0;
  class AudioContext {
    constructor() {
      this.destination = {};
      this.sources = [];
      this.resumes = 0;
      audioContexts.push(this);
    }
    get currentTime() { return now / 1000; }
    resume() { this.resumes++; return Promise.resolve(); }
    decodeAudioData(data) { return Promise.resolve(data); }
    createBufferSource() {
      const source = {
        connect() {}, disconnect() { this.disconnected = true; },
        start(when, offset) { this.offset = offset; this.started = true; },
        stop() { this.stopped = true; },
      };
      this.sources.push(source);
      return source;
    }
  }
  function element() {
    return {
      style: {}, children: [], width: 32, height: 32,
      clientWidth: 320, clientHeight: 320,
      dataset: {}, hidden: false, scrollTop: 0, attributes: {},
      classList: {
        values: new Set(),
        add(name) { this.values.add(name); },
        remove(name) { this.values.delete(name); },
        contains(name) { return this.values.has(name); },
      },
      setAttribute(name, value) { this.attributes[name] = value; },
      removeAttribute(name) { delete this.attributes[name]; },
      focus() { sandbox.document.activeElement = this; },
      scrollTo({top}) { this.scrollTop = top; this.scrolls = (this.scrolls || 0) + 1; },
      addEventListener() {}, append(...items) { this.children.push(...items); },
      replaceChildren(...items) { this.children = items; },
      getBoundingClientRect() { return this.rect || { left: 0, top: 0, bottom: 320, width: 32, height: 32 }; },
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
      createTextNode(text) { const node = element(); node.textContent = text; return node; },
    },
    Image: class {
      constructor() { this.width = 32; this.height = 32; images.push(this); }
    },
    fetch(url) {
      return new Promise((resolve, reject) => {
        requests.push({ url, reject,
          reply(data) { resolve({ ok: true, json: async () => data }); },
          replyAudio(data) { resolve({ ok: true, arrayBuffer: async () => data }); },
        });
      });
    },
    window: { AudioContext, addEventListener(name, listener) { listeners.set(name, listener); } },
    location: { search: "" }, URLSearchParams,
    performance: { now() { return now; } },
    requestAnimationFrame(callback) { animationFrames.push(callback); }, setTimeout,
  };
  const context = vm.createContext(sandbox);
  const transcriptFilename = path.join(__dirname, "../../algan/viewer/static/transcript.js");
  vm.runInContext(fs.readFileSync(transcriptFilename, "utf8"), context, { filename: transcriptFilename });
  const audioFilename = path.join(__dirname, "../../algan/viewer/static/audio.js");
  vm.runInContext(fs.readFileSync(audioFilename, "utf8"), context, { filename: audioFilename });
  const filename = path.join(__dirname, "../../algan/viewer/static/viewer.js");
  const source = fs.readFileSync(filename, "utf8");
  const bootstrap = source.indexOf("(function start() {");
  assert.notEqual(bootstrap, -1, "test must remove only the page's automatic bootstrap");
  vm.runInContext(source.slice(0, bootstrap), context, { filename });
  const run = (code) => vm.runInContext(code, context);
  run("state.sceneReady = true; state.totalFrames = 10; renderFragments = (data) => { el('fragments').innerHTML = data.label; };");
  return { run, elements, images, draws, requests, audioContexts, animationFrames, listeners,
    advance(seconds) { now += seconds * 1000; },
  };
}

module.exports = { client };
