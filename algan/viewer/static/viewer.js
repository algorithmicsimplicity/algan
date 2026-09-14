/* The Algan viewer's page.
 *
 * Frames arrive as PNGs and are drawn to a canvas rather than shown in an
 * <img>, which is what makes the hover readout free: the pixel under the mouse
 * is one getImageData away, with no round trip to the server. The fragment list
 * behind a pixel does need the server, because only the renderer knows it.
 */
"use strict";

const state = {
  fps: 30, totalFrames: 1, duration: 0, width: 1, height: 1,
  frame: 0, playing: false, selected: null, drawn: false,
  pixel: { x: 0, y: 0 },
  images: new Map(),
  playStartedAt: 0, playStartedFrame: 0,
  // Zoom is 1 = fit the stage. It is presentation only: the canvas keeps the
  // render's own pixel grid, so an inspected pixel is the same pixel however
  // far in you are.
  zoom: 1, epoch: 0, resolutionName: null, resolutionKeys: "",
  pixelRequest: 0, attributeRequest: 0,
  sceneId: null, sceneVersion: null, sceneKeys: "", generation: 0,
  switching: false,
};

const el = (id) => document.getElementById(id);
const canvas = el("frame");
const stage = el("stage");
const ctx = canvas.getContext("2d", { willReadFrequently: true });

// The player remains the single clock: transcript seeking uses the same path
// as the scrubber, and highlighting follows the marker even during rendering.
const transcriptView = new TranscriptView((seconds) =>
  seek(Math.ceil(seconds * state.fps - 1e-7)));

/* ---------- server ---------- */

/* This viewer session's key to its own API.
 *
 * Every route but the page and its static files requires it, because binding
 * to 127.0.0.1 does not keep other pages out: any origin can POST to a
 * localhost URL, which is all it takes to hit /api/shutdown. It arrives in the
 * URL the viewer printed, so a reload keeps it and a bookmark of the bare port
 * does not. */
const TOKEN = new URLSearchParams(location.search).get("t") || "";

/* ``path`` with the token added, whichever separator it needs. */
function api(path) {
  const globalRoute = /^\/api\/(state|scene|shutdown)(\?|$)/.test(path);
  const scene = !globalRoute && state.sceneVersion !== null
    ? `&s=${state.sceneVersion}` : "";
  return path + (path.includes("?") ? "&" : "?") + "t=" + encodeURIComponent(TOKEN) + scene;
}

async function getJSON(url, address = api(url)) {
  const response = await fetch(address);
  if (!response.ok) throw new Error((await response.json()).error || response.statusText);
  return response.json();
}

/* The same, but patient about the server being busy.
 *
 * Every route that touches the Scene queues behind whatever render is holding
 * it, so a request can take a batch to answer -- and a browser that gives up
 * waiting rejects with a TypeError ("Failed to fetch") rather than with an
 * answer. Those are worth asking again. A rejection carrying the server's own
 * message is an answer, and asking twice would not improve it.
 */
async function getJSONPatiently(url, attempts = 5) {
  let last;
  // A retry must address the original scene, not whichever tab is now active.
  const address = api(url);
  for (let attempt = 0; attempt < attempts; attempt++) {
    try {
      return await getJSON(url, address);
    } catch (err) {
      if (!(err instanceof TypeError)) throw err;
      last = err;
      await new Promise((r) => setTimeout(r, Math.min(400 * 2 ** attempt, 3000)));
    }
  }
  throw last;
}

function frameImage(index) {
  if (state.images.has(index)) return state.images.get(index);
  const promise = new Promise((resolve, reject) => {
    const image = new Image();
    image.onload = () => resolve(image);
    image.onerror = () => {
      // A request from a discarded resolution may finish after its replacement.
      if (state.images.get(index) === promise) state.images.delete(index);
      reject(new Error("not ready"));
    };
    // The epoch busts both caches after a resolution change: this map's, and
    // the browser's own, which is told frames are immutable for a day.
    image.src = api(`/frame/${index}.png?v=${state.epoch}`);
  });
  state.images.set(index, promise);
  // The cache is a convenience, not a store: an hour of video would be a lot of
  // decoded bitmaps, so old ones are dropped once there are plenty.
  if (state.images.size > 400) {
    state.images.delete(state.images.keys().next().value);
  }
  return promise;
}

/* ---------- drawing ---------- */

async function showFrame(index, { redrawOnly = false } = {}) {
  index = Math.max(0, Math.min(index, state.totalFrames - 1));
  state.frame = index;
  const epoch = state.epoch;
  const generation = state.generation;
  updateReadouts();
  try {
    const image = await frameImage(index);
    if (state.frame !== index || state.epoch !== epoch || state.generation !== generation) return true;
    if (canvas.width !== image.width || canvas.height !== image.height) {
      canvas.width = image.width;
      canvas.height = image.height;
      applyZoom();
    }
    ctx.drawImage(image, 0, 0);
    state.drawn = true;
    setStatus("");
    return true;
  } catch (err) {
    if (state.frame !== index || state.epoch !== epoch || state.generation !== generation) return true;
    if (!redrawOnly) setStatus("rendering…", "busy");
    return false;
  }
}

/* ---------- zoom ---------- */

/* The scale at which the frame exactly fits the stage, which is what zoom 1
 * means. Recomputed rather than remembered: the stage changes size with the
 * window, and the frame changes size with the resolution picker. */
function fitScale() {
  const w = Math.max(1, stage.clientWidth);
  const h = Math.max(1, stage.clientHeight);
  return Math.min(w / canvas.width, h / canvas.height);
}

function applyZoom() {
  // Nothing can be scrolled at the fit, and leaving `overflow: auto` on there
  // starts a feedback loop: a rounding pixel raises a scrollbar, the scrollbar
  // narrows the stage, and the fit no longer fits. Only zoomed-in views scroll.
  stage.style.overflow = state.zoom > 1 ? "auto" : "hidden";
  const scale = fitScale() * state.zoom;
  canvas.style.width = `${Math.floor(canvas.width * scale)}px`;
  canvas.style.height = `${Math.floor(canvas.height * scale)}px`;
  el("zoom-readout").textContent = `${Math.round(state.zoom * 100)}%`;
  if (document.activeElement !== el("zoom")) {
    el("zoom").value = String(Math.round(state.zoom * 100));
  }
}

/* Zoom, keeping one point of the frame under the same screen position.
 * Without the anchor a wheel zoom walks away from whatever you were aiming at,
 * which is exactly the thing this feature exists to make easy. */
function setZoom(next, anchor) {
  const lo = 1, hi = 16;
  next = Math.max(lo, Math.min(next, hi));
  if (next === state.zoom) return;
  const before = canvas.getBoundingClientRect();
  const fx = anchor ? (anchor.clientX - before.left) / before.width : 0.5;
  const fy = anchor ? (anchor.clientY - before.top) / before.height : 0.5;
  const keepX = anchor ? anchor.clientX : before.left + before.width / 2;
  const keepY = anchor ? anchor.clientY : before.top + before.height / 2;
  state.zoom = next;
  applyZoom();
  const after = canvas.getBoundingClientRect();
  stage.scrollLeft += (after.left + fx * after.width) - keepX;
  stage.scrollTop += (after.top + fy * after.height) - keepY;
}

function updateReadouts() {
  const time = state.frame / state.fps;
  if (document.activeElement !== el("time-input")) {
    el("time-input").value = time.toFixed(3);
  }
  if (document.activeElement !== el("frame-input")) {
    el("frame-input").value = String(state.frame);
  }
  const fraction = state.totalFrames > 1
    ? state.frame / (state.totalFrames - 1) : 0;
  el("progress").style.width = `${fraction * 100}%`;
  el("playhead").style.left = `${fraction * 100}%`;
  transcriptView.update(time);
}

function setStatus(text, kind = "") {
  const node = el("status");
  node.textContent = text;
  node.className = `status ${kind}`;
}

/* ---------- playback ---------- */

function play() {
  if (state.playing || state.switching) return;
  state.playing = true;
  el("play").textContent = "Stop";
  state.playStartedAt = performance.now();
  state.playStartedFrame = state.frame >= state.totalFrames - 1 ? 0 : state.frame;
  requestAnimationFrame(tick);
}

function stop() {
  state.playing = false;
  el("play").textContent = "Play";
}

async function tick() {
  if (!state.playing) return;
  const generation = state.generation;
  const elapsed = (performance.now() - state.playStartedAt) / 1000;
  const target = state.playStartedFrame + Math.floor(elapsed * state.fps);
  if (target >= state.totalFrames) { stop(); await showFrame(state.totalFrames - 1); return; }
  if (target !== state.frame) {
    const drawn = await showFrame(target);
    if (!state.playing || generation !== state.generation) return;
    if (!drawn) {
      // The frame is not rendered yet. Hold the clock where it is so playback
      // resumes from here instead of skipping the frames spent waiting.
      state.playStartedAt = performance.now();
      state.playStartedFrame = target;
    }
  }
  requestAnimationFrame(tick);
}

async function seek(index) {
  if (state.switching) return;
  const generation = state.generation;
  stop();
  index = Math.max(0, Math.min(Math.round(index), state.totalFrames - 1));
  fetch(api(`/api/prefetch?frame=${index}`)).catch(() => {});
  const drawn = await showFrame(index);
  if (!drawn) {
    // The frame request blocks server-side until the worker reaches it, so a
    // failure here means it gave up waiting rather than that it answered
    // instantly. Back off anyway, so a server that does start answering
    // quickly (an error, a restart) cannot turn this into a spin.
    for (let attempt = 0; attempt < 30 && state.frame === index
         && generation === state.generation; attempt++) {
      if (await showFrame(index)) break;
      await new Promise((resolve) => setTimeout(resolve, 500));
    }
  }
  if (generation === state.generation) refreshPixel();
}

/* ---------- pixel inspection ---------- */

function canvasPixel(event) {
  // The canvas element box is exactly the drawn frame -- the script sizes it in
  // CSS pixels from the bitmap -- so this is a plain proportion, at any zoom.
  // The answer is always in the *render's* pixel grid, which is what makes a
  // zoomed-in click name the same pixel a fitted one would.
  const rect = canvas.getBoundingClientRect();
  const x = Math.floor((event.clientX - rect.left) / rect.width * canvas.width);
  const y = Math.floor((event.clientY - rect.top) / rect.height * canvas.height);
  return { x: Math.max(0, Math.min(x, canvas.width - 1)),
           y: Math.max(0, Math.min(y, canvas.height - 1)) };
}

function readPixel(x, y) {
  try {
    return Array.from(ctx.getImageData(x, y, 1, 1).data);
  } catch (err) {
    return null;
  }
}

function showPixelColour(x, y) {
  el("x-input").value = String(x);
  el("y-input").value = String(y);
  const rgba = readPixel(x, y);
  if (!rgba) { el("rgba").textContent = "—"; return; }
  const [r, g, b, a] = rgba;
  el("swatch").style.background = `rgb(${r} ${g} ${b})`;
  el("rgba").textContent = `rgba(${r}, ${g}, ${b}, ${a})`;
}

async function inspect(x, y) {
  if (state.switching) return;
  const generation = state.generation;
  state.pixel = { x, y };
  showPixelColour(x, y);
  const target = el("fragments");
  const frame = state.frame;
  const epoch = state.epoch;
  const request = ++state.pixelRequest;
  const isCurrent = () => generation === state.generation && request === state.pixelRequest
    && frame === state.frame && epoch === state.epoch
    && state.pixel.x === x && state.pixel.y === y;
  // ``/api/fragments``, not ``/api/pixel``: content blockers ship generic
  // ``/pixel?`` rules that match the path alone, and localhost is not exempt.
  const url = `/api/fragments?frame=${frame}&x=${x}&y=${y}`;
  target.innerHTML = `<p class="empty">Reading fragments…</p>`;
  try {
    let data = await getJSONPatiently(url);
    // The server answers `pending` rather than holding the request open, so a
    // slow inspection is a series of quick polls instead of one socket the
    // browser eventually gives up on. It matters for the *first* inspection of
    // a session, which compiles a GPU kernel variant and takes ~12s idle and
    // considerably longer while frames are still rendering.
    for (let waited = 0; data.pending && waited < 300; waited++) {
      // Abandon the poll if the click or the playhead moved on: the answer
      // being waited for is no longer the one on screen.
      if (!isCurrent()) return;
      target.innerHTML =
        `<p class="empty">Reading fragments… (${waited + 1}s)`
        + `${waited > 3 ? "<br>First inspection compiles a GPU kernel." : ""}</p>`;
      await new Promise((r) => setTimeout(r, 1000));
      if (!isCurrent()) return;
      data = await getJSONPatiently(url);
    }
    if (!isCurrent()) return;
    if (data.pending) {
      target.innerHTML = `<p class="empty">Gave up waiting for this pixel.</p>`;
      return;
    }
    renderFragments(data);
  } catch (err) {
    if (!isCurrent()) return;
    target.innerHTML = `<p class="empty">${escapeHTML(err.message)}</p>`;
  }
}

function refreshPixel() {
  if (el("fragments").children.length) inspect(state.pixel.x, state.pixel.y);
}

function renderFragments(data) {
  const target = el("fragments");
  if (!data.available) {
    target.innerHTML = `<p class="empty">${escapeHTML(data.reason || "no data")}</p>`;
    return;
  }
  if (!data.fragments.length) {
    target.innerHTML = `<p class="empty">Nothing covers this pixel &mdash; background.</p>`;
    return;
  }
  const parts = [`<p class="hint">${data.fragments.length} fragment(s), nearest first`
    + ` &mdash; from ${data.raw_fragments} raw hit(s)</p>`];
  for (const f of data.fragments) {
    const rgb = f.rgb_srgb
      ? `rgb(${f.rgb_srgb.map((v) => Math.round(Math.max(0, Math.min(1, v)) * 255)).join(" ")})`
      : "transparent";
    const rows = [
      ["depth", f.depth.toFixed(5)],
      ["mesh_id", f.mesh_id === null ? "—" : f.mesh_id],
      ["mob", f.mob || "—"],
    ];
    // A circuit has no triangle surface id; it has a circuit index instead.
    if (f.kind === "circuit") {
      rows.push(["circuit", f.circuit], ["border", f.border.toFixed(3)]);
    }
    rows.push(
      ["albedo", f.rgb ? f.rgb.map((v) => v.toFixed(4)).join(", ") : "—"],
      ["opacity", f.opacity === null || f.opacity === undefined
        ? "—" : Number(f.opacity).toFixed(3)],
      ["weight", f.weight.toFixed(4)],
      ["source", f.albedo_source || "—"],
    );
    const flags = ["backface", "sliver", "one_mesh", "opaque"].filter((k) => f[k]);
    if (flags.length) rows.push(["flags", flags.join(", ")]);
    parts.push(
      `<div class="frag"><div class="frag-head">`
      + `<span class="swatch" style="background:${rgb}"></span>`
      + `<strong>#${f.index}</strong> <span class="hint">${escapeHTML(f.kind)}</span></div>`
      + `<dl>${rows.map(([k, v]) =>
          `<dt>${k}</dt><dd>${escapeHTML(String(v))}</dd>`).join("")}</dl></div>`
    );
  }
  target.innerHTML = parts.join("");
}

/* ---------- hierarchy ---------- */

function nodeRow(node) {
  const generation = state.generation;
  const item = document.createElement("li");
  const row = document.createElement("div");
  row.className = "row";

  const arrow = document.createElement("span");
  arrow.className = node.has_children ? "arrow" : "arrow leaf";
  arrow.textContent = "▶";

  const name = document.createElement("span");
  name.className = "name" + (node.spawned ? "" : " unspawned");
  name.textContent = node.label;
  name.title = node.spawned ? node.label : `${node.label} (never spawned)`;

  row.append(arrow, name);
  if (node.kind !== "mob") {
    const badge = document.createElement("span");
    badge.className = "badge";
    badge.textContent = node.kind;
    row.append(badge);
  }
  item.append(row);

  const children = document.createElement("ul");
  children.hidden = true;
  item.append(children);

  let loaded = false;
  let loading = false;
  arrow.onclick = async () => {
    if (!node.has_children || generation !== state.generation) return;
    children.hidden = !children.hidden;
    arrow.textContent = children.hidden ? "▶" : "▼";
    if (loaded || loading || children.hidden) return;
    loading = true;
    const components = el("show-components").checked ? 1 : 0;
    try {
      const data = await getJSONPatiently(
        `/api/children?node=${node.node}&components=${components}`);
      if (generation !== state.generation) return;
      children.replaceChildren(...data.children.map(nodeRow));
      loaded = true;
    } catch (err) {
      if (generation !== state.generation) return;
      const notice = document.createElement("li");
      notice.className = "empty";
      notice.textContent = `${err.message} — collapse and expand to retry.`;
      children.replaceChildren(notice);
    } finally {
      loading = false;
    }
  };
  name.onclick = () => {
    if (generation === state.generation) selectNode(node, name);
  };
  return item;
}

async function selectNode(node, element) {
  document.querySelectorAll(".name.selected")
    .forEach((n) => n.classList.remove("selected"));
  element.classList.add("selected");
  state.selected = node.node;
  await showAttributes();
}

async function showAttributes() {
  const generation = state.generation;
  const target = el("attrs");
  const selected = state.selected;
  const frame = state.frame;
  const epoch = state.epoch;
  const request = ++state.attributeRequest;
  const isCurrent = () => generation === state.generation && request === state.attributeRequest
    && selected === state.selected && frame === state.frame && epoch === state.epoch;
  if (state.selected === null) {
    target.innerHTML = `<p class="empty">Select a mob in the hierarchy.</p>`;
    return;
  }
  target.innerHTML = `<p class="empty">Reading…</p>`;
  try {
    const data = await getJSONPatiently(
      `/api/attrs?node=${selected}&frame=${frame}`);
    if (!isCurrent()) return;
    const rows = data.attributes.map((a) => {
      let value = `<span class="note">${escapeHTML(a.note || "—")}</span>`;
      if (a.value) {
        const numbers = a.value.map((v) => Number(v).toFixed(3));
        value = a.channels
          ? a.channels.map((c, i) => `${c} ${numbers[i]}`).join("<br>")
          : numbers.join(", ");
        if (a.note) value += `<br><span class="note">${escapeHTML(a.note)}</span>`;
      }
      return `<tr><th>${escapeHTML(a.name)}</th><td class="num-cell">${value}</td></tr>`;
    });
    target.innerHTML =
      `<p class="hint">${escapeHTML(data.label)} at t=${(data.at ?? 0).toFixed(3)}s</p>`
      + `<table>${rows.join("")}</table>`;
  } catch (err) {
    if (!isCurrent()) return;
    target.innerHTML = `<p class="empty">${escapeHTML(err.message)}</p>`;
  }
}

/* Load the tree, and never throw doing it.
 *
 * This used to be the page's single point of failure: it ran once, from a
 * `Promise.all` in `start()`, so one request lost to a busy render rejected the
 * whole bootstrap -- which also meant the state poll below it never started.
 * The page then sat empty for the life of the tab, with no tree, no scrubber
 * and no way back. Now a failure leaves the tree empty and `refreshState` asks
 * again a second later, which self-heals as soon as the server answers.
 */
let hierarchyPending = null;

async function loadHierarchy() {
  const generation = state.generation;
  if (hierarchyPending === generation) return;
  hierarchyPending = generation;
  const tree = el("tree");
  try {
    const data = await getJSON("/api/hierarchy");
    if (generation !== state.generation) return;
    tree.replaceChildren(...data.roots.map(nodeRow));
  } catch (err) {
    if (generation === state.generation) tree.replaceChildren();
  } finally {
    if (hierarchyPending === generation) hierarchyPending = null;
  }
}

let transcriptLoaded = false;
let transcriptPending = null;

async function loadTranscript() {
  const generation = state.generation;
  if (transcriptLoaded || transcriptPending === generation) return;
  transcriptPending = generation;
  try {
    const data = await getJSON("/api/transcript");
    if (generation !== state.generation) return;
    transcriptView.setData(data);
    transcriptLoaded = true;
    transcriptView.update(state.frame / state.fps, true);
  } catch (err) {
    if (generation === state.generation) {
      el("transcript-status").textContent = `Transcript unavailable; retrying… ${err.message}`;
    }
  } finally {
    if (transcriptPending === generation) transcriptPending = null;
  }
}

/* ---------- project scenes ---------- */

function syncScenes(data) {
  if (!data.scenes) return;
  const tabs = el("scene-tabs");
  const keys = JSON.stringify(data.scenes);
  tabs.hidden = false;
  if (keys !== state.sceneKeys) {
    state.sceneKeys = keys;
    tabs.replaceChildren();
    for (const scene of data.scenes) {
      const button = document.createElement("button");
      button.id = `scene-tab-${scene.id}`;
      button.type = "button";
      button.textContent = scene.name;
      button.dataset.scene = String(scene.id);
      button.setAttribute("role", "tab");
      button.setAttribute("aria-controls", "scene-panel");
      button.onclick = () => changeScene(scene.id);
      button.onkeydown = (event) => {
        const index = data.scenes.findIndex((entry) => entry.id === scene.id);
        let next;
        if (event.key === "Home") next = 0;
        else if (event.key === "End") next = data.scenes.length - 1;
        else if (event.key === "ArrowRight") next = (index + 1) % data.scenes.length;
        else if (event.key === "ArrowLeft") next = (index + data.scenes.length - 1) % data.scenes.length;
        else return;
        event.preventDefault();
        tabs.children[next].focus();
        changeScene(data.scenes[next].id);
      };
      tabs.append(button);
    }
    el("scene-panel").setAttribute("role", "tabpanel");
  }
  for (const button of tabs.children) {
    const selected = Number(button.dataset.scene) === data.scene_id;
    button.setAttribute("aria-selected", String(selected));
    button.tabIndex = selected ? 0 : -1;
  }
  el("scene-panel").setAttribute("aria-labelledby", `scene-tab-${data.scene_id}`);
}

function clearScene() {
  stop();
  state.generation++;
  state.images.clear();
  state.drawn = false;
  state.frame = 0;
  state.selected = null;
  state.pixel = { x: 0, y: 0 };
  state.resolutionKeys = "";
  state.pixelRequest++;
  state.attributeRequest++;
  hierarchyPending = null;
  transcriptPending = null;
  transcriptLoaded = false;
  el("tree").replaceChildren();
  el("fragments").replaceChildren();
  el("attrs").innerHTML = '<p class="empty">Select a mob in the hierarchy.</p>';
  el("cached").style.width = "0%";
  el("x-input").value = el("y-input").value = "0";
  el("rgba").textContent = "—";
  el("swatch").style.background = "transparent";
  // Resizing clears the previous bitmap immediately, including during a slow
  // handoff. Never leave another scene's picture under the newly selected tab.
  canvas.width = state.width;
  canvas.height = state.height;
  state.zoom = 1;
  stage.scrollLeft = stage.scrollTop = 0;
  transcriptView.setData({ blocks: [] });
  el("transcript-status").textContent = "Loading transcript…";
  updateReadouts();
  applyZoom();
}

async function changeScene(id) {
  if (state.switching || id === state.sceneId) return;
  state.switching = true;
  clearScene();
  el("scene-panel").inert = true;
  el("scene-panel").setAttribute("aria-busy", "true");
  for (const button of el("scene-tabs").children) button.disabled = true;
  setStatus("switching scene…", "busy");
  try {
    const response = await fetch(api(`/api/scene?id=${encodeURIComponent(id)}`), { method: "POST" });
    if (!response.ok) throw new Error((await response.json()).error || response.statusText);
    adoptState(await response.json());
    setStatus("rendering…", "busy");
  } catch (err) {
    setStatus(err.message, "error");
  } finally {
    state.switching = false;
    el("scene-panel").inert = false;
    el("scene-panel").removeAttribute("aria-busy");
    for (const button of el("scene-tabs").children) {
      button.disabled = false;
      // Disabling the focused button during the handoff removes focus in real
      // browsers. Restore it so the next arrow key still navigates scene tabs.
      if (Number(button.dataset.scene) === state.sceneId) button.focus();
    }
    // The next state poll also recovers a lost selection response.
    loadHierarchy();
    loadTranscript();
    showFrame(state.frame);
  }
}

/* ---------- resolution ---------- */

/* Rebuild the picker only when the offered set actually changes, so the
 * once-a-second poll cannot reset a menu the user has open. */
function syncResolution(data) {
  const options = data.resolution_options || [];
  const keys = JSON.stringify(options);
  const select = el("resolution");
  if (keys !== state.resolutionKeys) {
    state.resolutionKeys = keys;
    select.innerHTML = "";
    for (const option of options) {
      const node = document.createElement("option");
      node.value = option.name;
      node.textContent = option.label;
      select.append(node);
    }
  }
  state.resolutionName = data.resolution_name;
  if (document.activeElement !== select) select.value = data.resolution_name;
}

async function changeResolution(name) {
  if (state.switching) return;
  const generation = state.generation;
  const select = el("resolution");
  select.disabled = true;
  setStatus("re-rendering at the new resolution…", "busy");
  try {
    const response = await fetch(api(`/api/resolution?name=${encodeURIComponent(name)}`),
                                 { method: "POST" });
    if (!response.ok) throw new Error((await response.json()).error || response.statusText);
    const data = await response.json();
    if (generation !== state.generation) return;
    // Everything on screen is the old size: drop the decoded frames, forget the
    // pixel that was inspected, and let the epoch in the URL keep the browser
    // from handing back a cached PNG of the wrong shape.
    state.images.clear();
    state.drawn = false;
    state.epoch = data.epoch;
    el("fragments").innerHTML = "";
    adoptState(data);
    await showFrame(Math.min(state.frame, data.total_frames - 1));
    if (generation === state.generation) setStatus("");
  } catch (err) {
    if (generation === state.generation) setStatus(err.message, "error");
  } finally {
    select.disabled = false;
  }
}

/* ---------- wiring ---------- */

function escapeHTML(text) {
  const div = document.createElement("div");
  div.textContent = text;
  return div.innerHTML;
}

function onSubmit(input, handler) {
  input.addEventListener("keydown", (event) => {
    if (event.key !== "Enter") return;
    event.preventDefault();
    const value = Number(input.value);
    if (Number.isFinite(value)) handler(value);
    input.blur();
  });
}

function scrubTo(event) {
  const rect = el("scrub").getBoundingClientRect();
  const fraction = Math.max(0, Math.min(1, (event.clientX - rect.left) / rect.width));
  seek(fraction * (state.totalFrames - 1));
}

function adoptState(data) {
  if (data.scene_version !== undefined && data.scene_version !== state.sceneVersion) {
    state.sceneVersion = data.scene_version;
    state.sceneId = data.scene_id;
    clearScene();
    state.epoch = data.epoch;
  }
  syncScenes(data);
  state.fps = data.fps;
  state.totalFrames = data.total_frames;
  state.duration = data.runtime ?? data.duration ?? 0;
  state.width = data.width;
  state.height = data.height;
  el("meta").textContent =
    `${data.width}×${data.height} · ${data.fps} fps · `
    + `${data.total_frames} frames · ${state.duration.toFixed(2)}s`;
  syncResolution(data);
  updateReadouts();
}

async function refreshState() {
  if (state.switching) return;
  const generation = state.generation;
  try {
    const data = await getJSON("/api/state");
    if (state.switching || generation !== state.generation) return;
    if (data.epoch !== state.epoch) {
      // Something else changed the resolution (another tab, a restart): drop
      // frames of the old size rather than drawing them at the new one.
      state.epoch = data.epoch;
      state.images.clear();
      state.drawn = false;
    }
    adoptState(data);
    const covered = data.cached.reduce((sum, [a, b]) => sum + (b - a + 1), 0);
    el("cached").style.width =
      `${(covered / Math.max(1, data.total_frames)) * 100}%`;
    if (data.error) setStatus(data.error, "error");
    // The server is answering, so anything the bootstrap failed to get is
    // worth asking for again. Both are cheap no-ops once they have landed:
    // the tree guards on its own flag, and a frame request already in flight
    // is shared rather than reissued.
    if (!el("tree").children.length) loadHierarchy();
    if (!transcriptLoaded) loadTranscript();
    if (!state.drawn) showFrame(state.frame);
  } catch (err) {
    if (!state.switching && generation === state.generation) setStatus(err.message, "error");
  }
}

el("play").onclick = () => (state.playing ? stop() : play());
el("scrub").onclick = scrubTo;
el("scrub").onkeydown = (event) => {
  if (event.key === "ArrowRight") seek(state.frame + 1);
  if (event.key === "ArrowLeft") seek(state.frame - 1);
};
onSubmit(el("time-input"), (seconds) => seek(seconds * state.fps));
onSubmit(el("frame-input"), (index) => seek(index));
onSubmit(el("x-input"), () => inspect(
  Number(el("x-input").value) | 0, Number(el("y-input").value) | 0));
onSubmit(el("y-input"), () => inspect(
  Number(el("x-input").value) | 0, Number(el("y-input").value) | 0));

canvas.addEventListener("mousemove", (event) => {
  const { x, y } = canvasPixel(event);
  showPixelColour(x, y);
});
canvas.addEventListener("mouseleave", () =>
  showPixelColour(state.pixel.x, state.pixel.y));
canvas.addEventListener("click", (event) => {
  const { x, y } = canvasPixel(event);
  inspect(x, y);
});
el("resolution").onchange = (event) => changeResolution(event.target.value);
el("zoom").oninput = (event) => setZoom(Number(event.target.value) / 100, null);
el("zoom-fit").onclick = () => setZoom(1, null);

// Passive listeners cannot preventDefault, and without that a wheel over the
// stage scrolls the pane instead of zooming.
stage.addEventListener("wheel", (event) => {
  if (event.ctrlKey) return;  // leave the browser's own page zoom alone
  event.preventDefault();
  const step = event.deltaY < 0 ? 1.15 : 1 / 1.15;
  setZoom(state.zoom * step, event);
}, { passive: false });

// The fit scale depends on the stage's size, so it has to be recomputed when
// the window changes shape.
window.addEventListener("resize", applyZoom);

el("show-components").onchange = () => {
  el("tree").innerHTML = "";
  loadHierarchy();
};

(function start() {
  applyZoom();
  // The poll is armed first and unconditionally. It is what refreshes the
  // scrubber, the cache bar and any error the worker reports, and it is also
  // what retries the two loads below -- so nothing here may be able to stop it
  // being scheduled.
  setInterval(refreshState, 1000);
  // Discover the selected scene/version before any scene-specific request.
  refreshState();
})();
