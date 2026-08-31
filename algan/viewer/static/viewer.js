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
  frame: 0, playing: false, selected: null,
  pixel: { x: 0, y: 0 },
  images: new Map(),
  playStartedAt: 0, playStartedFrame: 0,
};

const el = (id) => document.getElementById(id);
const canvas = el("frame");
const ctx = canvas.getContext("2d", { willReadFrequently: true });

/* ---------- server ---------- */

async function getJSON(url) {
  const response = await fetch(url);
  if (!response.ok) throw new Error((await response.json()).error || response.statusText);
  return response.json();
}

function frameImage(index) {
  if (state.images.has(index)) return state.images.get(index);
  const promise = new Promise((resolve, reject) => {
    const image = new Image();
    image.onload = () => resolve(image);
    image.onerror = () => { state.images.delete(index); reject(new Error("not ready")); };
    image.src = `/frame/${index}.png`;
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
  updateReadouts();
  try {
    const image = await frameImage(index);
    if (state.frame !== index) return true;   // moved on while waiting
    if (canvas.width !== image.width || canvas.height !== image.height) {
      canvas.width = image.width;
      canvas.height = image.height;
    }
    ctx.drawImage(image, 0, 0);
    setStatus("");
    return true;
  } catch (err) {
    if (!redrawOnly) setStatus("rendering…", "busy");
    return false;
  }
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
}

function setStatus(text, kind = "") {
  const node = el("status");
  node.textContent = text;
  node.className = `status ${kind}`;
}

/* ---------- playback ---------- */

function play() {
  if (state.playing) return;
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
  const elapsed = (performance.now() - state.playStartedAt) / 1000;
  const target = state.playStartedFrame + Math.floor(elapsed * state.fps);
  if (target >= state.totalFrames) { stop(); await showFrame(state.totalFrames - 1); return; }
  if (target !== state.frame) {
    const drawn = await showFrame(target);
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
  stop();
  index = Math.max(0, Math.min(Math.round(index), state.totalFrames - 1));
  fetch(`/api/prefetch?frame=${index}`).catch(() => {});
  const drawn = await showFrame(index);
  if (!drawn) {
    // The frame request blocks server-side until the worker reaches it, so a
    // failure here means it gave up waiting rather than that it answered
    // instantly. Back off anyway, so a server that does start answering
    // quickly (an error, a restart) cannot turn this into a spin.
    for (let attempt = 0; attempt < 30 && state.frame === index; attempt++) {
      if (await showFrame(index)) break;
      await new Promise((resolve) => setTimeout(resolve, 500));
    }
  }
  refreshPixel();
}

/* ---------- pixel inspection ---------- */

function canvasPixel(event) {
  // The canvas is laid out with `object-fit: contain`, so its element box is
  // usually taller or wider than the frame drawn inside it. Mapping the mouse
  // through the element box alone would report a pixel offset by the letterbox
  // -- worse the further the window's aspect is from the render's.
  const rect = canvas.getBoundingClientRect();
  const scale = Math.min(rect.width / canvas.width, rect.height / canvas.height);
  const drawnWidth = canvas.width * scale;
  const drawnHeight = canvas.height * scale;
  const left = rect.left + (rect.width - drawnWidth) / 2;
  const top = rect.top + (rect.height - drawnHeight) / 2;
  const x = Math.floor((event.clientX - left) / scale);
  const y = Math.floor((event.clientY - top) / scale);
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
  state.pixel = { x, y };
  showPixelColour(x, y);
  const target = el("fragments");
  target.innerHTML = `<p class="empty">Reading fragments…</p>`;
  try {
    const data = await getJSON(`/api/pixel?frame=${state.frame}&x=${x}&y=${y}`);
    renderFragments(data);
  } catch (err) {
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
  arrow.onclick = async () => {
    if (!node.has_children) return;
    children.hidden = !children.hidden;
    arrow.textContent = children.hidden ? "▶" : "▼";
    if (loaded || children.hidden) return;
    loaded = true;
    const components = el("show-components").checked ? 1 : 0;
    const data = await getJSON(`/api/children?node=${node.node}&components=${components}`);
    for (const child of data.children) children.append(nodeRow(child));
  };
  name.onclick = () => selectNode(node, name);
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
  const target = el("attrs");
  if (state.selected === null) {
    target.innerHTML = `<p class="empty">Select a mob in the hierarchy.</p>`;
    return;
  }
  target.innerHTML = `<p class="empty">Reading…</p>`;
  try {
    const data = await getJSON(`/api/attrs?node=${state.selected}&frame=${state.frame}`);
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
    target.innerHTML = `<p class="empty">${escapeHTML(err.message)}</p>`;
  }
}

async function loadHierarchy() {
  const tree = el("tree");
  tree.innerHTML = "";
  const data = await getJSON("/api/hierarchy");
  for (const node of data.roots) tree.append(nodeRow(node));
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

async function refreshState() {
  try {
    const data = await getJSON("/api/state");
    state.fps = data.fps;
    state.totalFrames = data.total_frames;
    state.duration = data.duration;
    state.width = data.width;
    state.height = data.height;
    el("meta").textContent =
      `${data.width}×${data.height} · ${data.fps} fps · `
      + `${data.total_frames} frames · ${data.duration.toFixed(2)}s`;
    const covered = data.cached.reduce((sum, [a, b]) => sum + (b - a + 1), 0);
    el("cached").style.width =
      `${(covered / Math.max(1, data.total_frames)) * 100}%`;
    if (data.error) setStatus(data.error, "error");
  } catch (err) {
    setStatus(err.message, "error");
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
el("show-components").onchange = loadHierarchy;

(async function start() {
  await refreshState();
  await Promise.all([showFrame(0), loadHierarchy()]);
  setInterval(refreshState, 1000);
})();
