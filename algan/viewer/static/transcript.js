/* A transcript is a read-only snapshot. The player owns time and seeking;
 * this view only maps its clock to words, without a per-frame HTTP request. */
"use strict";

class TranscriptView {
  constructor(seekSeconds) {
    this.seekSeconds = seekSeconds;
    this.cues = [];
    this.prefixEnds = [];
    this.active = new Set();
    this.anchor = null;
    this.time = 0;
    this.tab = "fragments";
    this.panel = document.getElementById("transcript-panel");
    for (const name of ["fragments", "transcript"]) {
      const button = document.getElementById(`${name}-tab`);
      button.onclick = () => this.selectTab(name);
      button.onkeydown = (event) => {
        let next;
        if (event.key === "Home") next = "fragments";
        else if (event.key === "End") next = "transcript";
        else if (event.key === "ArrowLeft" || event.key === "ArrowRight") {
          next = name === "fragments" ? "transcript" : "fragments";
        } else return;
        event.preventDefault();
        this.selectTab(next);
        document.getElementById(`${next}-tab`).focus();
      };
    }
  }

  selectTab(name) {
    this.tab = name;
    for (const tab of ["fragments", "transcript"]) {
      const selected = tab === name;
      const button = document.getElementById(`${tab}-tab`);
      button.setAttribute("aria-selected", String(selected));
      button.tabIndex = selected ? 0 : -1;
      document.getElementById(`${tab}-panel`).hidden = !selected;
    }
    if (name === "transcript") this.update(this.time, true);
  }

  setData(data) {
    const container = document.getElementById("transcript");
    container.replaceChildren();
    this.cues = [];
    this.active = new Set();
    this.anchor = null;
    let estimated = false, unavailable = false;
    for (const block of data.blocks) {
      const paragraph = document.createElement("p");
      paragraph.className = "transcript-block";
      // Python offsets count Unicode code points, not JavaScript UTF-16 units.
      const characters = Array.from(block.text);
      let offset = 0;
      estimated ||= block.timing === "estimated";
      unavailable ||= block.timing === "unavailable";
      for (const word of block.words) {
        paragraph.append(document.createTextNode(characters.slice(offset, word.offset).join("")));
        if (Number.isFinite(word.start) && Number.isFinite(word.end) && word.end > word.start) {
          const button = document.createElement("button");
          button.type = "button";
          button.className = "transcript-word";
          button.textContent = word.text;
          button.title = `Seek to ${word.start.toFixed(3)} s`
            + (block.timing === "estimated" ? " (estimated word timing)" : "");
          button.onclick = () => this.seekSeconds(word.start);
          paragraph.append(button);
          this.cues.push({ start: word.start, end: word.end, element: button });
        } else {
          paragraph.append(document.createTextNode(word.text));
        }
        offset = word.end_offset;
      }
      paragraph.append(document.createTextNode(characters.slice(offset).join("")));
      container.append(paragraph);
    }
    // Display stays in authoring order. The lookup is separately sorted so
    // Sync/Lag blocks, overlaps and backwards seeks still select the right word.
    this.cues.sort((a, b) => a.start - b.start);
    let end = -Infinity;
    this.prefixEnds = this.cues.map((cue) => (end = Math.max(end, cue.end)));
    const status = document.getElementById("transcript-status");
    if (!data.blocks.length) {
      status.textContent = "No Speech blocks were recorded before Scene.view().";
    } else {
      status.textContent = "Click a word to seek. The highlight follows the playhead."
        + (estimated ? " Some word timings are estimated because the speech source supplied no usable alignment." : "")
        + (unavailable ? " Text without playable audio is shown without seek links." : "");
    }
    this.update(this.time, true);
  }

  update(time, forceScroll = false) {
    this.time = time;
    // Upper bound: the latest word that has started at this scene time.
    let lo = 0, hi = this.cues.length;
    while (lo < hi) {
      const mid = (lo + hi) >>> 1;
      if (this.cues[mid].start <= time) lo = mid + 1;
      else hi = mid;
    }
    const active = new Set();
    let anchor = null;
    // The prefix maximum lets us stop before unrelated earlier speech. More
    // than one word can be audible when Speech contexts overlap; highlight all.
    for (let i = lo - 1; i >= 0 && this.prefixEnds[i] > time; i--) {
      const cue = this.cues[i];
      if (cue.end > time) {
        active.add(cue.element);
        if (!anchor) anchor = cue.element;
      }
    }
    for (const element of this.active) {
      if (!active.has(element)) {
        element.classList.remove("current");
        element.removeAttribute("aria-current");
      }
    }
    for (const element of active) {
      if (!this.active.has(element)) {
        element.classList.add("current");
        element.setAttribute("aria-current", "true");
      }
    }
    this.active = active;
    // During silence keep the nearest preceding word in view, but do not
    // pretend it is still being spoken. Before narration, show its first word.
    anchor ||= this.cues[Math.max(0, lo - 1)]?.element;
    if (this.tab === "transcript" && anchor && (forceScroll || anchor !== this.anchor)) {
      const word = anchor.getBoundingClientRect();
      const panel = this.panel.getBoundingClientRect();
      if (word.top < panel.top + 24 || word.bottom > panel.bottom - 24) {
        // Scroll only this panel, never the whole page or the video stage.
        this.panel.scrollTo({
          top: Math.max(0, this.panel.scrollTop + word.top - panel.top
            - this.panel.clientHeight / 2 + word.height / 2),
          behavior: "auto",
        });
      }
    }
    this.anchor = anchor;
  }
}
