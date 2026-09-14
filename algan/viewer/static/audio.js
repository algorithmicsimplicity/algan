/* One decoded scene mix; its audio clock also drives the picture. */
"use strict";

class ViewerAudio {
  constructor() {
    this.context = null;
    this.reset();
  }

  prepare(url) {
    if (url !== this.url) { this.reset(); this.url = url; }
    const Context = window.AudioContext || window.webkitAudioContext;
    if (!Context) return Promise.reject(new Error("This browser does not support audio playback"));
    if (!this.context) this.context = new Context();
    // Invoke resume synchronously in the Play gesture, before fetching or
    // waiting for any frames, so browser autoplay policy cannot block it.
    const resumed = this.context.resume();
    if (!this.pending) {
      const pending = fetch(url).then(async response => {
        if (!response.ok) throw new Error((await response.json()).error || response.statusText);
        return this.context.decodeAudioData(await response.arrayBuffer());
      }).then(buffer => {
        if (this.pending === pending) this.buffer = buffer;
      }).catch(error => {
        if (this.pending === pending) this.pending = null;
        throw error;
      });
      this.pending = pending;
    }
    return Promise.all([resumed, this.pending]);
  }

  start(seconds) {
    this.pause();
    this.offset = seconds;
    this.startedAt = this.context.currentTime;
    this.running = true;
    if (seconds < this.buffer.duration) {
      const source = this.context.createBufferSource();
      source.buffer = this.buffer;
      source.connect(this.context.destination);
      source.start(0, seconds);
      this.source = source;
    }
  }

  time() {
    return this.offset + (this.running ? this.context.currentTime - this.startedAt : 0);
  }

  pause() {
    if (this.running) this.offset = this.time();
    this.running = false;
    if (this.source) {
      this.source.stop();
      this.source.disconnect();
      this.source = null;
    }
  }

  reset() {
    this.pause();
    this.buffer = null;
    this.pending = null;
    this.url = null;
    this.offset = 0;
  }
}
