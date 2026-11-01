// Recebe o microfone no próprio navegador, envia blocos PCM à interface.
class MicCollector extends AudioWorkletProcessor {
  constructor() {
    super();
    this.frames = new Float32Array(4096);
    this.position = 0;
  }
  process(inputs, outputs) {
    const channel = inputs[0] && inputs[0][0];
    if (channel) {
      for (let i = 0; i < channel.length; i++) {
        this.frames[this.position++] = channel[i];
        if (this.position === this.frames.length) {
          const ready = this.frames;
          this.port.postMessage(ready, [ready.buffer]);
          this.frames = new Float32Array(4096);
          this.position = 0;
        }
      }
    }
    // Saída silenciosa: não reproduz o som do microfone no alto-falante.
    if (outputs[0]) {
      for (const output of outputs[0]) output.fill(0);
    }
    return true;
  }
}
registerProcessor('mic-collector', MicCollector);
