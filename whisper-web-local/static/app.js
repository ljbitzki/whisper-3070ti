'use strict';

const $ = id => document.getElementById(id);
const configIds = ['model', 'language', 'temperature', 'beam_size', 'best_of', 'no_speech_thold', 'prompt', 'chunk_seconds'];
let capturing = false;
let starting = false;
let recordingStream = null;
let audioContext = null;
let audioSource = null;
let worklet = null;
let captureParts = [];
let captureLength = 0;
let jobs = Promise.resolve();
let pending = 0;
let cumulativeSeconds = 0;
let srtSegments = [];
let flushResolver = null;
let sourceName = 'transcricao';
let settingsSnapshot = null;
let fileBusy = false;

function status(message, error = false) {
  $('status').textContent = message;
  $('status').classList.toggle('error', error);
}

function lockControls(locked) {
  for (const id of configIds) $(id).disabled = locked;
  $('transcribe-file').disabled = locked;
  $('audio-file').disabled = locked;
  $('tab-file').disabled = locked;
  $('tab-mic').disabled = locked;
}

async function fetchJSON(path, options = {}) {
  let response;
  try {
    response = await fetch(path, options);
  } catch (err) {
    throw new Error('Falha de conexão: ' + err.message);
  }
  let data;
  try {
    data = await response.json();
  } catch (_) {
    throw new Error('Resposta inválida do servidor (' + response.status + ')');
  }
  if (!response.ok) {
    const detail = typeof data.detail === 'string' ? data.detail : JSON.stringify(data.detail || data);
    throw new Error(detail || ('Erro HTTP ' + response.status));
  }
  return data;
}

function optionsSnapshot() {
  return Object.fromEntries(['model', 'language', 'temperature', 'beam_size', 'best_of', 'no_speech_thold', 'prompt']
    .map(id => [id, $(id).value]));
}

function dataForFile(blob, filename, settings) {
  const form = new FormData();
  form.append('file', blob, filename);
  for (const [key, value] of Object.entries(settings)) form.append(key, value);
  return form;
}

function setResult(text, segments, fromChunk = false, offset = 0) {
  const current = $('result').value;
  const cleaned = (text || '').trim();
  if (fromChunk) {
    if (cleaned) $('result').value = current + (current.trim() ? '\n' : '') + cleaned;
  } else {
    $('result').value = cleaned;
    srtSegments = [];
  }
  for (const item of segments || []) {
    const value = (item.text || '').trim();
    if (value) srtSegments.push({
      start: Number(item.start || 0) + offset,
      end: Number(item.end || 0) + offset,
      text: value,
    });
  }
  $('stats').textContent = $('result').value.length + ' caracteres · ' + srtSegments.length + ' segmentos';
  $('result').scrollTop = $('result').scrollHeight;
}

async function loadModels() {
  try {
    const state = await fetchJSON('/api/status');
    const menu = $('model');
    menu.replaceChildren();
    for (const name of state.models) {
      const option = document.createElement('option');
      option.value = name;
      option.textContent = name.replace(/^ggml-/, '').replace(/\.bin$/, '');
      if (name === state.active_model) option.selected = true;
      menu.append(option);
    }
    $('backend-state').textContent = state.online ? '● CUDA · conectado' : 'Offline';
    if (!state.online || !state.models.length) status('Nenhum servidor/modelo ativo.', true);
  } catch (error) {
    $('backend-state').textContent = '● Desconectado';
    status(error.message, true);
  }
}

$('tab-file').addEventListener('click', () => changeTab('file'));
$('tab-mic').addEventListener('click', () => changeTab('mic'));
function changeTab(which) {
  $('file-area').hidden = which !== 'file';
  $('mic-area').hidden = which !== 'mic';
  $('tab-file').classList.toggle('selected', which === 'file');
  $('tab-mic').classList.toggle('selected', which === 'mic');
}

$('transcribe-file').addEventListener('click', async () => {
  const file = $('audio-file').files[0];
  if (!file) { status('Selecione um arquivo primeiro.', true); return; }
  if (file.size > 300 * 1024 * 1024) { status('Máximo permitido: 300 MB.', true); return; }
  fileBusy = true;
  lockControls(true);
  sourceName = file.name.replace(/\.[^/.]+$/, '') || 'transcricao';
  status('Convertendo e transcrevendo o arquivo... (pode levar algum tempo)');
  const start = performance.now();
  try {
    const result = await fetchJSON('/api/transcribe', {
      method: 'POST', body: dataForFile(file, file.name, optionsSnapshot()),
    });
    setResult(result.text, result.segments);
    status('Transcrição concluída em ' + ((performance.now() - start) / 1000).toFixed(1) + ' s · ' + result.model);
  } catch (error) { status(error.message, true); }
  finally { fileBusy = false; lockControls(false); }
});

function encodeWAV(audio, inputRate) {
  const outputRate = 16000;
  const count = Math.max(1, Math.round(audio.length * outputRate / inputRate));
  const bytes = new ArrayBuffer(44 + count * 2);
  const view = new DataView(bytes);
  const writeAscii = (position, value) => {
    for (let i = 0; i < value.length; i++) view.setUint8(position + i, value.charCodeAt(i));
  };
  writeAscii(0, 'RIFF'); view.setUint32(4, 36 + count * 2, true);
  writeAscii(8, 'WAVE'); writeAscii(12, 'fmt ');
  view.setUint32(16, 16, true); view.setUint16(20, 1, true);
  view.setUint16(22, 1, true); view.setUint32(24, outputRate, true);
  view.setUint32(28, outputRate * 2, true); view.setUint16(32, 2, true);
  view.setUint16(34, 16, true); writeAscii(36, 'data');
  view.setUint32(40, count * 2, true);
  for (let i = 0; i < count; i++) {
    const origin = i * inputRate / outputRate;
    const left = Math.floor(origin);
    const fraction = origin - left;
    const x = audio[Math.min(left, audio.length - 1)] || 0;
    const y = audio[Math.min(left + 1, audio.length - 1)] || 0;
    const sample = Math.max(-1, Math.min(1, x + (y - x) * fraction));
    view.setInt16(44 + 2 * i, sample < 0 ? sample * 32768 : sample * 32767, true);
  }
  return new Blob([bytes], {type: 'audio/wav'});
}

function removeSamples(count) {
  const result = new Float32Array(count);
  let cursor = 0;
  while (cursor < count && captureParts.length) {
    const next = captureParts[0];
    const take = Math.min(next.length, count - cursor);
    result.set(next.subarray(0, take), cursor);
    cursor += take;
    if (take === next.length) captureParts.shift();
    else captureParts[0] = next.subarray(take);
  }
  captureLength -= count;
  return result;
}

function handleSamples(frames) {
  captureParts.push(frames);
  captureLength += frames.length;
  const chunkFrames = Math.round(Number($('chunk_seconds').value) * audioContext.sampleRate);
  while (captureLength >= chunkFrames) enqueueAudio(removeSamples(chunkFrames));
}

function enqueueAudio(samples) {
  const rate = audioContext.sampleRate;
  const duration = samples.length / rate;
  const begin = cumulativeSeconds;
  cumulativeSeconds += duration;
  const wav = encodeWAV(samples, rate);
  const settings = settingsSnapshot;
  pending++;
  jobs = jobs.then(async () => {
    status('Processando bloco do microfone · ' + pending + ' na fila...');
    const result = await fetchJSON('/api/transcribe', {
      method: 'POST', body: dataForFile(wav, 'microfone.wav', settings),
    });
    setResult(result.text, result.segments, true, begin);
  }).catch(error => status('Erro em um bloco: ' + error.message, true))
    .finally(() => {
      pending--;
      if (capturing && pending === 0) status('Ouvindo microfone · aguardando o próximo bloco...');
    });
}

$('start-mic').addEventListener('click', async () => {
  if (capturing || starting || fileBusy) return;
  starting = true;
  $('start-mic').disabled = true;
  status('Solicitando acesso ao microfone...');
  try {
    if (!navigator.mediaDevices || !window.AudioWorkletNode) {
      throw new Error('Navegador sem suporte a AudioWorklet ou captura de microfone. Use Chrome/LibreWolf em localhost.');
    }
    recordingStream = await navigator.mediaDevices.getUserMedia({audio: {
      echoCancellation: true, noiseSuppression: true, autoGainControl: true,
    }, video: false});
    audioContext = new AudioContext();
    await audioContext.audioWorklet.addModule('/static/mic-worklet.js');
    audioSource = audioContext.createMediaStreamSource(recordingStream);
    worklet = new AudioWorkletNode(audioContext, 'mic-collector');
    worklet.port.onmessage = event => {
      if (event.data && event.data.flushed) {
        if (flushResolver) { flushResolver(); flushResolver = null; }
      } else if (event.data instanceof Float32Array) {
        handleSamples(event.data);
      }
    };
    // Inicialize a fila ANTES de ativar o pipeline de áudio, evitando perder o primeiro bloco.
    captureParts = []; captureLength = 0; cumulativeSeconds = 0;
    jobs = Promise.resolve(); pending = 0; srtSegments = [];
    settingsSnapshot = optionsSnapshot();
    sourceName = 'microfone';
    setResult('', []);
    audioSource.connect(worklet);
    worklet.connect(audioContext.destination);
    await audioContext.resume();
    capturing = true;
    lockControls(true);
    $('start-mic').disabled = true;
    $('stop-mic').disabled = false;
    $('record-indicator').hidden = false;
    status('Ouvindo microfone · aguardando primeiro bloco...');
  } catch (error) {
    status('Falha ao iniciar microfone: ' + error.message, true);
    await cleanupMic();
    $('start-mic').disabled = false;
  } finally { starting = false; }
});

async function cleanupMic() {
  if (worklet) { worklet.disconnect(); worklet.port.onmessage = null; worklet = null; }
  if (audioSource) { audioSource.disconnect(); audioSource = null; }
  if (recordingStream) { recordingStream.getTracks().forEach(track => track.stop()); recordingStream = null; }
  if (audioContext) { await audioContext.close(); audioContext = null; }
}

$('stop-mic').addEventListener('click', async () => {
  if (!capturing) return;
  $('stop-mic').disabled = true;
  status('Encerrando gravação e processando os últimos blocos...');
  // Recupera as amostras restantes do AudioWorklet antes de fechá-lo.
  if (worklet) {
    await new Promise(resolve => {
      const timeout = setTimeout(() => { flushResolver = null; resolve(); }, 1500);
      flushResolver = () => { clearTimeout(timeout); resolve(); };
      worklet.port.postMessage('flush');
    });
  }
  if (captureLength > (audioContext?.sampleRate || 16000) * 0.5) {
    enqueueAudio(removeSamples(captureLength));
  }
  capturing = false;
  $('record-indicator').hidden = true;
  await cleanupMic();
  await jobs;
  lockControls(false);
  $('start-mic').disabled = false;
  if (!$('status').classList.contains('error')) status('Gravação finalizada. Transcrição disponível.');
});

$('copy').addEventListener('click', async () => {
  try { await navigator.clipboard.writeText($('result').value); status('Texto copiado.'); }
  catch (_) { $('result').select(); document.execCommand('copy'); status('Texto copiado.'); }
});

function timestamp(seconds) {
  const total = Math.max(0, Math.round(seconds * 1000));
  const hours = Math.floor(total / 3600000);
  const minutes = Math.floor(total % 3600000 / 60000);
  const secs = Math.floor(total % 60000 / 1000);
  const milliseconds = total % 1000;
  return `${String(hours).padStart(2,'0')}:${String(minutes).padStart(2,'0')}:${String(secs).padStart(2,'0')},${String(milliseconds).padStart(3,'0')}`;
}
function download(text, extension, mime) {
  const data = new Blob([text], {type: mime});
  const url = URL.createObjectURL(data);
  const link = document.createElement('a');
  link.href = url;
  link.download = sourceName + '.' + extension;
  document.body.append(link);
  link.click(); link.remove();
  setTimeout(() => URL.revokeObjectURL(url), 1500);
}
$('save-txt').addEventListener('click', () => download($('result').value, 'txt', 'text/plain;charset=utf-8'));
$('save-srt').addEventListener('click', () => {
  if (!srtSegments.length) { status('Nenhum segmento com timestamps para gerar SRT.', true); return; }
  const content = srtSegments.map((s, i) => `${i+1}\n${timestamp(s.start)} --> ${timestamp(s.end)}\n${s.text}\n`).join('\n');
  download(content, 'srt', 'application/x-subrip;charset=utf-8');
});
$('clear').addEventListener('click', () => {
  if (capturing) { status('Pare o microfone antes de limpar a transcrição.', true); return; }
  setResult('', []); status('Transcrição limpa.');
});
loadModels();
