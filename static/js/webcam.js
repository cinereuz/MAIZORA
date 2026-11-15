// Realtime Webcam Stream & Frame Detection

const REALTIME_INTERVAL_MS = 400; // ~2.5 FPS

export function initWebcam({ showLoading, onResult }) {
  const webcamVideo = document.getElementById('webcam-video');
  const webcamCanvas = document.getElementById('webcam-canvas');
  const realtimeOutput = document.getElementById('realtime-output');
  const realtimeSummary = document.getElementById('realtime-summary');
  const realtimeStatus = document.getElementById('realtime-status');
  const btnStartWebcam = document.getElementById('btn-start-webcam');
  const btnStopWebcam = document.getElementById('btn-stop-webcam');

  if (!webcamVideo || !webcamCanvas || !btnStartWebcam || !btnStopWebcam) return;

  let webcamStream = null;
  let realtimeIntervalId = null;
  let isProcessingFrame = false;
  let lastRealtimeFrame = null;

  function captureStandardizedSquare(videoEl, targetDim = 224, quality = 0.85) {
    const vw = videoEl.videoWidth;
    const vh = videoEl.videoHeight;
    const side = Math.min(vw, vh);
    const sx = Math.floor((vw - side) / 2);
    const sy = Math.floor((vh - side) / 2);
    webcamCanvas.width = targetDim;
    webcamCanvas.height = targetDim;
    const ctx = webcamCanvas.getContext('2d');
    ctx.drawImage(videoEl, sx, sy, side, side, 0, 0, targetDim, targetDim);
    return webcamCanvas.toDataURL('image/jpeg', quality);
  }

  function renderRealtimeResult(data) {
    // Ignore late responses arriving after the camera was stopped.
    if (!webcamStream) return;
    realtimeOutput.src = data.annotated;

    const s = data.summary;
    realtimeStatus.textContent = `Live -- kernels detected: ${s.total_biji}`;
    realtimeStatus.classList.remove('is-error');
    realtimeStatus.classList.add('is-live');

    const totalJagung = (s.sehat || 0) + (s.terkontaminasi || 0);
    const bukanJagung = s.bukan_jagung || 0;
    let text = '';
    if (totalJagung === 0) {
      text = bukanJagung > 0 ? `Not Corn (${bukanJagung} non-corn object(s) detected).` : 'No corn kernels detected in this frame yet.';
    } else if (bukanJagung > totalJagung) {
      text = `Not Corn -- non-corn objects predominate (${bukanJagung} non-corn vs ${totalJagung} corn-like).`;
    } else {
      text = `${s.sehat} healthy, ${s.terkontaminasi} mold-contaminated (${s.persen_kontaminasi}%)`;
      if (bukanJagung > 0) {
        text += ` -- ignored non-corn objects: ${bukanJagung}.`;
      }
    }
    realtimeSummary.textContent = text;
  }

  async function captureAndDetectFrame() {
    // Skip if a previous frame is still in flight to avoid request pile-up.
    if (isProcessingFrame || !webcamVideo.videoWidth) return;

    const frameDataUrl = captureStandardizedSquare(webcamVideo, 224, 0.82);
    lastRealtimeFrame = frameDataUrl;

    isProcessingFrame = true;
    try {
      const res = await fetch('/api/detect/frame', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ image: frameDataUrl }),
      });

      if (!res.ok) {
        realtimeStatus.textContent = 'This frame failed to process, retrying...';
        realtimeStatus.classList.add('is-error');
        return;
      }

      const data = await res.json();
      renderRealtimeResult(data);
    } catch (err) {
      realtimeStatus.textContent = 'Server connection problem...';
      realtimeStatus.classList.add('is-error');
    } finally {
      isProcessingFrame = false;
    }
  }

  function stopWebcamCleanup() {
    if (realtimeIntervalId) {
      clearInterval(realtimeIntervalId);
      realtimeIntervalId = null;
    }
    if (webcamStream) {
      webcamStream.getTracks().forEach((track) => track.stop());
      webcamStream = null;
    }
    webcamVideo.srcObject = null;
    isProcessingFrame = false;
  }

  async function startWebcam() {
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
      alert('The camera can only be used over HTTPS or localhost. Open the app via an https:// address or use one of the solutions in the guide.');
      return;
    }

    try {
      webcamStream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: 'environment' },
      });
    } catch (err) {
      alert('Unable to access the camera: ' + err.message);
      return;
    }

    webcamVideo.srcObject = webcamStream;

    btnStartWebcam.disabled = true;
    btnStopWebcam.disabled = false;
    realtimeStatus.textContent = 'Camera active, starting detection...';
    realtimeStatus.classList.remove('is-live', 'is-error');
    lastRealtimeFrame = null;

    realtimeIntervalId = setInterval(captureAndDetectFrame, REALTIME_INTERVAL_MS);
  }

  async function stopWebcamAndAnalyze() {
    let finalFrame = null;
    if (webcamVideo.videoWidth) {
      finalFrame = captureStandardizedSquare(webcamVideo, 224, 0.92);
    }
    if (!finalFrame && lastRealtimeFrame) {
      finalFrame = lastRealtimeFrame;
    }

    stopWebcamCleanup();

    btnStartWebcam.disabled = false;
    btnStopWebcam.disabled = true;
    realtimeStatus.textContent = finalFrame ? 'Camera stopped — processing the last frame...' : 'Camera stopped';
    realtimeStatus.classList.remove('is-live', 'is-error');

    if (finalFrame) {
      showLoading(true);
      try {
        const res = await fetch('/api/detect/frame/full', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ image: finalFrame }),
        });
        if (res.ok) {
          const data = await res.json();
          onResult(data);
          realtimeStatus.textContent = 'Camera stopped — full analysis results are shown below';
        } else {
          realtimeStatus.textContent = 'Camera stopped (failed to analyze the last frame)';
          realtimeStatus.classList.add('is-error');
        }
      } catch (err) {
        console.error('Failed to analyze final frame:', err);
        realtimeStatus.textContent = 'Camera stopped (server connection error)';
        realtimeStatus.classList.add('is-error');
      } finally {
        showLoading(false);
      }
    }

    lastRealtimeFrame = null;
  }

  btnStartWebcam.addEventListener('click', startWebcam);
  btnStopWebcam.addEventListener('click', stopWebcamAndAnalyze);

  // Release camera stream when leaving the tab without clicking Stop.
  window.addEventListener('beforeunload', stopWebcamCleanup);
}
