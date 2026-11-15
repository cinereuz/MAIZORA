// Corn Kernel Mold Detection Frontend Application Entry Point

import { initAdminDashboard, renderStagingNotice } from './admin.js';
import { initInspector, renderInspectorResults } from './inspector.js';
import { initWebcam } from './webcam.js';

function initApp() {
  const fileInput = document.getElementById('file-input');
  const dropZone = document.getElementById('drop-zone');
  const loadingOverlay = document.getElementById('loading-overlay');
  const resultsSection = document.getElementById('results-section');

  function showLoading(show) {
    if (show) {
      loadingOverlay.classList.add('active');
    } else {
      loadingOverlay.classList.remove('active');
    }
  }

  function displayResults(data) {
    const summary = data.summary;
    const bukanJagungCount = summary.bukan_jagung || 0;
    const totalJagung = summary.sehat + summary.terkontaminasi;

    const statusBanner = document.getElementById('status-banner');
    statusBanner.className = `status-banner ${summary.badge}`;
    document.getElementById('status-title').textContent = summary.status_mutu;

    let descText = '';
    if (totalJagung === 0) {
      descText = 'No corn kernels were detected in this image.';
      if (bukanJagungCount > 0) {
        descText += ` Objects detected as NOT corn: ${bukanJagungCount}.`;
      }
    } else if (bukanJagungCount > totalJagung) {
      descText = `Not Corn: detected ${bukanJagungCount} non-corn object(s) vs only ${totalJagung} corn-like object(s) (${summary.sehat} healthy, ${summary.terkontaminasi} mold-infected). Because non-corn objects predominate, the overall result is classified as Not Corn.`;
    } else {
      descText = `Corn kernels detected: ${totalJagung} (${summary.sehat} healthy, ${summary.terkontaminasi} mold-infected). Contamination rate: ${summary.persen_kontaminasi}%.`;
      if (bukanJagungCount > 0) {
        descText += ` Other objects not recognized as corn kernels: ${bukanJagungCount} (excluded from the quality assessment).`;
      }
    }
    document.getElementById('status-desc').textContent = descText;

    document.getElementById('metric-total').textContent = summary.total_biji;
    document.getElementById('metric-healthy').textContent = summary.sehat;
    document.getElementById('metric-moldy').textContent = summary.terkontaminasi;
    document.getElementById('metric-rate').textContent = `${summary.persen_kontaminasi}%`;

    const notcornCard = document.getElementById('metric-card-notcorn');
    if (bukanJagungCount > 0) {
      notcornCard.style.display = '';
      document.getElementById('metric-notcorn').textContent = bukanJagungCount;
    } else {
      notcornCard.style.display = 'none';
    }

    renderInspectorResults(data);
    renderStagingNotice(data.staging);

    resultsSection.style.display = 'flex';
    resultsSection.scrollIntoView({ behavior: 'smooth' });
  }

  async function detectFromUpload(file) {
    showLoading(true);
    const formData = new FormData();
    formData.append('file', file);

    try {
      const res = await fetch('/api/detect/upload', {
        method: 'POST',
        body: formData,
      });
      if (!res.ok) {
        const errData = await res.json();
        throw new Error(errData.detail || 'An error occurred while processing the image.');
      }
      const data = await res.json();
      displayResults(data);
    } catch (err) {
      alert('Error: ' + err.message);
    } finally {
      showLoading(false);
    }
  }

  dropZone.addEventListener('click', () => fileInput.click());

  dropZone.addEventListener('dragover', (e) => {
    e.preventDefault();
    dropZone.classList.add('dragover');
  });

  dropZone.addEventListener('dragleave', () => {
    dropZone.classList.remove('dragover');
  });

  dropZone.addEventListener('drop', (e) => {
    e.preventDefault();
    dropZone.classList.remove('dragover');
    if (e.dataTransfer.files.length > 0) {
      detectFromUpload(e.dataTransfer.files[0]);
    }
  });

  fileInput.addEventListener('change', (e) => {
    if (e.target.files.length > 0) {
      detectFromUpload(e.target.files[0]);
    }
  });

  initInspector();
  initWebcam({ showLoading, onResult: displayResults });
  initAdminDashboard();
}

if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', initApp);
} else {
  initApp();
}
