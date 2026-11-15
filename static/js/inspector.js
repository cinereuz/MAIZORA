// 5-Stage Methodology Pipeline Inspector, Charts, Kernel Cards, & CSV Export

const STEP_TITLES = {
  1: 'Input & ROI Standardization',
  2: 'Preprocessing (Grayscale & Blur)',
  3: 'Segmentation (Otsu & K-Means)',
  4: 'Feature Extraction (GLCM & HSV)',
  5: 'SVM Classification & Novelty Gate',
};

const PREDICTION_LABELS = {
  sehat: 'Healthy',
  terkontaminasi: 'Contaminated',
  bukan_jagung: 'Not Corn',
};

let currentResults = null;
let activePipelineStep = 1;
let selectedKernelIndex = 0;
let isShowAllMode = false;
let pagerKernels = [];

let pagerCard = null;
let pagerInfo = null;
let pagerPrev = null;
let pagerNext = null;
let kernelPagerCard = null;
let kernelPagerInfo = null;
let kernelPagerPrev = null;
let kernelPagerNext = null;

function getKernelState(k) {
  if (k.prediction === 'bukan_jagung') {
    return { stateClass: 'notcorn', label: 'Not Corn', colorHex: 'var(--notcorn-gray)' };
  }
  if (k.is_moldy) {
    return { stateClass: 'moldy', label: 'Mold Contaminated', colorHex: 'var(--mold-red)' };
  }
  return { stateClass: 'healthy', label: 'Healthy', colorHex: 'var(--healthy-green)' };
}

function formatStatus(prediction) {
  return PREDICTION_LABELS[prediction] || String(prediction);
}

function formatConfidence(k) {
  return k.confidence !== null && k.confidence !== undefined ? `${k.confidence}%` : '-';
}

function opencvHsvToCssRgb(hOpenCv, sOpenCv, vOpenCv) {
  const h = Math.max(0, Math.min(360, Number(hOpenCv) * 2));
  const s = Math.max(0, Math.min(1, Number(sOpenCv) / 255));
  const v = Math.max(0, Math.min(1, Number(vOpenCv) / 255));
  const c = v * s;
  const x = c * (1 - Math.abs(((h / 60) % 2) - 1));
  const m = v - c;
  let r1 = 0, g1 = 0, b1 = 0;
  if (h < 60) { r1 = c; g1 = x; b1 = 0; }
  else if (h < 120) { r1 = x; g1 = c; b1 = 0; }
  else if (h < 180) { r1 = 0; g1 = c; b1 = x; }
  else if (h < 240) { r1 = 0; g1 = x; b1 = c; }
  else if (h < 300) { r1 = x; g1 = 0; b1 = c; }
  else { r1 = c; g1 = 0; b1 = x; }
  const r = Math.round((r1 + m) * 255);
  const g = Math.round((g1 + m) * 255);
  const b = Math.round((b1 + m) * 255);
  return `rgb(${r}, ${g}, ${b})`;
}

function switchPipelineStep(stepNum) {
  activePipelineStep = Math.max(1, Math.min(5, Number(stepNum)));
  const tabs = document.querySelectorAll('.stepper-tab');
  const panes = document.querySelectorAll('.pipeline-step-pane');
  const selectorBar = document.getElementById('kernel-selector-bar');
  const prevBtn = document.getElementById('btn-step-prev');
  const nextBtn = document.getElementById('btn-step-next');
  const progLabel = document.getElementById('stepper-progress-label');

  tabs.forEach((t) => {
    const s = Number(t.dataset.step);
    t.classList.toggle('active', s === activePipelineStep);
    t.classList.toggle('completed', s < activePipelineStep);
  });

  panes.forEach((p) => {
    const s = Number(p.dataset.stepPane);
    p.classList.toggle('active', s === activePipelineStep);
  });

  if (selectorBar) {
    const showKernelBar = isShowAllMode || activePipelineStep >= 3;
    selectorBar.style.display = showKernelBar ? 'flex' : 'none';
  }

  if (progLabel) {
    progLabel.textContent = `Stage ${activePipelineStep} of 5: ${STEP_TITLES[activePipelineStep]}`;
  }
  if (prevBtn) {
    prevBtn.disabled = activePipelineStep <= 1;
    prevBtn.innerHTML = activePipelineStep > 1
      ? `&#8592; Prev: ${STEP_TITLES[activePipelineStep - 1].split(' (')[0]}`
      : '&#8592; Previous Stage';
  }
  if (nextBtn) {
    nextBtn.disabled = activePipelineStep >= 5;
    nextBtn.innerHTML = activePipelineStep < 5
      ? `Next: ${STEP_TITLES[activePipelineStep + 1].split(' (')[0]} &#8594;`
      : 'Final Stage Reached &#10003;';
  }

  // Re-draw canvases when their pane becomes visible
  if (currentResults) {
    if (activePipelineStep === 2 || isShowAllMode) {
      drawPreprocessHistogram(currentResults.stage_meta?.preprocess);
    }
    if ((activePipelineStep === 4 || isShowAllMode) && currentResults.kernels?.length) {
      const k = currentResults.kernels[selectedKernelIndex] || currentResults.kernels[0];
      drawFeatureZScoreCanvas(k);
    }
  }
}

function selectKernel(index) {
  if (!currentResults || !currentResults.kernels || currentResults.kernels.length === 0) return;
  const total = currentResults.kernels.length;
  selectedKernelIndex = Math.max(0, Math.min(total - 1, index));
  const k = currentResults.kernels[selectedKernelIndex];

  document.querySelectorAll('.kernel-pill').forEach((pill, idx) => {
    const isAct = idx === selectedKernelIndex;
    pill.classList.toggle('active', isAct);
    if (isAct && typeof pill.scrollIntoView === 'function') {
      pill.scrollIntoView({ behavior: 'smooth', block: 'nearest', inline: 'center' });
    }
  });

  const kDropdown = document.getElementById('kernel-select-dropdown');
  const kPrevBtn = document.getElementById('kernel-nav-prev');
  const kNextBtn = document.getElementById('kernel-nav-next');
  if (kDropdown) kDropdown.value = String(selectedKernelIndex);
  if (kPrevBtn) kPrevBtn.disabled = selectedKernelIndex <= 0;
  if (kNextBtn) kNextBtn.disabled = selectedKernelIndex >= total - 1;

  document.querySelectorAll('#kernels-grid .kernel-card').forEach((card, idx) => {
    card.classList.toggle('selected-card', idx === selectedKernelIndex);
  });

  document.querySelectorAll('#feature-table tbody tr').forEach((row, idx) => {
    row.classList.toggle('selected-row', idx === selectedKernelIndex);
  });

  renderKernelOverviewPager();
  renderPager();

  renderStep3SelectedKernel(k);
  renderStep4SelectedKernel(k);
  renderStep5SelectedKernel(k, currentResults.summary);
}

function drawPreprocessHistogram(prepStats) {
  const canvas = document.getElementById('step2-hist-canvas');
  if (!canvas) return;
  const ctx = canvas.getContext('2d');
  const W = canvas.width;
  const H = canvas.height;
  ctx.clearRect(0, 0, W, H);

  const hGray = prepStats?.hist_gray || [];
  const hBlur = prepStats?.hist_blur || [];
  if (hGray.length === 0) return;

  const padL = 36, padR = 14, padT = 24, padB = 28;
  const plotW = W - padL - padR;
  const plotH = H - padT - padB;
  const maxVal = Math.max(...hGray, ...hBlur, 1);
  const bins = hGray.length;
  const binW = plotW / bins;

  ctx.strokeStyle = '#D9F0E8';
  ctx.lineWidth = 1;
  for (let i = 0; i <= 4; i++) {
    const y = padT + (plotH * i) / 4;
    ctx.beginPath();
    ctx.moveTo(padL, y);
    ctx.lineTo(W - padR, y);
    ctx.stroke();
  }

  for (let i = 0; i < bins; i++) {
    const bh = (hGray[i] / maxVal) * plotH;
    const x = padL + i * binW + 1;
    const y = padT + plotH - bh;
    ctx.fillStyle = 'rgba(11, 42, 107, 0.30)';
    ctx.fillRect(x, y, Math.max(2, binW - 2), bh);
  }

  if (hBlur.length === bins) {
    ctx.strokeStyle = '#0FA9A3';
    ctx.lineWidth = 2.4;
    ctx.beginPath();
    for (let i = 0; i < bins; i++) {
      const bh = (hBlur[i] / maxVal) * plotH;
      const cx = padL + (i + 0.5) * binW;
      const cy = padT + plotH - bh;
      if (i === 0) ctx.moveTo(cx, cy);
      else ctx.lineTo(cx, cy);
    }
    ctx.stroke();
  }

  ctx.fillStyle = '#2F4577';
  ctx.font = '11px JetBrains Mono, monospace';
  ctx.fillText('0 (Dark)', padL, H - 8);
  ctx.fillText('128 (Mid)', padL + plotW * 0.45, H - 8);
  ctx.fillText('255 (Bright)', W - padR - 72, H - 8);

  ctx.fillStyle = 'rgba(11, 42, 107, 0.45)';
  ctx.fillRect(padL, 6, 12, 10);
  ctx.fillStyle = '#0B2A6B';
  ctx.fillText('Grayscale Histogram', padL + 17, 15);

  ctx.strokeStyle = '#0FA9A3';
  ctx.lineWidth = 2.5;
  ctx.beginPath();
  ctx.moveTo(padL + 165, 11);
  ctx.lineTo(padL + 180, 11);
  ctx.stroke();
  ctx.fillStyle = '#0B7F8C';
  ctx.fillText('After 5x5 Gaussian Blur', padL + 185, 15);
}

function renderStep3SelectedKernel(k) {
  const panel = document.getElementById('step3-selected-kernel-panel');
  if (!panel || !k) return;
  const st = getKernelState(k);
  const bbox = k.bbox || [0, 0, 0, 0];

  let swatchesHtml = '';
  if (k.cluster_centers && k.cluster_centers.length > 0) {
    swatchesHtml = '<div class="cluster-swatches">' + k.cluster_centers.map((c, idx) => {
      const cssColor = opencvHsvToCssRgb(c[0], c[1], c[2]);
      return `<span class="cluster-swatch-chip">
        <span class="swatch-box" style="background:${cssColor};"></span>
        C${idx + 1}: H=${c[0]} S=${c[1]} V=${c[2]}
      </span>`;
    }).join('') + '</div>';
  }

  panel.innerHTML = `
    <div class="step-compare-grid three-col" style="margin-bottom:0.5rem;">
      <div class="visual-box">
        <div class="visual-box-head">
          <span>Kernel #${k.kernel_id} Crop ROI</span>
          <span class="badge ${st.stateClass}">${st.label}</span>
        </div>
        <div class="visual-img-frame" style="max-height:210px;">
          <img src="${k.crop_b64}" alt="Kernel #${k.kernel_id} Crop">
        </div>
        <p class="visual-caption">BBox: <code>[x=${bbox[0]}, y=${bbox[1]}, w=${bbox[2]}, h=${bbox[3]}]</code> &bull; Area: <strong>${k.area_piksel ?? '-'} px</strong></p>
      </div>

      <div class="visual-box">
        <div class="visual-box-head">
          <span>HSV K-Means Quantization</span>
          <span class="mono-tag">K = 3 Clusters</span>
        </div>
        <div class="visual-img-frame" style="max-height:210px;">
          <img src="${k.kmeans_b64 || k.crop_b64}" alt="K-Means Color Quantization">
        </div>
        ${swatchesHtml || '<p class="visual-caption">Pixels grouped into 3 color clusters in HSV space.</p>'}
      </div>

      <div class="visual-box">
        <div class="visual-box-head">
          <span>Isolated Mold Lesion Mask</span>
          <span class="mono-tag">${st.stateClass === 'notcorn' ? 'Excluded' : k.mold_ratio + '% Area'}</span>
        </div>
        <div class="visual-img-frame" style="max-height:210px;">
          <img src="${k.mold_mask_b64 || k.crop_b64}" alt="Mold Lesion Binary Mask">
        </div>
        <p class="visual-caption">Cluster with minimum health score (<code>1.5&times;S + 0.8&times;V</code>) isolated as fungal lesion mask.</p>
      </div>
    </div>
  `;
}

function drawFeatureZScoreCanvas(k) {
  const canvas = document.getElementById('step4-feature-canvas');
  if (!canvas || !k) return;
  const ctx = canvas.getContext('2d');
  const W = canvas.width;
  const H = canvas.height;
  ctx.clearRect(0, 0, W, H);

  const sf = k.scaled_features || {
    contrast: 0, correlation: 0, energy: 0, homogeneity: 0, hue: 0, saturation: 0, value: 0,
  };
  const keys = ['contrast', 'correlation', 'energy', 'homogeneity', 'hue', 'saturation', 'value'];
  const labels = ['Contrast', 'Correl', 'Energy', 'Homog', 'Hue', 'Sat', 'Value'];

  const padL = 38, padR = 14, padT = 24, padB = 28;
  const plotW = W - padL - padR;
  const plotH = H - padT - padB;
  const midY = padT + plotH / 2;

  ctx.strokeStyle = '#9BD8F0';
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(padL, midY);
  ctx.lineTo(W - padR, midY);
  ctx.stroke();

  ctx.fillStyle = '#0B2A6B';
  ctx.font = '600 11px Poppins, sans-serif';
  ctx.fillText(`Standardized Feature Vector X_scaled (Z-Scores for Kernel #${k.kernel_id})`, padL, 15);

  ctx.font = '10px JetBrains Mono, monospace';
  ctx.fillStyle = '#5B6B8C';
  ctx.fillText('+3\u03c3', 8, padT + 8);
  ctx.fillText(' 0', 12, midY + 4);
  ctx.fillText('-3\u03c3', 8, padT + plotH);

  const colW = plotW / keys.length;
  keys.forEach((key, idx) => {
    const z = Math.max(-3.2, Math.min(3.2, Number(sf[key] || 0)));
    const barH = (Math.abs(z) / 3.2) * (plotH / 2);
    const x = padL + idx * colW + colW * 0.18;
    const bw = colW * 0.64;
    const y = z >= 0 ? midY - barH : midY;

    ctx.fillStyle = z >= 0 ? '#0FA9A3' : '#0B2A6B';
    ctx.fillRect(x, y, bw, Math.max(2, barH));

    ctx.fillStyle = '#1F2A44';
    ctx.fillText(labels[idx], padL + idx * colW + 6, H - 8);
  });
}

function renderStep4SelectedKernel(k) {
  const badgeEl = document.getElementById('step4-kernel-badge');
  const glcmContainer = document.getElementById('step4-glcm-bars');
  const hsvContainer = document.getElementById('step4-hsv-panel');
  if (!k) return;

  const st = getKernelState(k);
  if (badgeEl) badgeEl.textContent = `Kernel #${k.kernel_id} (${st.label})`;

  const f = k.features || {};
  const sf = k.scaled_features || {};

  const glcmSpecs = [
    {
      key: 'contrast',
      name: 'Contrast (Local Intensity Variation)',
      val: f.contrast,
      z: sf.contrast,
      pct: Math.min(100, Math.max(4, (Number(f.contrast) / 60) * 100)),
      desc: 'Measures local intensity variation. Moldy/damaged kernels exhibit higher contrast due to rough fungal lesions.',
    },
    {
      key: 'correlation',
      name: 'Correlation (Linear Gray-Level Dependency)',
      val: f.correlation,
      z: sf.correlation,
      pct: Math.min(100, Math.max(4, Number(f.correlation) * 100)),
      desc: 'Measures linear dependency of neighboring pixels (0 to 1). Smooth pericarp surfaces maintain high correlation.',
    },
    {
      key: 'energy',
      name: 'Energy / Angular Second Moment (Uniformity)',
      val: f.energy,
      z: sf.energy,
      pct: Math.min(100, Math.max(4, (Number(f.energy) / 0.35) * 100)),
      desc: 'Sum of squared GLCM elements. Higher energy indicates uniform, homogeneous surface texture.',
    },
    {
      key: 'homogeneity',
      name: 'Homogeneity (Inverse Difference Moment)',
      val: f.homogeneity,
      z: sf.homogeneity,
      pct: Math.min(100, Math.max(4, Number(f.homogeneity) * 100)),
      desc: 'Measures closeness of GLCM distribution to the diagonal. Drops when fungal spores disrupt surface smoothness.',
    },
  ];

  if (glcmContainer) {
    glcmContainer.innerHTML = glcmSpecs.map((item) => `
      <div class="feat-bar-item">
        <div class="feat-bar-top">
          <span>${item.name}</span>
          <span class="feat-bar-vals">${item.val} ${item.z !== undefined ? `(Z = ${item.z > 0 ? '+' : ''}${item.z})` : ''}</span>
        </div>
        <div class="feat-bar-track">
          <div class="feat-bar-fill" style="width:${item.pct.toFixed(1)}%;"></div>
        </div>
        <div class="feat-bar-desc">${item.desc}</div>
      </div>
    `).join('');
  }

  if (hsvContainer) {
    const liveRgb = opencvHsvToCssRgb(f.hue, f.saturation, f.value);
    const huePct = Math.min(100, Math.max(2, (Number(f.hue) / 179) * 100));
    const satPct = Math.min(100, Math.max(2, (Number(f.saturation) / 255) * 100));
    const valPct = Math.min(100, Math.max(2, (Number(f.value) / 255) * 100));

    hsvContainer.innerHTML = `
      <div class="hsv-swatch-row">
        <div class="hsv-live-swatch" style="background:${liveRgb};" title="Mean Kernel Color"></div>
        <div>
          <div style="font-size:0.82rem; font-weight:600; color:var(--navy);">Mean Foreground HSV Color Swatch</div>
          <div style="font-family:var(--font-mono); font-size:0.74rem; color:var(--teal-dark);">
            H = ${f.hue} (OpenCV 0–179) &bull; S = ${f.saturation} (0–255) &bull; V = ${f.value} (0–255)
          </div>
          <div style="font-size:0.69rem; color:var(--text-secondary);">
            Valid Corn Hue Gate: <code>12.0 &le; Hue &le; 41.0</code> (Golden-yellow to olive-brown)
          </div>
        </div>
      </div>
      <div class="feat-bar-item">
        <div class="feat-bar-top">
          <span>Hue (Dominant Wavelength)</span>
          <span class="feat-bar-vals">${f.hue} ${sf.hue !== undefined ? `(Z = ${sf.hue > 0 ? '+' : ''}${sf.hue})` : ''}</span>
        </div>
        <div class="feat-bar-track">
          <div class="feat-bar-fill" style="width:${huePct.toFixed(1)}%; background:linear-gradient(90deg, #EAB308, #10A54A);"></div>
        </div>
      </div>
      <div class="feat-bar-item">
        <div class="feat-bar-top">
          <span>Saturation (Chroma Purity)</span>
          <span class="feat-bar-vals">${f.saturation} ${sf.saturation !== undefined ? `(Z = ${sf.saturation > 0 ? '+' : ''}${sf.saturation})` : ''}</span>
        </div>
        <div class="feat-bar-track">
          <div class="feat-bar-fill" style="width:${satPct.toFixed(1)}%;"></div>
        </div>
      </div>
      <div class="feat-bar-item">
        <div class="feat-bar-top">
          <span>Value (Luminance Brightness)</span>
          <span class="feat-bar-vals">${f.value} ${sf.value !== undefined ? `(Z = ${sf.value > 0 ? '+' : ''}${sf.value})` : ''}</span>
        </div>
        <div class="feat-bar-track">
          <div class="feat-bar-fill" style="width:${valPct.toFixed(1)}%;"></div>
        </div>
      </div>
    `;
  }

  drawFeatureZScoreCanvas(k);
}

function renderStep5SelectedKernel(k, summary) {
  const container = document.getElementById('step5-decision-inspector');
  const batchBox = document.getElementById('step5-batch-summary-box');
  if (!container || !k) return;

  const st = getKernelState(k);
  const f = k.features || {};
  const hueOk = f.hue >= 12.0 && f.hue <= 41.0;
  const morphOk = (k.area_piksel || 500) <= 10500 && (k.aspect_ratio || 1.2) <= 2.35 && (k.solidity || 0.9) >= 0.58;
  const od = k.outlier_distance;
  const ot = k.outlier_threshold;
  const mahaOk = k.is_corn && (od === null || ot === null || od <= ot);

  const dec = k.decision_score;
  // Map decision score [-3, +3] to percentage [5%, 95%] where 50% is hyperplane f(x)=0
  const markerPct = dec !== null && dec !== undefined
    ? Math.max(5, Math.min(95, 50 + (Number(dec) / 3.0) * 45))
    : 50;

  const probHealthy = k.probabilities?.sehat ?? 0;
  const probMoldy = k.probabilities?.terkontaminasi ?? 0;

  container.innerHTML = `
    <div class="sub-panel">
      <div class="sub-panel-head">
        <h4 class="sub-panel-title">5A. Multi-Layer Novelty Gate (<code>check_is_corn</code>)</h4>
        <span class="badge ${st.stateClass}">Kernel #${k.kernel_id}: ${st.label}</span>
      </div>
      <div class="gate-checklist">
        <div class="gate-item">
          <div>
            <strong>Layer 1: Color &amp; Warmth Constraints</strong>
            <div style="font-size:0.7rem; color:var(--text-secondary);">Hue = ${f.hue} (valid [12, 41]), Sat = ${f.saturation}, Warmth R &gt; B</div>
          </div>
          <span class="gate-status ${hueOk ? 'pass' : 'fail'}">${hueOk ? 'PASS' : 'REJECT'}</span>
        </div>
        <div class="gate-item">
          <div>
            <strong>Layer 2: Morphological Geometry</strong>
            <div style="font-size:0.7rem; color:var(--text-secondary);">Area = ${k.area_piksel ?? '-'} px, Aspect = ${k.aspect_ratio ?? '-'}:1, Solidity = ${k.solidity ?? '-'}</div>
          </div>
          <span class="gate-status ${morphOk ? 'pass' : 'fail'}">${morphOk ? 'PASS' : 'REJECT'}</span>
        </div>
        <div class="gate-item">
          <div>
            <strong>Layer 3: 7D Subcluster Mahalanobis Distance</strong>
            <div style="font-size:0.7rem; color:var(--text-secondary);">
              Distance d<sub>M</sub> = <code>${od ?? '-'}</code> vs Threshold &tau; = <code>${ot ?? '-'}</code>
              ${k.outlier_nearest_class ? `(Nearest: ${formatStatus(k.outlier_nearest_class)})` : ''}
            </div>
          </div>
          <span class="gate-status ${mahaOk ? 'pass' : 'fail'}">${mahaOk ? 'VALID CORN' : 'OUTLIER'}</span>
        </div>
      </div>
    </div>

    <div class="sub-panel">
      <div class="sub-panel-head">
        <h4 class="sub-panel-title">5B. RBF-SVM Hyperplane Decision Function</h4>
        <span class="mono-tag">${dec !== null && dec !== undefined ? `f(x) = ${dec > 0 ? '+' : ''}${dec}` : 'Bypassed (Not Corn)'}</span>
      </div>
      ${k.prediction === 'bukan_jagung' ? `
        <div class="theory-box" style="border-left-color:var(--notcorn-gray);">
          <strong>Rejected by Novelty Gate:</strong>
          <span>Kernel #${k.kernel_id} was rejected as a non-corn object before SVM classification and is excluded from the batch contamination rate.</span>
        </div>
      ` : `
        <div class="hyperplane-meter">
          <div class="hyperplane-scale">
            <span style="color:var(--healthy-green);">&#8592; Healthy Class (f(x) &lt; 0)</span>
            <span>Hyperplane (0.0)</span>
            <span style="color:var(--mold-red);">Mold Class (f(x) &gt; 0) &#8594;</span>
          </div>
          <div class="hyperplane-track">
            <div class="hyperplane-zero" title="Decision Boundary f(x) = 0"></div>
            <div class="hyperplane-marker" style="left:${markerPct.toFixed(1)}%;" title="f(x) = ${dec}"></div>
          </div>
          <div style="display:flex; justify-content:space-between; font-family:var(--font-mono); font-size:0.75rem; margin-top:0.4rem;">
            <span>P(Healthy): <strong>${probHealthy}%</strong></span>
            <span>Confidence: <strong>${k.confidence}%</strong></span>
            <span>P(Mold): <strong>${probMoldy}%</strong></span>
          </div>
        </div>
        <div class="theory-box">
          <strong>Platt Sigmoid Calibration:</strong>
          <span>RBF-SVM maps the 7D scaled vector into a high-dimensional kernel space and computes signed distance <code>f(x) = ${dec}</code> from the separating hyperplane, converted to posterior probability via <code>P = 1 / (1 + exp(-f(x)))</code>.</span>
        </div>
      `}
    </div>
  `;

  if (batchBox && summary) {
    const totalJagung = (summary.sehat || 0) + (summary.terkontaminasi || 0);
    batchBox.innerHTML = `
      <div class="visual-box-head">
        <span>Batch Quality Grading Summary</span>
        <span class="mono-tag">Contamination: ${summary.persen_kontaminasi}%</span>
      </div>
      <div class="step-params-grid compact" style="margin-top:0.25rem;">
        <div class="param-item">
          <span class="param-label">Valid Corn Kernels</span>
          <span class="param-val">${totalJagung} Kernel(s)</span>
          <span class="param-sub">Excluded Non-Corn: ${summary.bukan_jagung || 0}</span>
        </div>
        <div class="param-item">
          <span class="param-label">Class Breakdown</span>
          <span class="param-val" style="color:var(--healthy-green);">${summary.sehat} Healthy</span> /
          <span class="param-val" style="color:var(--mold-red);">${summary.terkontaminasi} Mold</span>
          <span class="param-sub">Ratio: ${summary.persen_kontaminasi}% infected</span>
        </div>
      </div>
      <div class="theory-box" style="margin-top:auto;">
        <strong>Final Quality Verdict:</strong>
        <span>${summary.status_mutu}</span>
        <div style="margin-top:0.35rem; font-size:0.72rem; color:var(--text-secondary);">
          Grading Thresholds: <strong>0%</strong> = Excellent (Seed/Consumption) &bull; <strong>&le; 15%</strong> = Standard (Sorting Required) &bull; <strong>&gt; 15%</strong> = Poor Quality (Unfit).
        </div>
      </div>
    `;
  }
}

function setupStages(images) {
  const stageImg = document.getElementById('stage-image');
  const stageDesc = document.getElementById('stage-desc');
  const stageButtons = document.querySelectorAll('.stage-btn');

  const stageData = {
    annotated: {
      img: images.annotated,
      desc: 'Final classification result of the Support Vector Machine (SVM): green bounding boxes indicate healthy kernels, red indicates mold-contaminated kernels, and gray indicates objects detected as NOT corn kernels (excluded from the quality assessment).',
    },
    original: {
      img: images.original,
      desc: 'Original high-resolution corn kernel image before preprocessing.',
    },
    'gray-blur': {
      img: images.blur,
      desc: 'Preprocessing Step 1: Conversion of the image to grayscale and noise reduction using Gaussian Blur.',
    },
    otsu: {
      img: images.otsu_mask,
      desc: 'Segmentation Step 2A: Separation of kernel objects from the background using Otsu Thresholding and an elliptical morphological filter.',
    },
  };

  stageImg.src = stageData.annotated.img;
  stageDesc.textContent = stageData.annotated.desc;

  stageButtons.forEach((btn) => {
    btn.onclick = () => {
      stageButtons.forEach((b) => b.classList.remove('active'));
      btn.classList.add('active');
      const stageKey = btn.dataset.stage;
      if (stageData[stageKey]) {
        stageImg.src = stageData[stageKey].img;
        stageDesc.textContent = stageData[stageKey].desc;
      }
    };
  });
}

function buildKernelCardHtml(k) {
  const state = getKernelState(k);
  const hasConfidence = k.confidence !== null && k.confidence !== undefined;
  const thirdThumbLabel = state.stateClass === 'notcorn'
    ? `Not Corn (distance ${k.outlier_distance ?? '-'} vs threshold ${k.outlier_threshold ?? '-'})`
    : `Mold Area (${k.mold_ratio}%)`;

  return `
    <div class="kernel-header">
      <span class="kernel-title">Kernel #${k.kernel_id}</span>
      <span class="badge ${state.stateClass}">${state.label}</span>
    </div>
    <div class="kernel-visuals">
      <div class="thumb-box">
        <img src="${k.crop_b64}" alt="Original Crop">
        <span>Kernel Crop</span>
      </div>
      <div class="thumb-box">
        <img src="${k.kmeans_b64 || k.crop_b64}" alt="K-Means Cluster">
        <span>K-Means (Color)</span>
      </div>
      <div class="thumb-box">
        <img src="${k.mold_mask_b64 || k.crop_b64}" alt="Mold Mask">
        <span>${thirdThumbLabel}</span>
      </div>
    </div>
    <div class="kernel-features-mini">
      <div>Contrast: <span class="feature-val">${k.features.contrast}</span></div>
      <div>Homogeneity: <span class="feature-val">${k.features.homogeneity}</span></div>
      <div>Correlation: <span class="feature-val">${k.features.correlation}</span></div>
      <div>Energy: <span class="feature-val">${k.features.energy}</span></div>
      <div>Hue: <span class="feature-val">${k.features.hue}</span></div>
      <div>Sat: <span class="feature-val">${k.features.saturation}</span></div>
    </div>
    ${!hasConfidence ? `<div style="font-size:0.78rem;color:${state.colorHex};margin-top:0.5rem;">Confidence not applicable -- object rejected before the SVM classification stage (see the distance details above)</div>` : ''}
  `;
}

function renderKernelCards(kernels) {
  const container = document.getElementById('kernels-grid');
  container.innerHTML = '';

  kernels.forEach((k, idx) => {
    const state = getKernelState(k);
    const card = document.createElement('div');
    card.className = `kernel-card selectable-kernel ${state.stateClass} ${idx === selectedKernelIndex ? 'selected-card' : ''}`;
    card.addEventListener('click', () => selectKernel(idx));
    card.innerHTML = buildKernelCardHtml(k);
    container.appendChild(card);
  });
}

function renderKernelOverviewPager() {
  if (!kernelPagerCard || !kernelPagerInfo || !kernelPagerPrev || !kernelPagerNext) return;
  const kernels = currentResults?.kernels || [];
  const total = kernels.length;
  if (total === 0) {
    kernelPagerCard.innerHTML = '<div class="pager-empty">No kernels detected yet.</div>';
    kernelPagerInfo.textContent = '0 of 0';
    kernelPagerPrev.disabled = true;
    kernelPagerNext.disabled = true;
    return;
  }

  const idx = Math.max(0, Math.min(total - 1, selectedKernelIndex));
  const k = kernels[idx];
  const state = getKernelState(k);
  kernelPagerCard.innerHTML = `
    <div class="kernel-card ${state.stateClass} selected-card">
      ${buildKernelCardHtml(k)}
    </div>
  `;

  kernelPagerInfo.textContent = `${idx + 1} of ${total}`;
  kernelPagerPrev.disabled = idx === 0;
  kernelPagerNext.disabled = idx === total - 1;
}

function renderPager() {
  if (!pagerCard || !pagerInfo || !pagerPrev || !pagerNext) return;
  const total = pagerKernels.length;
  if (total === 0) {
    pagerCard.innerHTML = '<div class="pager-empty">No kernels detected yet.</div>';
    pagerInfo.textContent = '0 of 0';
    pagerPrev.disabled = true;
    pagerNext.disabled = true;
    return;
  }

  const idx = Math.max(0, Math.min(total - 1, selectedKernelIndex));
  const k = pagerKernels[idx];
  const state = getKernelState(k);
  const rows = [
    ['SVM Status', `<span class="st-${state.stateClass}">${formatStatus(k.prediction)}</span>`],
    ['Conf', formatConfidence(k)],
    ['% Mold', `${k.mold_ratio}%`],
    ['Contrast', k.features.contrast],
    ['Correlation', k.features.correlation],
    ['Energy', k.features.energy],
    ['Homogeneity', k.features.homogeneity],
    ['Hue', k.features.hue],
    ['Saturation', k.features.saturation],
    ['Value', k.features.value],
  ];

  pagerCard.innerHTML =
    `<div class="pager-head">
      <img class="pager-thumb" src="${k.crop_b64}" alt="Kernel crop #${k.kernel_id}">
      <span class="pager-id">#${k.kernel_id}</span>
    </div>` + rows.map((r) => `<div class="pager-row"><span class="pager-label">${r[0]}</span><span class="pager-value">${r[1]}</span></div>`).join('');

  pagerInfo.textContent = `${idx + 1} of ${total}`;
  pagerPrev.disabled = idx === 0;
  pagerNext.disabled = idx === total - 1;
}

function renderFeatureTable(kernels) {
  const tbody = document.querySelector('#feature-table tbody');
  tbody.innerHTML = '';

  kernels.forEach((k, idx) => {
    const tr = document.createElement('tr');
    if (idx === selectedKernelIndex) tr.classList.add('selected-row');
    tr.addEventListener('click', () => selectKernel(idx));
    const state = getKernelState(k);

    tr.innerHTML = `
      <td>#${k.kernel_id}</td>
      <td class="st-${state.stateClass}">${formatStatus(k.prediction)}</td>
      <td>${formatConfidence(k)}</td>
      <td>${k.mold_ratio}%</td>
      <td>${k.features.contrast}</td>
      <td>${k.features.correlation}</td>
      <td>${k.features.energy}</td>
      <td>${k.features.homogeneity}</td>
      <td>${k.features.hue}</td>
      <td>${k.features.saturation}</td>
      <td>${k.features.value}</td>
    `;
    tbody.appendChild(tr);
  });

  pagerKernels = kernels;
  renderPager();
}

function setupPipelineInspector(data) {
  const images = data.images || {};
  const meta = data.stage_meta || {};
  const kernels = data.kernels || [];

  const rawImgEl = document.getElementById('step1-raw-img');
  const stdImgEl = document.getElementById('step1-std-img');
  const rawDimEl = document.getElementById('step1-raw-dim');
  const metaGrid1 = document.getElementById('step1-meta-grid');

  if (rawImgEl) rawImgEl.src = images.raw_input || images.original;
  if (stdImgEl) stdImgEl.src = images.original;

  const inMeta = meta.input || {};
  const segMeta = meta.segmentation || {};
  const origW = inMeta.orig_width || 224;
  const origH = inMeta.orig_height || 224;
  if (rawDimEl) rawDimEl.textContent = `${origW} \u00d7 ${origH} px`;

  if (metaGrid1) {
    metaGrid1.innerHTML = `
      <div class="param-item">
        <span class="param-label">Original Resolution</span>
        <span class="param-val">${origW} &times; ${origH} px</span>
        <span class="param-sub">Aspect Ratio ${inMeta.orig_aspect || '1:1'}</span>
      </div>
      <div class="param-item">
        <span class="param-label">Standardized ROI</span>
        <span class="param-val">${inMeta.std_width || 224} &times; ${inMeta.std_height || 224} px</span>
        <span class="param-sub">${inMeta.was_cropped ? 'Auto-centered square ROI crop' : 'Already standardized 224\u00d7224'}</span>
      </div>
      <div class="param-item">
        <span class="param-label">Background Polarity</span>
        <span class="param-val">${segMeta.bg_mode ? segMeta.bg_mode.split(' (')[0] : 'Dark Background'}</span>
        <span class="param-sub">Border Gray: ${segMeta.bg_gray ?? '-'} | Sat: ${segMeta.bg_sat ?? '-'}</span>
      </div>
      <div class="param-item">
        <span class="param-label">Detected Segments</span>
        <span class="param-val">${kernels.length} Object(s)</span>
        <span class="param-sub">${data.summary.sehat} Healthy, ${data.summary.terkontaminasi} Moldy</span>
      </div>
    `;
  }

  const s2Rgb = document.getElementById('step2-rgb-img');
  const s2Gray = document.getElementById('step2-gray-img');
  const s2Blur = document.getElementById('step2-blur-img');
  const s2Stats = document.getElementById('step2-stats-grid');

  if (s2Rgb) s2Rgb.src = images.original;
  if (s2Gray) s2Gray.src = images.gray;
  if (s2Blur) s2Blur.src = images.blur;

  const prepStats = meta.preprocess || {};
  const stdDiff = (prepStats.gray_std && prepStats.blur_std)
    ? Math.max(0, prepStats.gray_std - prepStats.blur_std).toFixed(2)
    : '0.45';
  if (s2Stats) {
    s2Stats.innerHTML = `
      <div class="param-item">
        <span class="param-label">Grayscale Mean (&mu;)</span>
        <span class="param-val">${prepStats.gray_mean ?? '-'}</span>
        <span class="param-sub">0 (Black) to 255 (White)</span>
      </div>
      <div class="param-item">
        <span class="param-label">Grayscale Std (&sigma;)</span>
        <span class="param-val">${prepStats.gray_std ?? '-'}</span>
        <span class="param-sub">Raw intensity dispersion</span>
      </div>
      <div class="param-item">
        <span class="param-label">Gaussian Blur (5&times;5)</span>
        <span class="param-val">&sigma; = ${prepStats.blur_std ?? '-'}</span>
        <span class="param-sub">Noise variance drop: -${stdDiff}</span>
      </div>
      <div class="param-item">
        <span class="param-label">Feature Patch Size</span>
        <span class="param-val">128 &times; 128 px</span>
        <span class="param-sub">Bicubic + 3&times;3 Gaussian</span>
      </div>
    `;
  }
  drawPreprocessHistogram(prepStats);

  const s3Otsu = document.getElementById('step3-otsu-img');
  const s3Overlay = document.getElementById('step3-overlay-img');
  const s3OtsuTag = document.getElementById('step3-otsu-tag');
  const s3RoiCount = document.getElementById('step3-roi-count');
  const s3Stats = document.getElementById('step3-seg-stats');

  if (s3Otsu) s3Otsu.src = images.otsu_mask;
  if (s3Overlay) s3Overlay.src = images.seg_overlay || images.annotated;
  if (s3OtsuTag) s3OtsuTag.textContent = `Otsu Threshold T = ${segMeta.otsu_threshold ?? '-'}`;
  if (s3RoiCount) s3RoiCount.textContent = `${kernels.length} Segmented ROI(s)`;

  if (s3Stats) {
    s3Stats.innerHTML = `
      <div class="param-item">
        <span class="param-label">Otsu Threshold (T*)</span>
        <span class="param-val">${segMeta.otsu_threshold ?? '-'} / 255</span>
        <span class="param-sub">Maximizes inter-class variance &sigma;&sup2;<sub>B</sub></span>
      </div>
      <div class="param-item">
        <span class="param-label">Polarity Mode</span>
        <span class="param-val">${segMeta.bg_mode || 'Standard Polarity'}</span>
        <span class="param-sub">Cool cloth glare suppression active</span>
      </div>
      <div class="param-item">
        <span class="param-label">Morphological Filter</span>
        <span class="param-val">3&times;3 Ellipse Open</span>
        <span class="param-sub">+ Binary Hole Filling</span>
      </div>
      <div class="param-item">
        <span class="param-label">Watershed Splitting</span>
        <span class="param-val">${segMeta.raw_components ?? kernels.length} Blob(s) &rarr; ${kernels.length} ROI(s)</span>
        <span class="param-sub">CLAHE + Distance Transform</span>
      </div>
    `;
  }

  const pillsContainer = document.getElementById('kernel-selector-pills');
  const kDropdown = document.getElementById('kernel-select-dropdown');
  if (pillsContainer) {
    pillsContainer.innerHTML = '';
    kernels.forEach((k, idx) => {
      const st = getKernelState(k);
      const btn = document.createElement('button');
      btn.type = 'button';
      btn.className = `kernel-pill ${idx === 0 ? 'active' : ''}`;
      btn.innerHTML = `
        <img class="kernel-pill-thumb" src="${k.crop_b64}" alt="Kernel #${k.kernel_id}">
        <span>Kernel #${k.kernel_id}</span>
        <span class="kernel-pill-dot ${st.stateClass}" title="${st.label}"></span>
      `;
      btn.addEventListener('click', () => selectKernel(idx));
      pillsContainer.appendChild(btn);
    });
  }
  if (kDropdown) {
    kDropdown.innerHTML = '';
    kernels.forEach((k, idx) => {
      const st = getKernelState(k);
      const opt = document.createElement('option');
      opt.value = String(idx);
      opt.textContent = `Kernel #${k.kernel_id} \u2014 ${st.label} (${idx + 1}/${kernels.length})`;
      kDropdown.appendChild(opt);
    });
  }

  const s5Ann = document.getElementById('step5-annotated-img');
  if (s5Ann) s5Ann.src = images.annotated;

  selectKernel(0);
  switchPipelineStep(activePipelineStep);
}

export function renderInspectorResults(data) {
  currentResults = data;
  selectedKernelIndex = 0;
  setupStages(data.images);
  renderKernelCards(data.kernels);
  renderFeatureTable(data.kernels);
  setupPipelineInspector(data);
}

export function initInspector() {
  pagerCard = document.getElementById('pager-card');
  pagerInfo = document.getElementById('pager-info');
  pagerPrev = document.getElementById('pager-prev');
  pagerNext = document.getElementById('pager-next');

  kernelPagerCard = document.getElementById('kernel-overview-pager-card');
  kernelPagerInfo = document.getElementById('kernel-pager-info');
  kernelPagerPrev = document.getElementById('kernel-pager-prev');
  kernelPagerNext = document.getElementById('kernel-pager-next');

  const kNavPrev = document.getElementById('kernel-nav-prev');
  const kNavNext = document.getElementById('kernel-nav-next');
  const kSelectDropdown = document.getElementById('kernel-select-dropdown');
  if (kNavPrev) {
    kNavPrev.addEventListener('click', () => {
      if (selectedKernelIndex > 0) selectKernel(selectedKernelIndex - 1);
    });
  }
  if (kNavNext) {
    kNavNext.addEventListener('click', () => {
      if (currentResults?.kernels && selectedKernelIndex < currentResults.kernels.length - 1) {
        selectKernel(selectedKernelIndex + 1);
      }
    });
  }
  if (kSelectDropdown) {
    kSelectDropdown.addEventListener('change', (e) => {
      selectKernel(Number(e.target.value));
    });
  }

  if (kernelPagerPrev) {
    kernelPagerPrev.addEventListener('click', () => {
      if (selectedKernelIndex > 0) selectKernel(selectedKernelIndex - 1);
    });
  }
  if (kernelPagerNext) {
    kernelPagerNext.addEventListener('click', () => {
      const total = currentResults?.kernels?.length || 0;
      if (selectedKernelIndex < total - 1) selectKernel(selectedKernelIndex + 1);
    });
  }

  if (pagerPrev) {
    pagerPrev.addEventListener('click', () => {
      if (selectedKernelIndex > 0) selectKernel(selectedKernelIndex - 1);
    });
  }
  if (pagerNext) {
    pagerNext.addEventListener('click', () => {
      if (selectedKernelIndex < pagerKernels.length - 1) selectKernel(selectedKernelIndex + 1);
    });
  }

  document.querySelectorAll('.stepper-tab').forEach((tab) => {
    tab.addEventListener('click', () => {
      switchPipelineStep(Number(tab.dataset.step));
    });
  });

  const btnStepPrev = document.getElementById('btn-step-prev');
  const btnStepNext = document.getElementById('btn-step-next');
  if (btnStepPrev) {
    btnStepPrev.addEventListener('click', () => {
      if (activePipelineStep > 1) switchPipelineStep(activePipelineStep - 1);
    });
  }
  if (btnStepNext) {
    btnStepNext.addEventListener('click', () => {
      if (activePipelineStep < 5) switchPipelineStep(activePipelineStep + 1);
    });
  }

  const btnModeStepper = document.getElementById('btn-mode-stepper');
  const btnModeAll = document.getElementById('btn-mode-all');
  const inspectorCard = document.querySelector('.pipeline-inspector-card');
  const stepperFooter = document.getElementById('stepper-footer');

  if (btnModeStepper && btnModeAll && inspectorCard) {
    btnModeStepper.addEventListener('click', () => {
      isShowAllMode = false;
      inspectorCard.classList.remove('show-all-steps');
      btnModeStepper.classList.add('active-mode');
      btnModeAll.classList.remove('active-mode');
      if (stepperFooter) stepperFooter.style.display = 'flex';
      switchPipelineStep(activePipelineStep);
    });
    btnModeAll.addEventListener('click', () => {
      isShowAllMode = true;
      inspectorCard.classList.add('show-all-steps');
      btnModeAll.classList.add('active-mode');
      btnModeStepper.classList.remove('active-mode');
      if (stepperFooter) stepperFooter.style.display = 'none';
      switchPipelineStep(activePipelineStep);
    });
  }

  const exportBtn = document.getElementById('btn-export-csv');
  if (exportBtn) {
    exportBtn.addEventListener('click', () => {
      if (!currentResults || !currentResults.kernels) return;

      let csv = 'ID,Status,Confidence(%),MoldArea(%),Contrast,Correlation,Energy,Homogeneity,Hue,Saturation,Value\n';
      currentResults.kernels.forEach((k) => {
        const confVal = k.confidence !== null && k.confidence !== undefined ? k.confidence : '';
        csv += `${k.kernel_id},${formatStatus(k.prediction)},${confVal},${k.mold_ratio},${k.features.contrast},${k.features.correlation},${k.features.energy},${k.features.homogeneity},${k.features.hue},${k.features.saturation},${k.features.value}\n`;
      });

      const blob = new Blob([csv], { type: 'text/csv' });
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `corn_disease_features_${Date.now()}.csv`;
      a.click();
      URL.revokeObjectURL(url);
    });
  }
}
