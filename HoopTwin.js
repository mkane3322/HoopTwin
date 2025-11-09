(() => {
    const fileInput = document.getElementById('fileInput');
  const loadedFilesEl = document.getElementById('loadedFiles');
  const playerSelect = document.getElementById('playerSelect');
  const autoFeaturesBtn = document.getElementById('autoFeaturesBtn');
  const runBoth = document.getElementById('runBoth');
  const runBrute = document.getElementById('runBrute');
  const runKD = document.getElementById('runKD');
  const kVal = document.getElementById('kVal');
  const status = document.getElementById('status');
  const resultsSection = document.getElementById('results');
  const timeBruteEl = document.getElementById('timeBrute');
  const timeKDEl = document.getElementById('timeKD');
  const theadBrute = document.getElementById('theadBrute');
  const tbodyBrute = document.getElementById('tbodyBrute');
  const theadKD = document.getElementById('theadKD');
  const tbodyKD = document.getElementById('tbodyKD');
  const downloadNorm = document.getElementById('downloadNorm');
  const featurePreview = document.getElementById('featurePreview');
  const compareCanvas = document.getElementById('compareChart');
  const ctx = compareCanvas.getContext('2d');
  let rawRows = [];       
  let headers = [];       
  let numericCols = [];   
  let normData = [];      
  let kdTree = null;      
  let idFieldCandidates = ['Player', 'Player-additional', 'player', 'player_id', 'playerId'];
  function safeSplitCSV(line, expectedCols) {
       const parts = line.split(',');
    if (expectedCols && parts.length > expectedCols) {
      const needed = expectedCols - 1;
      const out = parts.slice(0, needed).concat([parts.slice(needed).join(',')]);
      return out;
    }
    return parts;
  }
  function parseCSV(text) {
    const rawLines = text.split(/\r?\n/).filter(l => l.trim() !== '');
    if (!rawLines.length) return {headers: [], rows: [] };
    const delimiter = (rawLines[0].includes('\t')) ? '\t' : ',';
    const headers = rawLines[0].split(delimiter).map(h => h.trim());
    const rows = [];
    for (let i = 1; i < rawLines.length; i++) {
        const cols = rawLines[i].split(delimiter);
        const obj = {};
        for (let j = 0; j < headers.length; j++) {
            obj[headers[j]] = (cols[j] || '').trim();
        }
        rows.push(obj);
    }
    return { headers, rows};
  }
  function mergeRowsById(allRows) {
    const map = new Map();
    for (const r of allRows) {
      let key = '';
      for (const cand of idFieldCandidates) {
        if (r[cand] && r[cand].trim()) { key = r[cand].trim(); break; }
      }
        if (!key) {
        const idLike = Object.values(r).find(v => /^[a-z]{5,}[0-9]{2,}/i.test(v));
        if (idLike) key = idLike;
      }
      if (!key) key = (r['Player'] || r['player'] || '').trim();
      if (!key) {
        continue;
      }

      if (!map.has(key)) {
        map.set(key, Object.assign({}, r));
      } else {
        const existing = map.get(key);
        for (const h of Object.keys(r)) {
          const v = r[h];
          if (v && v.trim()) existing[h] = v;
        }
      }
    }
    return Array.from(map.values());
  }
  function unionHeaders(listOfHeaders) {
    const set = new Set();
    for (const h of listOfHeaders) set.add(h);
    return Array.from(set);
  }
  function detectNumericCols(rows, hdrs) {
    const numeric = [];
    for (const h of hdrs) {
      let numCount = 0, total = 0;
      for (let i = 0; i < Math.min(rows.length, 500); i++) {
        const v = (rows[i][h] ?? '').toString();
        if (!v) continue;
        total++;
        const cleaned = v.replace(/[%\s]/g,'').replace(/[^0-9eE\.\-+]/g,'');
        if (cleaned === '') continue;
        const n = Number(cleaned);
        if (!Number.isNaN(n)) numCount++;
      }
      if (total > 0 && (numCount / total) > 0.8) numeric.push(h);
    }
    return numeric.filter(h => !/player|team|pos|award|rank|rk/i.test(h));
  }
  function buildNormalized(rows, features) {
    const vectors = rows.map(r => features.map(f => {
      const raw = (r[f] ?? '').toString();
      const cleaned = raw.replace(/[^0-9eE\.\-+]/g,'');
      const n = Number(cleaned);
      return Number.isNaN(n) ? NaN : n;
    }));
    const mins = Array(features.length).fill(Infinity);
    const maxs = Array(features.length).fill(-Infinity);
    const sums = Array(features.length).fill(0);
    const counts = Array(features.length).fill(0);
    for (const v of vectors) {
      for (let i = 0; i < v.length; i++) {
        const val = v[i];
        if (!Number.isNaN(val)) {
          sums[i] += val;
          counts[i] += 1;
          if (val < mins[i]) mins[i] = val;
          if (val > maxs[i]) maxs[i] = val;
        }
      }
    }
    const means = mins.map((m, i) => counts[i] > 0 ? sums[i] / counts[i] : 0);
    for (let i = 0; i < mins.length; i++) {
      if (!isFinite(mins[i])) mins[i] = means[i];
      if (!isFinite(maxs[i])) maxs[i] = means[i];
    }
    const norm = vectors.map(vec => vec.map((v, i) => {
      let val = Number.isNaN(v) ? means[i] : v;
      const range = maxs[i] - mins[i];
      return range === 0 ? 0.5 : (val - mins[i]) / range;
    }));
    return { norm, mins, maxs, features };
  }
  function euclidean(a, b) {
    let s = 0;
    for (let i = 0; i < a.length; i++) {
      const d = a[i] - b[i];
      s += d * d;
    }
    return Math.sqrt(s);
  }
  function bruteForceKNN(queryIdx, data, k) {
    const q = data[queryIdx];
    const arr = [];
    for (let i = 0; i < data.length; i++) {
      if (i === queryIdx) continue;
      arr.push({ i, dist: euclidean(q, data[i]) });
    }
    arr.sort((a, b) => a.dist - b.dist);
    return arr.slice(0, k);
  }
  function buildKDTree(points, ids, depth = 0) {
    if (!points.length) return null;
    const k = points[0].length;
    const axis = depth % k;
    const items = points.map((p, i) => ({ p, id: ids[i], i }));
    items.sort((a, b) => a.p[axis] - b.p[axis]);
    const mid = Math.floor(items.length / 2);
    const node = {
      point: items[mid].p,
      id: items[mid].id,
      idx: items[mid].i,
      axis,
      left: buildKDTree(items.slice(0, mid).map(x => x.p), items.slice(0, mid).map(x => x.id), depth + 1),
      right: buildKDTree(items.slice(mid + 1).map(x => x.p), items.slice(mid + 1).map(x => x.id), depth + 1)
    };
    return node;
  }
  function kdTreeKNN(root, queryPoint, k) {
    const best = []; // max-heap simulated with array sorted desc
    function pushCandidate(item) {
      best.push(item);
      best.sort((a, b) => b.dist - a.dist);
      if (best.length > k) best.pop();
    }
    function search(node) {
      if (!node) return;
      const d = euclidean(queryPoint, node.point);
      if (best.length < k || d < best[0].dist) pushCandidate({ id: node.id, dist: d, idx: node.idx });
      const axis = node.axis;
      const diff = queryPoint[axis] - node.point[axis];
      const first = diff <= 0 ? node.left : node.right;
      const second = diff <= 0 ? node.right : node.left;
      search(first);
      if (best.length < k || Math.abs(diff) < best[0].dist) search(second);
    }
    search(root);
    return best.sort((a, b) => a.dist - b.dist);
  }
  fileInput.addEventListener('change', async (ev) => {
    const files = Array.from(ev.target.files || []);
    if (!files.length) {
      status.textContent = 'No files selected.';
      return;
    }
    loadedFilesEl.textContent = `Loading ${files.length} file(s): ${files.map(f => f.name).join(', ')}`;
    status.textContent = `Parsing ${files.length} file(s)...`;
    let allRows = [];
    const headersSeen = [];
    for (const f of files) {
      try {
        const txt = await f.text();
        const { headers: parsedHeaders, rows } = parseCSV(txt);
        headersSeen.push(...parsedHeaders);
        allRows = allRows.concat(rows);
      } catch (err) {
        console.error('Failed to read', f.name, err);
      }
    }
    headers = unionHeaders(headersSeen);
    rawRows = mergeRowsById(allRows);
    status.textContent = `Loaded ${files.length} file(s). Found ${rawRows.length} unique players after merging.`;
    loadedFilesEl.textContent = `Files: ${files.map(f => f.name).join(', ')}`;
    const detected = detectNumericCols(rawRows, headers);
    numericCols = detected.slice(0, 10);
    featurePreview.textContent = numericCols.length ? `Auto features: ${numericCols.join(', ')}` : 'No numeric features detected';
    if (numericCols.length) {
      const built = buildNormalized(rawRows, numericCols);
      normData = built.norm;
      const ids = rawRows.map(r => r['Player-additional'] || r['Player'] || r['player'] || '');
      kdTree = buildKDTree(normData, ids);
    } else {
      normData = [];
      kdTree = null;
    }
    populatePlayerSelect();
    enableControls();
  });
  function populatePlayerSelect() {
    playerSelect.innerHTML = '';
    const placeholder = document.createElement('option');
    placeholder.value = '';
    placeholder.textContent = 'Choose player...';
    playerSelect.appendChild(placeholder);
    for (const r of rawRows) {
      const display = r['Player'] || r['player'] || r['Player-additional'] || '(unknown)';
      const opt = document.createElement('option');
      opt.value = display;
      opt.textContent = display;
      playerSelect.appendChild(opt);
    }
    playerSelect.disabled = false;
  }
  function enableControls() {
    runBoth.disabled = false;
    runBrute.disabled = false;
    runKD.disabled = false;
    downloadNorm.disabled = !(normData && normData.length);
    resultsSection.classList.remove('hidden');
  }
  autoFeaturesBtn.addEventListener('click', () => {
    if (!rawRows.length || !headers.length) {
      status.textContent = 'Load datasets first.';
      return;
    }
    const detected = detectNumericCols(rawRows, headers);
    numericCols = detected.slice(0, 12);
    featurePreview.textContent = numericCols.length ? `Features: ${numericCols.join(', ')}` : 'No numeric features detected';
    if (numericCols.length) {
      const built = buildNormalized(rawRows, numericCols);
      normData = built.norm;
      const ids = rawRows.map(r => r['Player-additional'] || r['Player'] || r['player'] || '');
      kdTree = buildKDTree(normData, ids);
      status.textContent = `Features set (${numericCols.length}). Ready.`;
      downloadNorm.disabled = false;
    } else {
      status.textContent = 'No numeric columns detected.';
    }
  });
  runBoth.addEventListener('click', () => runAlgorithm('both'));
  runBrute.addEventListener('click', () => runAlgorithm('brute'));
  runKD.addEventListener('click', () => runAlgorithm('kd'));
  function runAlgorithm(mode = 'both') {
    if (!rawRows.length) { status.textContent = 'Load datasets first.'; return; }
    if (!numericCols.length || !normData.length) { status.textContent = 'Select numeric features first.'; return; }
    const playerName = playerSelect.value;
    if (!playerName) { status.textContent = 'Select a player.'; return; }
    const queryIdx = rawRows.findIndex(r => {
      const p = (r['Player'] || r['player'] || r['Player-additional'] || '').toString();
      return p === playerName;
    });
    if (queryIdx === -1) { status.textContent = 'Selected player not found in merged data.'; return; }
    const k = Math.max(1, Math.min(50, Number(kVal.value) || 5));
    if (mode === 'both' || mode === 'brute') {
      const t0 = performance.now();
      const bf = bruteForceKNN(queryIdx, normData, k);
      const t1 = performance.now();
      timeBruteEl.textContent = (t1 - t0).toFixed(3);
      const items = bf.map(x => ({ i: x.i, dist: x.dist }));
      renderTable(theadBrute, tbodyBrute, items);
      status.textContent = `Brute-Force: top ${items.length} neighbors computed.`;
    }
    if (mode === 'both' || mode === 'kd') {
      if (!kdTree) {
        const ids = rawRows.map(r => r['Player-additional'] || r['Player'] || r['player'] || '');
        kdTree = buildKDTree(normData, ids);
      }
      const t0 = performance.now();
      const kdres = kdTreeKNN(kdTree, normData[queryIdx], k);
      const t1 = performance.now();
      timeKDEl.textContent = (t1 - t0).toFixed(3);
      const items = kdres.map(x => {
        const idx = (typeof x.idx === 'number') ? x.idx : rawRows.findIndex(r => (r['Player-additional'] || r['Player'] || r['player'] || '') === x.id);
        return { i: idx, dist: x.dist };
      }).filter(it => it.i !== -1);
      renderTable(theadKD, tbodyKD, items);
      status.textContent = `KD-Tree: top ${items.length} neighbors computed.`;
      if (items.length > 0) drawComparison(queryIdx, items[0].i);
    }
  }
  function renderTable(thead, tbody, items) {
    thead.innerHTML = '';
    tbody.innerHTML = '';
    const columns = ['Player', 'Distance', ...numericCols];
    const tr = document.createElement('tr');
    for (const c of columns) {
      const th = document.createElement('th');
      th.textContent = c;
      tr.appendChild(th);
    }
    thead.appendChild(tr);
    for (const it of items) {
      const r = rawRows[it.i];
      if (!r) continue;
      const tr2 = document.createElement('tr');
      const nameTd = document.createElement('td'); nameTd.textContent = r['Player'] || r['player'] || r['Player-additional'] || ''; tr2.appendChild(nameTd);
      const dTd = document.createElement('td'); dTd.textContent = (it.dist !== undefined) ? it.dist.toFixed(4) : ''; tr2.appendChild(dTd);
      for (const f of numericCols) {
        const td = document.createElement('td'); td.textContent = r[f] || ''; tr2.appendChild(td);
      }
      tbody.appendChild(tr2);
    }
  }
  function drawComparison(idxA, idxB) {
    if (!normData[idxA] || !normData[idxB]) return;
    const a = normData[idxA], b = normData[idxB];
    const w = compareCanvas.width;
    const h = compareCanvas.height;
    ctx.clearRect(0, 0, w, h);
    ctx.font = '12px Inter, Arial';
    ctx.textBaseline = 'middle';
    const pad = 10;
    const barH = Math.min(22, (h - pad * 2) / numericCols.length - 8);
    for (let i = 0; i < numericCols.length; i++) {
      const y = pad + i * (barH + 10);
      ctx.fillStyle = '#9aa4b2';
      ctx.fillText(numericCols[i], 8, y + barH / 2);
      const ax = 180;
      const aw = Math.max(2, Math.round((w - ax - 40) * a[i]));
      ctx.fillStyle = '#3b82f6';
      ctx.fillRect(ax, y, aw, barH);
      ctx.fillStyle = '#e6eef6';
      ctx.fillText(a[i].toFixed(2), ax + aw + 6, y + barH / 2);
      const bx = ax + 260;
      const bw = Math.max(2, Math.round((w - bx - 40) * b[i]));
      ctx.fillStyle = '#10b981';
      ctx.fillRect(bx, y, bw, barH);
      ctx.fillStyle = '#e6eef6';
      ctx.fillText(b[i].toFixed(2), bx + bw + 6, y + barH / 2);
    }
  }
  downloadNorm.addEventListener('click', () => {
    if (!normData.length) return;
    const hdr = ['Player', ...numericCols];
    const lines = [hdr.join(',')];
    for (let i = 0; i < rawRows.length; i++) {
      const row = rawRows[i];
      const name = row['Player'] || row['player'] || row['Player-additional'] || '';
      const vals = normData[i].map(v => v.toFixed(6));
      lines.push([name, ...vals].join(','));
    }
    const blob = new Blob([lines.join('\n')], { type: 'text/csv' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url; a.download = 'hooptwin_normalized_dataset.csv';
    a.click();
    URL.revokeObjectURL(url);
  });
  status.textContent = 'Waiting for files (Pt.1 / Pt.2 / Pt.3)...';
  loadedFilesEl.textContent = 'No files loaded.';
})();