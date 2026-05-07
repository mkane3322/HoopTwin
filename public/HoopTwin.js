(() => {
  'use strict';
  const BASE = '/api/nba';
  const STATS = [
    { key: 'pts', lbl: 'PTS', src: 't', field: 'points'       },
    { key: 'ast', lbl: 'AST', src: 't', field: 'assists'      },
    { key: 'trb', lbl: 'REB', src: 't', field: 'totalRb'      },
    { key: 'stl', lbl: 'STL', src: 't', field: 'steals'       },
    { key: 'blk', lbl: 'BLK', src: 't', field: 'blocks'       },
    { key: 'fg',  lbl: 'FG%', src: 't', field: 'fieldPercent' },
    { key: 'x3p', lbl: '3P%', src: 't', field: 'threePercent' },
    { key: 'ts',  lbl: 'TS%', src: 'a', field: 'tsPercent'    },
    { key: 'per', lbl: 'PER', src: 'a', field: 'per'          },
    { key: 'ws',  lbl: 'WS',  src: 'a', field: 'winShares'    },
  ];
  const SK = STATS.map(s => s.key);
  const SL = STATS.map(s => s.lbl);
  const $ = id => document.getElementById(id);
  const playerInput = $('playerInput');
  const acList      = $('acList');
  const yearInput   = $('yearInput');
  const kValEl      = $('kVal');
  const findBtn     = $('findBtn');
  const tabReg      = $('tabReg');
  const tabPost     = $('tabPost');
  const sdot        = $('sdot');
  const stext       = $('stext');
  const prog        = $('prog');
  const results     = $('results');
  const simArc      = $('simArc');
  const simPct      = $('simPct');
  const simName     = $('simName');
  const simSub      = $('simSub');
  const matchChips  = $('matchChips');
  const queryChips  = $('queryChips');
  const qName       = $('qName');
  const qSeason     = $('qSeason');
  const tBrute      = $('tBrute');
  const tKD         = $('tKD');
  const dSize       = $('dSize');
  const thB         = $('thB');
  const tbB         = $('tbB');
  const thK         = $('thK');
  const tbK         = $('tbK');
  const barCanvas   = $('barCanvas');
  const barCtx      = barCanvas.getContext('2d');
  const legA        = $('legA');
  const legB        = $('legB');
  let isPlayoff = false;
  let rawRows   = [];  
  let normData  = []; 
  let kdTree    = null;
  /**
   * Update the status bar dot + message.
   * @param {string} msg
   * @param {'idle'|'loading'|'active'|'error'} mode
   */
  function setStatus(msg, mode = 'idle') {
    stext.textContent = msg;
    sdot.className = 'sdot' + (
      mode === 'active'  ? ' active'  :
      mode === 'loading' ? ' loading' :
      mode === 'error'   ? ' error'   : ''
    );
  }
  const setProgress = pct => { prog.style.width = pct + '%'; };
  tabReg.addEventListener('click', () => {
    isPlayoff = false;
    tabReg.classList.add('active');
    tabPost.classList.remove('active');
  });
  tabPost.addEventListener('click', () => {
    isPlayoff = true;
    tabPost.classList.add('active');
    tabReg.classList.remove('active');
  });
  let acTimer, acHighlight = -1;
  playerInput.addEventListener('input', () => {
    clearTimeout(acTimer);
    const q = playerInput.value.trim();
    if (q.length < 2) { closeAC(); return; }
    acTimer = setTimeout(() => fetchAutocomplete(q), 320);
  });
  playerInput.addEventListener('keydown', e => {
    const items = acList.querySelectorAll('.ac-item');
    if (e.key === 'ArrowDown') {
      e.preventDefault();
      acHighlight = Math.min(acHighlight + 1, items.length - 1);
      highlightAC(items);
    } else if (e.key === 'ArrowUp') {
      e.preventDefault();
      acHighlight = Math.max(acHighlight - 1, 0);
      highlightAC(items);
    } else if (e.key === 'Enter') {
      if (acHighlight >= 0 && items[acHighlight]) items[acHighlight].click();
      else findBtn.click();
    } else if (e.key === 'Escape') {
      closeAC();
    }
  });
  document.addEventListener('click', e => {
    if (!e.target.closest('.ac-wrap')) closeAC();
  });
  function highlightAC(items) {
    items.forEach((el, i) => el.classList.toggle('hi', i === acHighlight));
    if (items[acHighlight]) items[acHighlight].scrollIntoView({ block: 'nearest' });
  }
  function closeAC() {
    acList.classList.remove('open');
    acHighlight = -1;
  }
  async function fetchAutocomplete(query) {
    try {
      const season = parseInt(yearInput.value) || 2016;
      const url = `${BASE}?endpoint=playertotals&season=${season}&isPlayoff=${isPlayoff}&pageSize=100&page=1`;
      const data = await fetch(url).then(r => r.json());
      const matches = (data.data || [])
        .filter(p => p.playerName && p.playerName.toLowerCase().includes(query.toLowerCase()))
        .slice(0, 8);

      acList.innerHTML = '';
      if (!matches.length) { closeAC(); return; }
      matches.forEach(p => {
        const div = document.createElement('div');
        div.className = 'ac-item';
        div.innerHTML = `${p.playerName}<span class="ac-year">${p.season} · ${p.team || ''} · ${p.position || ''}</span>`;
        div.addEventListener('click', () => {
          playerInput.value = p.playerName;
          closeAC();
          findBtn.click();
        });
        acList.appendChild(div);
      });
      acHighlight = -1;
      acList.classList.add('open');
    } catch (err) {
      closeAC();
    }
  }
  /**
   * Build the proxy URL for a given endpoint + page.
   * @param {'playertotals'|'playeradvancedstats'} endpoint
   * @param {number} season
   * @param {boolean} playoff
   * @param {number} page
   */
  function proxyUrl(endpoint, season, playoff, page = 1) {
    return `${BASE}?endpoint=${endpoint}&season=${season}&isPlayoff=${playoff}&pageSize=100&page=${page}`;
  }
  /**
   * Fetch ALL pages of an endpoint through the proxy.
   * Fires page 1 first to get the total page count,
   * then fetches remaining pages in parallel.
   * @returns {Promise<Object[]>}
   */
  async function fetchAllPages(endpoint, season, playoff) {
    const first = await fetch(proxyUrl(endpoint, season, playoff, 1)).then(r => r.json());
    if (first.error) throw new Error(first.error);
    const data = [...(first.data || [])];
    const totalPages = first.pagination?.pages || 1;
    if (totalPages > 1) {
      const rest = await Promise.all(
        Array.from({ length: totalPages - 1 }, (_, i) =>
          fetch(proxyUrl(endpoint, season, playoff, i + 2)).then(r => r.json())
        )
      );
      rest.forEach(r => data.push(...(r.data || [])));
    }
    return data;
  }
  findBtn.addEventListener('click', async () => {
    const name   = playerInput.value.trim();
    const season = parseInt(yearInput.value) || 2016;
    const k      = Math.max(1, Math.min(20, parseInt(kValEl.value) || 5));
    if (!name) { setStatus('Enter a player name first.', 'error'); return; }
    findBtn.disabled = true;
    closeAC();
    setStatus(`Fetching ${season} ${isPlayoff ? 'playoff' : 'regular season'} data via proxy…`, 'loading');
    setProgress(8);
    try {
      const [totals, adv] = await Promise.all([
        fetchAllPages('playertotals', season, isPlayoff),
        fetchAllPages('playeradvancedstats', season, isPlayoff),
      ]);
      setProgress(60);
      const advMap = new Map();
      adv.forEach(a => { if (a.playerId) advMap.set(a.playerId, a); });
      rawRows = [];
      totals.forEach(t => {
        if (!t.playerName) return;
        const a = advMap.get(t.playerId) || {};
        const row = {
          playerName: t.playerName,
          team:       t.team,
          season:     t.season,
          position:   t.position,
        };
        STATS.forEach(s => {
          const raw = s.src === 't' ? t[s.field] : a[s.field];
          row[s.key] = parseFloat(raw) || 0;
        });
        rawRows.push(row);
      });
      setProgress(72);
      let qi = rawRows.findIndex(r => r.playerName.toLowerCase() === name.toLowerCase());
      if (qi === -1) qi = rawRows.findIndex(r => r.playerName.toLowerCase().includes(name.toLowerCase()));
      if (qi === -1) {
        setStatus(
          `"${name}" not found in ${season} ${isPlayoff ? 'postseason' : 'regular season'}. ` +
          `Check spelling or try a different year.`,
          'error'
        );
        findBtn.disabled = false;
        setProgress(0);
        return;
      }
      normData = buildNormalized(rawRows, SK);
      dSize.textContent = rawRows.length;
      setProgress(82);
      const t0 = performance.now();
      const bruteResults = bruteForceKNN(qi, normData, k);
      tBrute.textContent = (performance.now() - t0).toFixed(3);
      const ids = rawRows.map(r => r.playerName);
      kdTree = buildKDTree(normData, ids);
      const t1 = performance.now();
      const kdResults = kdTreeKNN(kdTree, normData[qi], k);
      tKD.textContent = (performance.now() - t1).toFixed(3);
      setProgress(94);
      const topKd  = kdResults[0];
      const topIdx = rawRows.findIndex((r, i) => i !== qi && r.playerName === topKd.id);
      const ti     = topIdx >= 0 ? topIdx : bruteResults[0].i;
      renderHero(qi, ti, topKd.dist, season);
      renderTable(thB, tbB, bruteResults.map(x => ({ i: x.i, dist: x.dist })));
      renderTable(thK, tbK,
        kdResults
          .map(x => ({ i: rawRows.findIndex(r => r.playerName === x.id), dist: x.dist }))
          .filter(x => x.i >= 0)
      );
      drawBarChart(qi, ti);
      results.classList.remove('hidden');
      setStatus(
        `Top ${k} HoopTwins for ${rawRows[qi].playerName} · ` +
        `${season} ${isPlayoff ? 'Playoffs' : 'Regular Season'}`,
        'active'
      );
      setProgress(100);
      setTimeout(() => setProgress(0), 700);

    } catch (err) {
      setStatus('Proxy error: ' + err.message, 'error');
      console.error('[HoopTwin]', err);
      setProgress(0);
    }
    findBtn.disabled = false;
  });
  function buildNormalized(rows, keys) {
    const vecs = rows.map(r =>
      keys.map(k => { const v = parseFloat(r[k]); return isNaN(v) ? NaN : v; })
    );
    const mins  = keys.map(() => Infinity);
    const maxs  = keys.map(() => -Infinity);
    const sums  = keys.map(() => 0);
    const cnts  = keys.map(() => 0);
    vecs.forEach(v => v.forEach((x, i) => {
      if (!isNaN(x)) {
        sums[i] += x; cnts[i]++;
        if (x < mins[i]) mins[i] = x;
        if (x > maxs[i]) maxs[i] = x;
      }
    }));
    const means = mins.map((_, i) => cnts[i] > 0 ? sums[i] / cnts[i] : 0);
    mins.forEach((m, i) => {
      if (!isFinite(m)) mins[i] = means[i];
      if (!isFinite(maxs[i])) maxs[i] = means[i];
    });
    return vecs.map(v => v.map((x, i) => {
      const val   = isNaN(x) ? means[i] : x;
      const range = maxs[i] - mins[i];
      return range === 0 ? 0.5 : (val - mins[i]) / range;
    }));
  }
  const euclidean = (a, b) => {
    let sum = 0;
    for (let i = 0; i < a.length; i++) { const d = a[i] - b[i]; sum += d * d; }
    return Math.sqrt(sum);
  };
  function bruteForceKNN(queryIdx, data, k) {
    const q = data[queryIdx];
    const arr = [];
    data.forEach((v, i) => { if (i !== queryIdx) arr.push({ i, dist: euclidean(q, v) }); });
    arr.sort((a, b) => a.dist - b.dist);
    return arr.slice(0, k);
  }
  function buildKDTree(points, ids, depth = 0) {
    if (!points.length) return null;
    const axis  = depth % points[0].length;
    const items = points.map((p, i) => ({ p, id: ids[i], i }));
    items.sort((a, b) => a.p[axis] - b.p[axis]);
    const mid = Math.floor(items.length / 2);
    return {
      point: items[mid].p,
      id:    items[mid].id,
      idx:   items[mid].i,
      ax:    axis,
      left:  buildKDTree(items.slice(0, mid).map(x => x.p),       items.slice(0, mid).map(x => x.id),       depth + 1),
      right: buildKDTree(items.slice(mid + 1).map(x => x.p),      items.slice(mid + 1).map(x => x.id),      depth + 1),
    };
  }
  function kdTreeKNN(root, queryPoint, k) {
    const best = [];
    const push = item => {
      best.push(item);
      best.sort((a, b) => b.dist - a.dist);   
      if (best.length > k) best.pop();
    };
    const search = node => {
      if (!node) return;
      const d = euclidean(queryPoint, node.point);
      if (best.length < k || d < best[0].dist) push({ id: node.id, dist: d });
      const diff = queryPoint[node.ax] - node.point[node.ax];
      search(diff <= 0 ? node.left  : node.right);
      if (best.length < k || Math.abs(diff) < best[0].dist)
        search(diff <= 0 ? node.right : node.left);
    };
    search(root);
    return best.sort((a, b) => a.dist - b.dist);
  }
  function renderHero(qi, ti, dist, season) {
    const q   = rawRows[qi];
    const top = rawRows[ti];
    const maxDist = Math.sqrt(SK.length);   
    const sim     = Math.max(0, Math.min(100, Math.round((1 - dist / maxDist) * 100)));
    const circ    = 2 * Math.PI * 40;     
    simArc.style.strokeDashoffset = circ - (circ * sim / 100);
    simArc.style.stroke = sim >= 75 ? '#10b981' : sim >= 50 ? '#f97316' : '#ef4444';
    simPct.textContent  = sim + '%';
    simPct.style.color  = sim >= 75 ? 'var(--green)' : sim >= 50 ? 'var(--orange)' : 'var(--red)';
    simName.textContent = top?.playerName || '—';
    simSub.textContent  = top ? `${top.team || ''} · ${season} ${isPlayoff ? 'Playoffs' : 'Reg Season'}` : '';
    qName.textContent   = q.playerName;
    qSeason.textContent = `${q.team || ''} · ${season} ${isPlayoff ? 'Playoffs' : 'Reg Season'}`;
    legA.textContent    = q.playerName;
    legB.textContent    = top?.playerName || 'Match';
    fillStatChips(matchChips, top);
    fillStatChips(queryChips, q);
  }
  function fillStatChips(el, row) {
    el.innerHTML = '';
    if (!row) return;
    [
      { k: 'pts', l: 'PTS' },
      { k: 'ast', l: 'AST' },
      { k: 'trb', l: 'REB' },
      { k: 'per', l: 'PER' },
      { k: 'ws',  l: 'WS'  },
    ].forEach(s => {
      const v = row[s.k];
      if (v == null) return;
      const chip = document.createElement('div');
      chip.className = 'sc';
      chip.innerHTML = `<div class="sc-v">${typeof v === 'number' ? v.toFixed(1) : v}</div><div class="sc-l">${s.l}</div>`;
      el.appendChild(chip);
    });
  }
  function renderTable(thead, tbody, items) {
    thead.innerHTML = '';
    tbody.innerHTML = '';

    const cols = ['#', 'Player', 'Team', 'Dist', ...SL];
    const headerRow = document.createElement('tr');
    cols.forEach(col => {
      const th = document.createElement('th');
      th.textContent = col;
      headerRow.appendChild(th);
    });
    thead.appendChild(headerRow);

    items.forEach(({ i, dist }, rank) => {
      const r = rawRows[i];
      if (!r) return;
      const tr = document.createElement('tr');
      const cells = [
        `<span class="rb">${rank + 1}</span>`,
        r.playerName || '',
        r.team || '',
        `<span class="dv">${dist !== undefined ? dist.toFixed(4) : ''}</span>`,
        ...STATS.map(s => {
          const v = r[s.key];
          return typeof v === 'number' ? v.toFixed(1) : '';
        }),
      ];
      cells.forEach(c => {
        const td = document.createElement('td');
        td.innerHTML = c;
        tr.appendChild(td);
      });
      tbody.appendChild(tr);
    });
  }
  function drawBarChart(qi, ti) {
    if (!normData[qi] || !normData[ti]) return;
    const a   = normData[qi];
    const b   = normData[ti];
    const dpr = window.devicePixelRatio || 1;
    const W   = barCanvas.parentElement.clientWidth || 800;
    const rowH   = 34;
    const startY = 18;
    const H      = startY + SK.length * rowH + 20;
    barCanvas.width        = W * dpr;
    barCanvas.height       = H * dpr;
    barCanvas.style.height = H + 'px';
    barCtx.setTransform(dpr, 0, 0, dpr, 0, 0);
    barCtx.clearRect(0, 0, W, H);
    const labelW = 54, gap = 10, padR = 18;
    const halfW  = (W - labelW - gap * 2 - padR) / 2;
    const cx     = labelW + halfW + gap;  
    for (let i = 0; i < SK.length; i++) {
      const cy = startY + i * rowH + rowH / 2;
      barCtx.font         = '500 10px "DM Mono",monospace';
      barCtx.fillStyle    = '#6b7fa3';
      barCtx.textAlign    = 'center';
      barCtx.textBaseline = 'middle';
      barCtx.fillText(SL[i], labelW / 2, cy);
      barCtx.strokeStyle = 'rgba(255,255,255,0.035)';
      barCtx.lineWidth   = 1;
      barCtx.beginPath();
      barCtx.moveTo(labelW, cy);
      barCtx.lineTo(W - padR, cy);
      barCtx.stroke();
      const aW = Math.max(2, Math.round(halfW * a[i]));
      barCtx.fillStyle = 'rgba(249,115,22,0.13)';
      barCtx.fillRect(cx - aW, cy - 6, aW, 12);
      barCtx.fillStyle = '#f97316';
      barCtx.fillRect(cx - aW, cy - 6, 3, 12);
      const bW = Math.max(2, Math.round(halfW * b[i]));
      barCtx.fillStyle = 'rgba(59,130,246,0.13)';
      barCtx.fillRect(cx + gap, cy - 6, bW, 12);
      barCtx.fillStyle = '#3b82f6';
      barCtx.fillRect(cx + gap + bW - 3, cy - 6, 3, 12);
      barCtx.font         = '400 10px "DM Mono",monospace';
      barCtx.textBaseline = 'middle';
      barCtx.fillStyle    = 'rgba(249,115,22,.75)';
      barCtx.textAlign    = 'right';
      barCtx.fillText(a[i].toFixed(2), cx - aW - 5, cy);
      barCtx.fillStyle = 'rgba(59,130,246,.75)';
      barCtx.textAlign = 'left';
      barCtx.fillText(b[i].toFixed(2), cx + gap + bW + 5, cy);
    }
    barCtx.strokeStyle = 'rgba(255,255,255,0.06)';
    barCtx.lineWidth   = 1;
    barCtx.setLineDash([3, 5]);
    barCtx.beginPath();
    barCtx.moveTo(cx, startY - 8);
    barCtx.lineTo(cx, H - 8);
    barCtx.stroke();
    barCtx.setLineDash([]);
  }
  setStatus('Type a player name above — data proxied securely through Next.js.');
})();