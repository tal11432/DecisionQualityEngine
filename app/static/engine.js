/* Decision Score engine in JavaScript: a port of research/dq/dscore (core.py, score.py) and app/api.py. StatsBomb units (120 x 80), the team in possession attacks toward x = 120.
   The parity with the Python code is tested (tests/test_js_parity.py). Works in the browser (window.DS) and in node (module.exports). */
(function (root) {
  const GX = 120, SC = [105 / 120, 68 / 80], SHOT_MIN_X = 60, DEFAULT_GK = [118, 40], CLIP = { nn_opp: 9.1, gk_d: 15.6, gk_lat: 4.9 };
  const M = root.DS_MODELS;                     // from models.js (window.DS_MODELS)
  const hyp = (a, b) => Math.hypot(a, b), sum = a => a.reduce((s, v) => s + v, 0);
  function sgm(m, X) { let z = m.b[0]; for (let i = 0; i < X.length; i++) z += m.b[i + 1] * (X[i] - m.mu[i]) / m.sd[i]; return 1 / (1 + Math.exp(-z)); }
  const feat = (m, f) => m.F.map(k => f[k]);
  const ang = (x, y) => Math.abs(Math.atan2(44 - y, GX - x) - Math.atan2(36 - y, GX - x));
  function tri(p, s) {
    const f = (ax, ay, bx, by, cx, cy) => (ax - cx) * (by - cy) - (bx - cx) * (ay - cy);
    const d1 = f(p[0], p[1], s[0], s[1], GX, 36), d2 = f(p[0], p[1], GX, 36, GX, 44), d3 = f(p[0], p[1], GX, 44, s[0], s[1]);
    return !(((d1 < 0) || (d2 < 0) || (d3 < 0)) && ((d1 > 0) || (d2 > 0) || (d3 > 0)));
  }
  function roundEven(x) { const r = Math.round(x); return (Math.abs(x % 1) === 0.5 && r % 2 !== 0) ? r - 1 : r; }   // Python's round()
  const zone = x => x < 40 ? 'own third' : (x < 80 ? 'middle' : 'attacking');
  // ---------------- shot
  function shotFeatures(a, O, gk) {
    const dd = O.map(o => hyp(o[0] - a[0], o[1] - a[1])), v = [GX - a[0], 40 - a[1]], dist = hyp(v[0], v[1]), an = ang(a[0], a[1]);
    return { dist, angle: an, header: 0, n_tri: O.filter(o => tri(o, a)).length, gk_d: Math.min(hyp(GX - gk[0], 40 - gk[1]), CLIP.gk_d),
      gk_lat: Math.min(Math.abs(v[0] * (gk[1] - a[1]) - v[1] * (gk[0] - a[0])) / (dist + 1e-9), CLIP.gk_lat), n_opp2: dd.filter(d => d <= 2).length,
      nn_opp: Math.min(Math.min(...dd), CLIP.nn_opp), da: dist * an, ldist: Math.log(dist) };
  }
  function shotValue(a, O, gk) { const f = shotFeatures(a, O, gk), xg = sgm(M.xg, feat(M.xg, f)), cont = sgm(M.cont, feat(M.cont, f)); return { xg, cont, total: xg + cont, f }; }
  // ---------------- pass
  function xpHazard(a, end, O) {
    const vx = end[0] - a[0], vy = end[1] - a[1], L = hyp(vx, vy); if (L < 1e-6) return 1;
    const ux = vx / L, uy = vy / L, n = Math.ceil(L / 2), hz = M.hz; let logp = 0;
    let pp = Math.min(Math.min(...O.map(o => hyp(o[0] - a[0], o[1] - a[1]))), 20);
    for (let k = 0; k < n; k++) {
      const s = k * 2, p0 = [a[0] + s * ux, a[1] + s * uy], p1 = [a[0] + (s + 2) * ux, a[1] + (s + 2) * uy], mid = [(p0[0] + p1[0]) / 2, (p0[1] + p1[1]) / 2];
      const abx = p1[0] - p0[0], aby = p1[1] - p0[1], ab2 = abx * abx + aby * aby; let dmin = 1e9, n3 = 0;
      for (const o of O) { const t = Math.min(1, Math.max(0, ((o[0] - p0[0]) * abx + (o[1] - p0[1]) * aby) / ab2)), d = hyp(p0[0] + t * abx - o[0], p0[1] + t * aby - o[1]); if (d < dmin) dmin = d; if (d <= 3) n3++; }
      dmin = Math.min(dmin, 20);
      const row = [s, s * s, dmin, Math.log1p(dmin), dmin / (s + 2), n3, mid[0], Math.abs(mid[1] - 40), pp, 1];
      let l = hz.b[0]; for (let i = 0; i < 10; i++) l += hz.b[i + 1] * (row[i] - hz.mu[i]) / hz.sd[i];
      logp += Math.log1p(-1 / (1 + Math.exp(-l)));
    }
    return Math.exp(logp);
  }
  function targetFeatures(a, O, gk, p) {
    const dp = O.map(o => hyp(o[0] - p[0], o[1] - p[1])), distP = hyp(GX - p[0], 40 - p[1]), anP = ang(p[0], p[1]);
    return { dist: distP, angle: anP, x: p[0], d2: distP * distP, ad: anP * distP, ldist: Math.log(Math.max(distP, 1)), n_tri: O.filter(o => tri(o, p)).length,
      n3: dp.filter(d => d <= 3).length, n6: dp.filter(d => d <= 6).length, nn: dp.length ? Math.min(...dp) : 99, gk_d: hyp(GX - gk[0], 40 - gk[1]), plen: hyp(p[0] - a[0], p[1] - a[1]), fwd: p[0] - a[0], pdist: hyp(GX - a[0], 40 - a[1]) };
  }
  function offsideStatus(marginM, px, nVis) {
    if (px <= 60 || marginM >= 1.0) return 'certain onside';
    if (marginM > -1.0) return 'DOUBT (within 1 m of the line)';
    if (nVis < 11) return 'DOUBT (offside by the visible players, but a defender is out of view)';
    return 'certain offside';
  }
  function offsideLine(a, O) { const ox = O.map(o => o[0]).sort((u, v) => u - v); return Math.max(ox.length > 1 ? ox[ox.length - 2] : ox[ox.length - 1], a[0]); }
  // ---------------- cost of losing the ball (turnover) and the marked assumption A1
  function around(p, O, T) {
    const dt = T.length ? T.map(t => hyp(t[0] - p[0], t[1] - p[1])) : [99], dO = O.length ? O.map(o => hyp(o[0] - p[0], o[1] - p[1])) : [99];
    return { n_opp3: dO.filter(d => d <= 3).length, n_opp6: dO.filter(d => d <= 6).length, nn_opp: Math.min(Math.min(...dO), 30), n_mate10: dt.filter(d => d <= 10).length, nn_mate: Math.min(Math.min(...dt), 30),
      n_opp_ahead: O.filter(o => o[0] < p[0]).length, n_cover: T.filter(t => t[0] < p[0]).length };
  }
  function lossDanger(p, O, T, kind) {
    const c = around(p, O, T), distOwn = hyp(p[0], p[1] - 40), an = Math.abs(Math.atan2(44 - p[1], p[0]) - Math.atan2(36 - p[1], p[0])), adv = c.n_opp_ahead - c.n_cover;
    const f = Object.assign({}, c, { dist_own: distOwn, ang_own: an, lx: p[0], d2: distOwn * distOwn, ad: an * distOwn, k_pass: kind === 'pass' ? 1 : 0, k_dribble: 0, k_dispossessed: 0 });
    const X = M.L.F.map(k => f[k]); let L = sgm(M.L.opp, X) - sgm(M.L.own, X);
    const flagged = adv >= 2 && p[0] <= 70; if (flagged) L = Math.max(L, M.C);
    return { L, flagged };
  }
  const qToScore = q => roundEven(100 * Math.min(1, Math.max(0, (q + M.C) / (M.Qhi + M.C))));
  // ---------------- all options of a moment (the same as dscore.score.all_options) and the API response
  function allOptions(a, mates, O, gk) {
    const opts = [];
    if (a[0] >= SHOT_MIN_X) { const s = shotValue(a, O, gk); opts.push({ kind: 'shot', v: s.total, xg: s.xg, cont: s.cont, status: 'certain onside', target: [GX, 40], counter_assumed: false, f: s.f }); }
    const line = offsideLine(a, O);
    for (const p of mates) {
      const d = hyp((p[0] - a[0]) * SC[0], (p[1] - a[1]) * SC[1]); if (d < 3 || d > 40) continue;
      const ft = targetFeatures(a, O, gk, p), xa = sgm(M.xa, feat(M.xa, ft)), xp = xpHazard(a, p, O), margin = (line - p[0]) * 105 / 120, status = offsideStatus(margin, p[0], O.length);
      const ld = lossDanger(p, O, mates, 'pass');
      opts.push({ kind: 'pass', v: xp * xa - (1 - xp) * ld.L, xp, xa, d, status, target: p, loss: ld.L, counter_assumed: ld.flagged });
    }
    const t = lossDanger(a, O, mates, 'pass'); opts.push({ kind: 'turnover', v: -t.L, status: 'certain onside', target: a, counter_assumed: t.flagged });
    for (const o of opts) o.score = qToScore(o.v);
    return opts;
  }
  function checkPt(p, name) {
    if (!p || p.length < 2 || !isFinite(+p[0]) || !isFinite(+p[1])) throw new Error('bad point for ' + name);
    const x = +p[0], y = +p[1]; if (x < 0 || x > 120 || y < 0 || y > 80) throw new Error(name + ' is outside the pitch'); return [x, y];
  }
  function scoreSituation(payload) {
    if (!payload.carrier) throw new Error('place the ball carrier');
    const a = checkPt(payload.carrier, 'carrier'), mates = (payload.teammates || []).map(m => checkPt(m, 'teammate')), opps = (payload.opponents || []).map(o => checkPt(o, 'opponent'));
    const gkIn = payload.keeper ? checkPt(payload.keeper, 'keeper') : null, O = opps.concat(gkIn ? [gkIn] : []);
    if (!O.length) throw new Error('place at least one opponent (or the opposing keeper)');
    const thr = payload.threshold === undefined ? 26 : +payload.threshold, top = Math.max(1, Math.min(parseInt(payload.top === undefined ? 10 : payload.top), 10)), gk = gkIn || DEFAULT_GK;
    const opts = allOptions(a, mates, O, gk), z = zone(a[0]), warnings = [];
    if (z !== 'attacking') warnings.push('the ball carrier is in the ' + z + ' zone: in this zone the score is not validated (differences between options are thousandths of xG); read the order, not the numbers');
    if (a[0] < SHOT_MIN_X) warnings.push('no shot option below x = 60');
    if (!gkIn) warnings.push('no opposing keeper marked: assumed at (118, 40) for the shot');
    if (O.length < 8) warnings.push('only ' + O.length + ' opponents marked: with few opponents the offside line and the pass risk are less reliable');
    const out = []; let offside = 0, turnover = null;
    for (const o of opts) {
      if (o.kind === 'turnover') { turnover = { score: o.score, q: o.v, counter_assumed: o.counter_assumed }; continue; }
      if (o.status === 'certain offside') { offside++; continue; }
      const row = { kind: o.kind, score: o.score, q: o.v, status: o.status, counter_assumed: o.counter_assumed, target: o.target };
      if (o.kind === 'shot') Object.assign(row, { xg: o.xg, cont: o.cont, dist_m: o.f.dist * 105 / 120, angle_deg: o.f.angle * 180 / Math.PI, n_tri: o.f.n_tri, gk_m: o.f.gk_d * 105 / 120 });
      else Object.assign(row, { xp: o.xp, chain: o.xa, dist_m: o.d, loss: o.loss });
      out.push(row);
    }
    out.sort((u, v) => v.q - u.q); const kept = out.filter(r => r.score >= thr).slice(0, top); kept.forEach((r, i) => r.rank = i + 1);
    return { options: kept, n_options: out.length, n_offside: offside, turnover, zone: z, warnings, threshold: thr };
  }
  const api = { scoreSituation, qToScore, allOptions, xpHazard, shotValue };
  if (typeof module !== 'undefined' && module.exports) module.exports = api; else root.DS = api;
})(typeof window !== 'undefined' ? window : globalThis);
