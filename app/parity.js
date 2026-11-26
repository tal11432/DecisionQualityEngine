// node app/parity.js cases.json : compares the JavaScript engine with the Python outputs stored in cases.json
globalThis.window = {}; require('./static/models.js'); const DS = require('./static/engine.js'); const cases = JSON.parse(require('fs').readFileSync(process.argv[2], 'utf8'));
let bad = [], maxq = 0, nopt = 0;
for (const c of cases) {
  let r; try { r = DS.scoreSituation(c.input); } catch (e) { bad.push([c.id, 'js error ' + e.message]); continue; }
  const p = c.output; if (r.n_options !== p.n_options || r.n_offside !== p.n_offside || r.zone !== p.zone || r.options.length !== p.options.length) { bad.push([c.id, 'counts', r.n_options, p.n_options, r.n_offside, p.n_offside, r.options.length, p.options.length]); continue; }
  if (Math.abs(r.turnover.score - p.turnover.score) > 1 || Math.abs(r.turnover.q - p.turnover.q) > 1e-7) bad.push([c.id, 'turnover', r.turnover, p.turnover]);
  if (JSON.stringify(r.warnings) !== JSON.stringify(p.warnings)) bad.push([c.id, 'warnings']);
  r.options.forEach((o, i) => { const q = p.options[i]; nopt++; const dq = Math.abs(o.q - q.q); maxq = Math.max(maxq, dq);
    if (o.kind !== q.kind || dq > 1e-7 || Math.abs(o.score - q.score) > 1 || o.status !== q.status || o.counter_assumed !== q.counter_assumed) bad.push([c.id, i, o.kind, q.kind, o.q, q.q, o.score, q.score, o.status, q.status]); });
}
console.log(JSON.stringify({ cases: cases.length, options_compared: nopt, max_abs_diff_q: maxq, mismatches: bad.length, first: bad.slice(0, 3) })); process.exit(bad.length ? 1 : 0);
