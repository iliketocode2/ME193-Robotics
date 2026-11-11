/* Grid-world behavior cloning: the rules, with no page code in it so it can be tested on its own.
 *
 * A robot stands on a grid facing one of four directions. It has three yes/no sensors that each look
 * one cell away (F front, L left, R right) and, optionally, a goal direction (G).
 * It has three actions: F (step forward), L (turn left, then step), R (turn right, then step).
 * A "card" is one (observation, action) pair. A policy is a lookup table built from cards.
 */
const GW = (function () {
  const DX = [0, 1, 0, -1];   // heading 0 = up, 1 = right, 2 = down, 3 = left
  const DY = [-1, 0, 1, 0];
  const ACTIONS = ['F', 'L', 'R'];
  const ACTION_WORD = { F: 'forward', L: 'turn left', R: 'turn right' };
  const BEARINGS = ['A', 'L', 'R', 'B'];
  const BEARING_WORD = { A: 'ahead', L: 'left', R: 'right', B: 'behind' };
  const MAX_STEPS = 150;
  const LOOP_VISITS = 3;

  const MAP_SOURCES = [
    { name: 'Open room', heading: 1, rows: [
      '##########',
      '#S.......#',
      '#..##....#',
      '#..##..#.#',
      '#.....##.#',
      '#S......G#',
      '##########'] },
    { name: 'Classroom', heading: 1, rows: [
      '##############',
      '#S...........#',
      '#.##..##..##.#',
      '#.##..##..##.#',
      '#S...........#',
      '#.##..##..##.#',
      '#.##..##..##.#',
      '#S.........G.#',
      '##############'] },
    { name: 'Maze', heading: 1, rows: [
      '#############',
      '#S#.....#...#',
      '#.#.###.#.#.#',
      '#.#.#...#.#.#',
      '#.#.#.###.#.#',
      '#...#.....#.#',
      '###.#######.#',
      '#S.........G#',
      '#############'] },
  ];

  function parseMap(src) {
    const rows = src.rows;
    const w = rows[0].length, h = rows.length;
    const starts = [];
    let goal = null;
    const grid = rows.map((row, y) => {
      if (row.length !== w) throw new Error('map "' + src.name + '" row ' + y + ' has the wrong length');
      return row.split('').map((c, x) => {
        if (c === 'S') starts.push({ x, y, h: src.heading });
        if (c === 'G') goal = { x, y };
        return c === '#' ? '#' : '.';
      });
    });
    return { name: src.name, w, h, grid, starts, goal, heading: src.heading };
  }
  const MAPS = MAP_SOURCES.map(parseMap);

  function blocked(map, x, y) {
    return y < 0 || y >= map.h || x < 0 || x >= map.w || map.grid[y][x] === '#';
  }

  function bearing(map, s) {
    const dx = map.goal.x - s.x, dy = map.goal.y - s.y;
    const fwd = dx * DX[s.h] + dy * DY[s.h];
    const rt = dx * DX[(s.h + 1) % 4] + dy * DY[(s.h + 1) % 4];
    if (fwd === 0 && rt === 0) return 'A';
    if (Math.abs(fwd) >= Math.abs(rt)) return fwd > 0 ? 'A' : 'B';
    return rt > 0 ? 'R' : 'L';
  }

  function sense(map, s) {
    const f = s.h, l = (s.h + 3) % 4, r = (s.h + 1) % 4;
    return {
      F: blocked(map, s.x + DX[f], s.y + DY[f]) ? 1 : 0,
      L: blocked(map, s.x + DX[l], s.y + DY[l]) ? 1 : 0,
      R: blocked(map, s.x + DX[r], s.y + DY[r]) ? 1 : 0,
      G: bearing(map, s),
    };
  }

  function key(obs, useGoal) {
    return 'F' + obs.F + ' L' + obs.L + ' R' + obs.R + (useGoal ? ' G' + obs.G : '');
  }

  function allObs(useGoal) {
    const out = [];
    for (let F = 0; F < 2; F++) for (let L = 0; L < 2; L++) for (let R = 0; R < 2; R++) {
      if (useGoal) BEARINGS.forEach(G => out.push({ F, L, R, G }));
      else out.push({ F, L, R, G: 'A' });
    }
    return out;
  }

  function step(map, s, action) {
    let h = s.h;
    if (action === 'L') h = (h + 3) % 4;
    else if (action === 'R') h = (h + 1) % 4;
    const nx = s.x + DX[h], ny = s.y + DY[h];
    if (blocked(map, nx, ny)) return { s: { x: s.x, y: s.y, h }, bumped: true };
    return { s: { x: nx, y: ny, h }, bumped: false };
  }

  function atGoal(map, s) { return s.x === map.goal.x && s.y === map.goal.y; }

  /* An episode: one attempt from one start. */
  function newEpisode(map, startIdx) {
    const st = map.starts[startIdx % map.starts.length];
    return { s: { x: st.x, y: st.y, h: st.h }, steps: 0, bumps: 0, over: false, outcome: null,
             seen: {}, startIdx: startIdx % map.starts.length };
  }

  function advance(ep, map, action, opts) {
    opts = opts || {};
    if (ep.over) return ep;
    const r = step(map, ep.s, action);
    ep.s = r.s;
    ep.steps += 1;
    if (r.bumped) ep.bumps += 1;
    if (atGoal(map, ep.s)) { ep.over = true; ep.outcome = 'goal'; return ep; }
    if (opts.detectLoops) {
      const k = ep.s.x + ',' + ep.s.y + ',' + ep.s.h;
      ep.seen[k] = (ep.seen[k] || 0) + 1;
      if (ep.seen[k] >= LOOP_VISITS) { ep.over = true; ep.outcome = 'looping'; return ep; }
    }
    if (ep.steps >= (opts.maxSteps || MAX_STEPS)) { ep.over = true; ep.outcome = 'timeout'; }
    return ep;
  }

  /* Cards -> lookup table. Each entry counts the cards per action (and per ink: expert / dagger). */
  function buildTable(cards, useGoal) {
    const table = new Map();
    cards.forEach((c, i) => {
      const k = key(c.obs, useGoal);
      let e = table.get(k);
      if (!e) {
        e = { key: k, obs: { F: c.obs.F, L: c.obs.L, R: c.obs.R, G: c.obs.G },
              counts: { F: 0, L: 0, R: 0 }, ink: { F: { expert: 0, dagger: 0 }, L: { expert: 0, dagger: 0 }, R: { expert: 0, dagger: 0 } },
              lastIdx: { F: -1, L: -1, R: -1 }, total: 0 };
        table.set(k, e);
      }
      e.counts[c.action] += 1;
      e.ink[c.action][c.src === 'dagger' ? 'dagger' : 'expert'] += 1;
      e.lastIdx[c.action] = i;
      e.total += 1;
    });
    table.forEach(e => {
      let best = null;
      ACTIONS.forEach(a => {
        if (e.counts[a] === 0) return;
        if (best === null || e.counts[a] > e.counts[best] ||
            (e.counts[a] === e.counts[best] && e.lastIdx[a] > e.lastIdx[best])) best = a;
      });
      e.action = best;
      e.conflict = ACTIONS.filter(a => e.counts[a] > 0).length > 1;
    });
    return table;
  }

  function makePolicy(cards, useGoal, fallback) {
    return { table: buildTable(cards, useGoal), useGoal: !!useGoal, fallback: fallback || 'nearest', nCards: cards.length };
  }

  function distance(obs, entryObs, useGoal) {
    let d = (obs.F !== entryObs.F) + (obs.L !== entryObs.L) + (obs.R !== entryObs.R);
    if (useGoal && obs.G !== entryObs.G) d += 1;
    return d;
  }

  /* What does the policy do for this observation?  known=false means no card had this reading. */
  function decide(policy, obs) {
    const e = policy.table.get(key(obs, policy.useGoal));
    if (e) return { action: e.action, known: true, conflict: e.conflict, matched: e.key };
    if (policy.fallback === 'forward' || policy.table.size === 0) return { action: 'F', known: false, conflict: false, matched: null };
    let best = null, bd = 99;
    policy.table.forEach(en => {
      const d = distance(obs, en.obs, policy.useGoal);
      if (d < bd || (d === bd && en.total > best.total) || (d === bd && en.total === best.total && en.key < best.key)) { best = en; bd = d; }
    });
    return { action: best.action, known: false, conflict: false, matched: best.key };
  }

  /* A scripted expert that sees the whole map (shortest path over position and heading).
     Used for tests and as stand-in data for the instructor. */
  function expertAction(map, s) {
    const id = (p) => (p.y * map.w + p.x) * 4 + p.h;
    const parent = new Map();
    const q = [s];
    parent.set(id(s), null);
    for (let i = 0; i < q.length; i++) {
      const cur = q[i];
      if (atGoal(map, cur)) {
        let node = cur, first = null;
        while (parent.get(id(node)) !== null) { const p = parent.get(id(node)); first = p.action; node = p.from; }
        return first;
      }
      for (const a of ACTIONS) {
        const r = step(map, cur, a);
        if (r.bumped) continue;
        if (parent.has(id(r.s))) continue;
        parent.set(id(r.s), { from: cur, action: a });
        q.push(r.s);
      }
    }
    return null;
  }

  function expertCards(map, startIdx) {
    const ep = newEpisode(map, startIdx);
    const cards = [];
    while (!ep.over) {
      const a = expertAction(map, ep.s);
      if (a === null) break;
      cards.push({ obs: sense(map, ep.s), action: a, src: 'expert', step: ep.steps + 1, pos: { x: ep.s.x, y: ep.s.y, h: ep.s.h } });
      advance(ep, map, a);
    }
    return cards;
  }

  function runPolicy(map, startIdx, policy) {
    const ep = newEpisode(map, startIdx);
    while (!ep.over) {
      const d = decide(policy, sense(map, ep.s));
      advance(ep, map, d.action, { detectLoops: true });
    }
    return ep;
  }

  return { DX, DY, ACTIONS, ACTION_WORD, BEARINGS, BEARING_WORD, MAX_STEPS, LOOP_VISITS, MAPS, parseMap,
           blocked, bearing, sense, key, allObs, step, atGoal, newEpisode, advance, buildTable,
           makePolicy, decide, expertAction, expertCards, runPolicy };
})();
if (typeof module !== 'undefined') module.exports = GW;
