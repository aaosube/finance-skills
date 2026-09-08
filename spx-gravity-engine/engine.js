'use strict';

/**
 * SPX Gravity Engine — auditable reconstruction core
 *
 * Design constraints:
 * 1) Standard mathematics is implemented directly.
 * 2) Experimental project rules are isolated and explicitly labelled.
 * 3) No silent/arbitrary weights are allowed.
 * 4) Stage 1 (structure) never consumes Stage 2 market-driver confirmation.
 * 5) Stage 3 monetization never changes Stage 1 targets.
 */

const CONTRACT_MULTIPLIER = 100;
const { scenarioRepricedGammaFlip } = require('./gamma_scenario');

function assertFinite(name, value) {
  if (!Number.isFinite(value)) throw new Error(`${name} must be finite`);
}

function normalCdf(x) {
  // Abramowitz-Stegun approximation; sufficient for diagnostics.
  const sign = x < 0 ? -1 : 1;
  const z = Math.abs(x) / Math.sqrt(2);
  const t = 1 / (1 + 0.3275911 * z);
  const erf = 1 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * Math.exp(-z * z);
  return 0.5 * (1 + sign * erf);
}

/**
 * Standard GEX convention used by the project audit:
 * GEX = gamma * OI * 100 * spot^2 * 0.01
 * Calls positive, puts negative.
 */
function gexPerOnePercent({ gamma, openInterest, spot, type }) {
  assertFinite('gamma', gamma);
  assertFinite('openInterest', openInterest);
  assertFinite('spot', spot);
  if (gamma < 0 || openInterest < 0 || spot <= 0) throw new Error('gamma/OI must be nonnegative and spot positive');
  const side = String(type).toLowerCase();
  if (side !== 'call' && side !== 'put') throw new Error(`Unknown option type: ${type}`);
  const signed = side === 'call' ? 1 : -1;
  return signed * gamma * openInterest * CONTRACT_MULTIPLIER * spot * spot * 0.01;
}

function aggregateByStrike(rows, spot) {
  const map = new Map();
  for (const r of rows) {
    if ([r.strike, r.gamma, r.openInterest ?? r.oi].some(v => v === null || v === undefined || typeof v === 'boolean' || String(v).trim() === '')) {
      throw new Error('strike, gamma and OI must be present numeric values');
    }
    const strike = Number(r.strike);
    const gamma = Number(r.gamma);
    const openInterest = Number(r.openInterest ?? r.oi);
    assertFinite('strike', strike);
    if (strike <= 0) throw new Error('strike must be positive');
    const gex = gexPerOnePercent({ gamma, openInterest, spot, type: r.type });
    if (!map.has(strike)) {
      map.set(strike, {
        strike,
        callGex: 0,
        putGex: 0,
        netGex: 0,
        callOi: 0,
        putOi: 0,
        rows: 0,
      });
    }
    const a = map.get(strike);
    if (String(r.type).toLowerCase() === 'call') {
      a.callGex += Math.abs(gex);
      a.callOi += openInterest;
    } else {
      a.putGex -= Math.abs(gex);
      a.putOi += openInterest;
    }
    a.netGex = a.callGex + a.putGex;
    a.rows += 1;
  }
  return [...map.values()].sort((a, b) => a.strike - b.strike);
}

function findWalls(strikes) {
  if (!strikes.length) return { callWall: null, putWall: null };
  const callWall = strikes.filter(x => x.callGex > 0).reduce((best, x) => (!best || x.callGex > best.callGex ? x : best), null);
  const putWall = strikes.filter(x => x.putGex < 0).reduce((best, x) => (!best || x.putGex < best.putGex ? x : best), null);
  return {
    callWall: callWall ? { strike: callWall.strike, gex: callWall.callGex } : null,
    putWall: putWall ? { strike: putWall.strike, gex: putWall.putGex } : null,
  };
}

/**
 * IMPORTANT: this is NOT a full scenario-repriced gamma flip.
 * It only detects a strike-to-strike sign transition in current net GEX.
 * A true gamma flip requires recomputing option gamma under alternate spot scenarios.
 */
function netGexSignFlipProxy(strikes, spot) {
  const flips = [];
  for (let i = 1; i < strikes.length; i++) {
    const a = strikes[i - 1];
    const b = strikes[i];
    if (a.netGex === 0) flips.push(a.strike);
    if (a.netGex * b.netGex < 0) {
      const w = Math.abs(a.netGex) / (Math.abs(a.netGex) + Math.abs(b.netGex));
      flips.push(a.strike + (b.strike - a.strike) * w);
    }
  }
  if (!flips.length) return null;
  return flips.reduce((best, x) => Math.abs(x - spot) < Math.abs(best - spot) ? x : best, flips[0]);
}

/** Distance-only reachability diagnostic. Never used as a structural weight by default. */
function distanceReachability(strike, spot, expectedMove) {
  assertFinite('strike', strike);
  assertFinite('spot', spot);
  assertFinite('expectedMove', expectedMove);
  if (expectedMove <= 0) throw new Error('expectedMove must be > 0');
  return Math.exp(-Math.abs(strike - spot) / expectedMove);
}

/**
 * Optional normal-move hit diagnostic. This is a model assumption, not an empirical probability.
 * expectedMove is interpreted as 1-sigma absolute move only when caller explicitly enables it.
 */
function normalHitDiagnostic(strike, spot, expectedMove) {
  assertFinite('strike', strike);
  assertFinite('spot', spot);
  assertFinite('expectedMove', expectedMove);
  if (expectedMove <= 0) throw new Error('expectedMove must be > 0');
  const z = Math.abs(strike - spot) / expectedMove;
  return 2 * (1 - normalCdf(z));
}

function nearest(strikes, spot, predicate) {
  const xs = strikes.filter(predicate);
  if (!xs.length) return null;
  return xs.reduce((best, x) => Math.abs(x.strike - spot) < Math.abs(best.strike - spot) ? x : best, xs[0]);
}

/**
 * Deterministic fallback structural diagnostic; NOT the historical canonical Stage-1 formula.
 * It exists only so a snapshot can be sanity-checked without hidden weights.
 */
function fallbackDiagnosticMap(strikes, spot, expectedMove) {
  const reachable = expectedMove > 0
    ? strikes.filter(x => Math.abs(x.strike - spot) <= expectedMove)
    : strikes.slice();
  const lowerPool = reachable.filter(x => x.strike < spot && x.netGex < 0);
  const upperPool = reachable.filter(x => x.strike > spot && x.netGex > 0);
  const lower = lowerPool.reduce((best, x) => (!best || x.netGex < best.netGex ? x : best), null);
  const upper = upperPool.reduce((best, x) => (!best || x.netGex > best.netGex ? x : best), null);
  const flip = netGexSignFlipProxy(strikes, spot);
  const pivotRow = nearest(strikes, spot, () => true);
  return {
    status: 'DIAGNOSTIC_ONLY_NON_CANONICAL',
    lower: lower?.strike ?? null,
    pivot: flip ?? pivotRow?.strike ?? null,
    upper: upper?.strike ?? null,
    note: 'Do not use as the historical Stage-1 forecast until original formula is reconciled.'
  };
}

function experimentalProjectionLevels(anchor, expectedMove, multipliers) {
  if (!Array.isArray(multipliers) || multipliers.length === 0) return [];
  assertFinite('anchor', anchor);
  assertFinite('expectedMove', expectedMove);
  if (expectedMove <= 0) throw new Error('expectedMove must be > 0');
  return multipliers.map(m => {
    const k = Number(m);
    assertFinite('projection multiplier', k);
    if (k <= 0) throw new Error('projection multiplier must be > 0');
    return {
      multiplier: k,
      lower: anchor - expectedMove / k,
      upper: anchor + expectedMove / k,
      status: 'EXPERIMENTAL_PROJECT_RULE'
    };
  });
}

function stage2Confirmation(drivers = {}) {
  const entries = Object.entries(drivers).filter(([, v]) => v === 'up' || v === 'down');
  if (!entries.length) return { state: 'DATA-LIMITED', up: 0, down: 0, used: [] };
  const up = entries.filter(([, v]) => v === 'up').length;
  const down = entries.filter(([, v]) => v === 'down').length;
  let state = 'NEUTRAL';
  if (up && down) state = Math.abs(up - down) <= 1 ? 'CONFLICT' : (up > down ? 'CONFIRMED-UP' : 'CONFIRMED-DOWN');
  else if (up) state = 'CONFIRMED-UP';
  else if (down) state = 'CONFIRMED-DOWN';
  return { state, up, down, used: entries.map(([k]) => k) };
}

function runEngine(snapshot) {
  const spot = Number(snapshot.spot);
  const expectedMove = Number(snapshot.expectedMove);
  assertFinite('spot', spot);
  if (!Array.isArray(snapshot.rows) || snapshot.rows.length === 0) throw new Error('rows[] is required');

  const strikes = aggregateByStrike(snapshot.rows, spot);
  const walls = findWalls(strikes);
  const flipProxy = netGexSignFlipProxy(strikes, spot);
  const reachability = Number.isFinite(expectedMove) && expectedMove > 0
    ? strikes.map(x => ({ strike: x.strike, r: distanceReachability(x.strike, spot, expectedMove) }))
    : [];

  const probabilityDiagnostic = snapshot.enableNormalProbabilityDiagnostic === true && expectedMove > 0
    ? strikes.map(x => ({ strike: x.strike, p: normalHitDiagnostic(x.strike, spot, expectedMove) }))
    : [];

  const rawMap = snapshot.approvedRawMap
    ? { status: 'APPROVED_EXTERNAL_STAGE1', ...snapshot.approvedRawMap }
    : fallbackDiagnosticMap(strikes, spot, expectedMove);

  const experimental = Number.isFinite(expectedMove) && expectedMove > 0
    ? experimentalProjectionLevels(Number(snapshot.projectionAnchor ?? spot), expectedMove, snapshot.experimentalMultipliers ?? [])
    : [];

  const scenarioGamma = snapshot.scenarioGamma
    ? scenarioRepricedGammaFlip(snapshot.rows, {
        spot,
        scenarioMin: Number(snapshot.scenarioGamma.scenarioMin),
        scenarioMax: Number(snapshot.scenarioGamma.scenarioMax),
        steps: snapshot.scenarioGamma.steps,
        rate: snapshot.scenarioGamma.rate,
        dividendYield: snapshot.scenarioGamma.dividendYield,
      })
    : null;

  return {
    engine: 'SPX Gravity Engine reconstruction',
    version: '0.1-audit',
    spot,
    expectedMove: Number.isFinite(expectedMove) ? expectedMove : null,
    stage1: {
      rawMap,
      walls,
      netGexSignFlipProxy: flipProxy,
      aggregate: strikes,
      reachabilityDiagnostic: reachability,
      experimentalProjectionLevels: experimental,
      probabilityDiagnostic,
      scenarioRepricedGammaFlip: scenarioGamma,
    },
    stage2: stage2Confirmation(snapshot.marketDrivers),
    stage3: {
      status: 'SEPARATE_NOT_USED_FOR_TARGET_SELECTION',
      note: 'Monetization/contract selection must be evaluated after Stage 1+2.'
    },
    qualityFlags: [
      'TRUE_GAMMA_FLIP_REQUIRES_SCENARIO_REPRICING',
      ...(snapshot.approvedRawMap ? [] : ['HISTORICAL_STAGE1_FORMULA_NOT_YET_RECONCILED']),
      ...(probabilityDiagnostic.length ? ['NORMAL_PROBABILITY_IS_DIAGNOSTIC_ASSUMPTION_ONLY'] : []),
      ...(experimental.length ? ['EXPERIMENTAL_MULTIPLIERS_ISOLATED_FROM_CANONICAL_MAP'] : []),
      ...(scenarioGamma?.status === 'DATA_INSUFFICIENT' ? ['SCENARIO_GAMMA_DATA_INSUFFICIENT'] : []),
      ...(scenarioGamma?.status === 'SCENARIO_REPRICED_DIAGNOSTIC' ? ['SCENARIO_GAMMA_IS_DIAGNOSTIC_NOT_CANONICAL'] : []),
    ]
  };
}

module.exports = {
  gexPerOnePercent,
  aggregateByStrike,
  findWalls,
  netGexSignFlipProxy,
  distanceReachability,
  normalHitDiagnostic,
  experimentalProjectionLevels,
  stage2Confirmation,
  runEngine,
};

if (require.main === module) {
  const fs = require('fs');
  const path = process.argv[2];
  if (!path) {
    console.error('Usage: node engine.js snapshot.json');
    process.exit(2);
  }
  const snapshot = JSON.parse(fs.readFileSync(path, 'utf8'));
  process.stdout.write(JSON.stringify(runEngine(snapshot), null, 2) + '\n');
}
