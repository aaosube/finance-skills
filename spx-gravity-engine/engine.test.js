'use strict';
const assert = require('assert');
const {
  gexPerOnePercent,
  runEngine,
} = require('./engine');

// GEX sign + scale invariant.
const spot = 7650;
const call = gexPerOnePercent({ gamma: 0.01, openInterest: 1000, spot, type: 'call' });
const put = gexPerOnePercent({ gamma: 0.01, openInterest: 1000, spot, type: 'put' });
assert(call > 0);
assert.strictEqual(put, -call);
assert.strictEqual(call, 0.01 * 1000 * 100 * spot * spot * 0.01);

const snapshot = {
  spot,
  expectedMove: 40,
  rows: [
    { strike: 7625, type: 'put', gamma: 0.015, openInterest: 4000 },
    { strike: 7650, type: 'call', gamma: 0.025, openInterest: 2000 },
    { strike: 7650, type: 'put', gamma: 0.030, openInterest: 2500 },
    { strike: 7675, type: 'call', gamma: 0.018, openInterest: 5000 },
  ],
  experimentalMultipliers: [2.6, 2.14, 1.6],
  marketDrivers: { ES: 'up', NQ: 'up', VIX: 'down' },
};

const out = runEngine(snapshot);
assert.strictEqual(out.stage1.rawMap.status, 'DIAGNOSTIC_ONLY_NON_CANONICAL');
assert.strictEqual(out.stage3.status, 'SEPARATE_NOT_USED_FOR_TARGET_SELECTION');
assert(out.qualityFlags.includes('HISTORICAL_STAGE1_FORMULA_NOT_YET_RECONCILED'));
assert(out.qualityFlags.includes('EXPERIMENTAL_MULTIPLIERS_ISOLATED_FROM_CANONICAL_MAP'));
assert.strictEqual(out.stage1.experimentalProjectionLevels.length, 3);

// Stage 2 is independent: changing drivers must not move Stage 1 raw map.
const down = runEngine({ ...snapshot, marketDrivers: { ES: 'down', NQ: 'down', VIX: 'up' } });
assert.deepStrictEqual(down.stage1.rawMap, out.stage1.rawMap);

console.log('SPX Gravity Engine reconstruction invariants: PASS');
