'use strict';
const assert = require('assert');
const { blackScholesGamma, scenarioRepricedGammaFlip } = require('./gamma_scenario');

const g = blackScholesGamma({ spot: 100, strike: 100, iv: 0.2, timeYears: 30 / 365 });
assert.ok(g > 0 && Number.isFinite(g));

const insufficient = scenarioRepricedGammaFlip([{ type: 'call', strike: 100, openInterest: 100, gamma: 0.01 }], { spot: 100, scenarioMin: 90, scenarioMax: 110 });
assert.strictEqual(insufficient.status, 'DATA_INSUFFICIENT');
assert.strictEqual(insufficient.flip, null);

const chain = [
  { type: 'put', strike: 95, openInterest: 2600, iv: 0.24, daysToExpiry: 14 },
  { type: 'put', strike: 100, openInterest: 800, iv: 0.22, daysToExpiry: 14 },
  { type: 'call', strike: 100, openInterest: 700, iv: 0.22, daysToExpiry: 14 },
  { type: 'call', strike: 105, openInterest: 2800, iv: 0.24, daysToExpiry: 14 },
];
const result = scenarioRepricedGammaFlip(chain, { spot: 100, scenarioMin: 90, scenarioMax: 110, steps: 401 });
assert.strictEqual(result.status, 'SCENARIO_REPRICED_DIAGNOSTIC');
assert.strictEqual(result.usedRows, 4);
assert.ok(result.flip === null || (Number.isFinite(result.flip) && result.flip >= 90 && result.flip <= 110));
assert.strictEqual(result.profile.length, 401);
console.log('gamma_scenario tests passed');
