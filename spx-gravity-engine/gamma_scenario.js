'use strict';

/** Scenario-repriced gamma exposure diagnostics. */
const CONTRACT_MULTIPLIER = 100;
const SQRT_2PI = Math.sqrt(2 * Math.PI);

function finite(name, value) {
  const x = Number(value);
  if (!Number.isFinite(x)) throw new Error(`${name} must be finite`);
  return x;
}
function normalPdf(x) { return Math.exp(-0.5 * x * x) / SQRT_2PI; }
function blackScholesGamma({ spot, strike, iv, timeYears, rate = 0, dividendYield = 0 }) {
  const S = finite('spot', spot), K = finite('strike', strike), sigma = finite('iv', iv), T = finite('timeYears', timeYears);
  const r = finite('rate', rate), q = finite('dividendYield', dividendYield);
  if (S <= 0 || K <= 0) throw new Error('spot and strike must be > 0');
  if (sigma <= 0 || sigma > 5) throw new Error('iv must be a decimal in (0, 5]; percent-form IV is not accepted');
  if (T <= 0) throw new Error('timeYears must be > 0');
  const rootT = Math.sqrt(T);
  const d1 = (Math.log(S / K) + (r - q + 0.5 * sigma * sigma) * T) / (sigma * rootT);
  return Math.exp(-q * T) * normalPdf(d1) / (S * sigma * rootT);
}
function rowTimeYears(row) {
  if (Number.isFinite(Number(row.timeToExpiryYears)) && Number(row.timeToExpiryYears) > 0) return Number(row.timeToExpiryYears);
  if (Number.isFinite(Number(row.minutesToExpiry)) && Number(row.minutesToExpiry) > 0) return Number(row.minutesToExpiry) / (365 * 24 * 60);
  if (Number.isFinite(Number(row.daysToExpiry)) && Number(row.daysToExpiry) > 0) return Number(row.daysToExpiry) / 365;
  throw new Error('one positive time field is required: timeToExpiryYears, minutesToExpiry, or daysToExpiry');
}
function prepareRows(rows, defaults = {}) {
  if (!Array.isArray(rows) || rows.length === 0) throw new Error('rows[] is required');
  return rows.map((row, index) => {
    try {
      const type = String(row.type ?? row.optionType ?? row.option_type).toLowerCase();
      if (type !== 'call' && type !== 'put') throw new Error(`unknown option type ${type}`);
      const iv = finite('iv', row.iv ?? row.impliedVolatility ?? row.implied_volatility);
      const strike = finite('strike', row.strike);
      const openInterest = finite('openInterest', row.openInterest ?? row.open_interest ?? row.oi);
      if (openInterest < 0) throw new Error('openInterest must be >= 0');
      return { type, iv, strike, openInterest, timeYears: rowTimeYears(row), rate: Number.isFinite(Number(row.rate)) ? Number(row.rate) : Number(defaults.rate ?? 0), dividendYield: Number.isFinite(Number(row.dividendYield ?? row.dividend_yield)) ? Number(row.dividendYield ?? row.dividend_yield) : Number(defaults.dividendYield ?? 0) };
    } catch (error) {
      throw new Error(`row ${index}: ${error.message}`);
    }
  });
}
function netGexAtSpot(preparedRows, scenarioSpot) {
  const S = finite('scenarioSpot', scenarioSpot);
  if (S <= 0) throw new Error('scenarioSpot must be > 0');
  let net = 0;
  for (const row of preparedRows) {
    const gamma = blackScholesGamma({ spot: S, strike: row.strike, iv: row.iv, timeYears: row.timeYears, rate: row.rate, dividendYield: row.dividendYield });
    const sign = row.type === 'call' ? 1 : -1;
    net += sign * gamma * row.openInterest * CONTRACT_MULTIPLIER * S * S * 0.01;
  }
  return net;
}
function interpolateZero(a, b) {
  if (a.netGex === 0) return a.spot;
  if (b.netGex === 0) return b.spot;
  const denom = Math.abs(a.netGex) + Math.abs(b.netGex);
  const w = denom === 0 ? 0.5 : Math.abs(a.netGex) / denom;
  return a.spot + (b.spot - a.spot) * w;
}
function scenarioRepricedGammaFlip(rows, config = {}) {
  const spot = finite('spot', config.spot), scenarioMin = finite('scenarioMin', config.scenarioMin), scenarioMax = finite('scenarioMax', config.scenarioMax);
  const steps = Number(config.steps ?? 121);
  if (!(scenarioMin > 0 && scenarioMax > scenarioMin)) throw new Error('scenarioMin/scenarioMax must define a positive increasing range');
  if (!Number.isInteger(steps) || steps < 3 || steps > 5001) throw new Error('steps must be an integer in [3, 5001]');
  let prepared;
  try { prepared = prepareRows(rows, config); }
  catch (error) { return { status: 'DATA_INSUFFICIENT', flip: null, reason: error.message, usedRows: 0, qualityFlags: ['NO_PARTIAL_CHAIN_REPRICING'] }; }
  const profile = [];
  for (let i = 0; i < steps; i++) {
    const s = scenarioMin + (scenarioMax - scenarioMin) * i / (steps - 1);
    profile.push({ spot: s, netGex: netGexAtSpot(prepared, s) });
  }
  const crossings = [];
  for (let i = 1; i < profile.length; i++) {
    const a = profile[i - 1], b = profile[i];
    if (a.netGex === 0 || b.netGex === 0 || a.netGex * b.netGex < 0) {
      const zero = interpolateZero(a, b);
      if (!crossings.some(x => Math.abs(x - zero) < 1e-9)) crossings.push(zero);
    }
  }
  const flip = crossings.length ? crossings.reduce((best, x) => Math.abs(x - spot) < Math.abs(best - spot) ? x : best, crossings[0]) : null;
  return { status: 'SCENARIO_REPRICED_DIAGNOSTIC', flip, crossings, profile, usedRows: prepared.length, assumptions: { model: 'BLACK_SCHOLES_GAMMA', ivConvention: 'DECIMAL_ONLY', callSign: 'POSITIVE', putSign: 'NEGATIVE', contractMultiplier: CONTRACT_MULTIPLIER, scenarioMin, scenarioMax, steps }, qualityFlags: ['DEALER_SIGN_CONVENTION_IS_MODEL_ASSUMPTION', 'STATIC_IV_ACROSS_SPOT_SCENARIOS', 'STATIC_OI_ACROSS_SPOT_SCENARIOS'] };
}
module.exports = { normalPdf, blackScholesGamma, rowTimeYears, prepareRows, netGexAtSpot, scenarioRepricedGammaFlip };
