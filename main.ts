// polymarket-faa-attribution-02 entry: historical-simulation forecaster v2.
//
// For each market and each horizon h = 1..240 minutes after the checkpoint,
// the forecast quantiles are the checkpoint price plus the empirical quantiles
// of the market's own h-minute price changes over its history (72 hours max),
// weighted toward recent windows. The median stays at the checkpoint price:
// realized 4-hour prices are essentially a martingale, and a historical trend
// does not persist over the window, so only the empirical SPREAD is used.
// The spread is widened a little at the 5%/95% tails (pinball punishes a
// too-narrow outer quantile far more than a too-wide one) and scaled by how
// volatile the market has been lately versus its whole history. Cross-market
// pooling was tried and rejected on the practice set: it made things worse.
// Every row is clamped to [0, 1] and non-decreasing; the program is
// deterministic (no randomness, fixed iteration order) and runs in well
// under the 8-second cap.
//
// Tunables (env, defaults are the practice-tuned values):
//   PFA_HL      recency half-life in minutes (0 = none)        [90]
//   PFA_DRIFT   how much of the empirical median trend to keep  [0]
//   PFA_WOUT    widening factor for the 5%/95% quantiles        [1.15]
//   PFA_WIN     widening factor for the 25%/75% quantiles       [1.0]
//   PFA_POOL    shrinkage toward the pooled cross-market shape  [0]
//   PFA_VOL     1 = scale spread by recent-vs-overall volatility [1]
//   PFA_VOLLO / PFA_VOLHI  clamp on the volatility factor       [0.7/1.5]
//
// Usage (outside the grader): MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out node main.ts

import { mkdirSync, readdirSync, readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';

const INPUT_DIR = process.env.MAKEFUN_INPUT_DIR ?? '/input';
const OUTPUT_DIR = process.env.MAKEFUN_OUTPUT_DIR ?? '/output';

/** Scored quantile levels, in the order each row lists them. */
const TAU = [0.05, 0.25, 0.5, 0.75, 0.95];
/** Fallback random-walk spread (the house baseline's) for degenerate history. */
const BASE_SIGMA = 0.0025;
const Z = [-1.644854, -0.67449, 0, 0.67449, 1.644854];
/** Horizons where the empirical change distribution is computed exactly;
    other horizons interpolate between these in sqrt(h). */
const GRID = [1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128, 180, 240];

const env = (k: string, d: number): number => {
  const v = process.env[k];
  return v === undefined || v === '' ? d : Number(v);
};
const HALF_LIFE = env('PFA_HL', 90);
const DRIFT = env('PFA_DRIFT', 0);
const W_OUT = env('PFA_WOUT', 1.15);
const W_IN = env('PFA_WIN', 1.0);
const POOL = env('PFA_POOL', 0);
const USE_VOL = env('PFA_VOL', 1) !== 0;
const VOL_LO = env('PFA_VOLLO', 0.7);
const VOL_HI = env('PFA_VOLHI', 1.5);
/** Recent window for the volatility ratio and the pooling scale, in minutes. */
const RECENT_WINDOW = 360;

interface CaseMarket {
  id: string;
  prices: number[];
}

interface CaseIn {
  horizon_minutes: number;
  quantiles: number[];
  markets: CaseMarket[];
}

/** x clamped to [0,1] and rounded to 6 decimals (0, never -0). */
const price6 = (x: number): number =>
  Math.round(Math.min(1, Math.max(0, x)) * 1e6) / 1e6 + 0;

/** Weighted quantiles of a histogram: keys are values, values are weights. */
function weightedQuantiles(hist: Map<number, number>, levels: number[]): number[] {
  const keys = [...hist.keys()].sort((a, b) => a - b);
  let total = 0;
  for (const k of keys) total += hist.get(k) as number;
  const out: number[] = [];
  let cum = 0;
  let ki = 0;
  for (const lv of levels) {
    const target = lv * total;
    while (ki < keys.length && cum < target) {
      cum += hist.get(keys[ki] as number) as number;
      ki++;
    }
    out.push(keys[Math.max(0, ki - 1)] as number);
  }
  return out;
}

/** 1-minute change dispersion (mean absolute change) over a slice of prices. */
function meanAbsStep(prices: number[], from: number, to: number): number {
  let s = 0;
  let c = 0;
  for (let t = Math.max(1, from); t < to; t++) {
    s += Math.abs((prices[t] as number) - (prices[t - 1] as number));
    c++;
  }
  return c > 0 ? s / c : 0;
}

/**
 * Per-grid-horizon empirical quantile deltas for one market, recency-weighted.
 * Entry j is null when the market's history is too short for that horizon.
 */
function gridDeltas(prices: number[], grid: number[]): (number[] | null)[] {
  const n = prices.length;
  const decay = HALF_LIFE > 0 ? Math.pow(0.5, 1 / HALF_LIFE) : 1;
  const out: (number[] | null)[] = [];
  for (const g of grid) {
    if (n - 1 <= g) {
      out.push(null);
      continue;
    }
    const hist = new Map<number, number>();
    for (let t = g; t < n; t++) {
      const d = (prices[t] as number) - (prices[t - g] as number);
      const w = HALF_LIFE > 0 ? Math.pow(decay, n - 1 - t) : 1;
      hist.set(d, (hist.get(d) ?? 0) + w);
    }
    out.push(weightedQuantiles(hist, TAU));
  }
  return out;
}

/**
 * Pooled cross-market shape: for each grid horizon, the quantiles of the
 * h-minute changes normalized by each market's recent 1-minute step size,
 * over all markets of the case. Returns quantiles in normalized units.
 */
function pooledShape(markets: CaseMarket[], grid: number[]): number[][] {
  const decay = HALF_LIFE > 0 ? Math.pow(0.5, 1 / HALF_LIFE) : 1;
  const scales = markets.map((m) => {
    const n = m.prices.length;
    const s = meanAbsStep(m.prices, n - RECENT_WINDOW, n);
    return s > 1e-6 ? s : 1e-6;
  });
  const out: number[][] = [];
  for (const g of grid) {
    const hist = new Map<number, number>();
    for (let mi = 0; mi < markets.length; mi++) {
      const prices = markets[mi]?.prices as number[];
      const n = prices.length;
      const s = scales[mi] as number;
      for (let t = g; t < n; t++) {
        const d = ((prices[t] as number) - (prices[t - g] as number)) / s;
        const w = HALF_LIFE > 0 ? Math.pow(decay, n - 1 - t) : 1;
        hist.set(d, (hist.get(d) ?? 0) + w);
      }
    }
    out.push(weightedQuantiles(hist, TAU));
  }
  return out;
}

/** Full 240-row forecast for one market: rows of 5 quantile deltas. */
function forecastDeltas(
  prices: number[],
  horizon: number,
  pool: number[][],
  grid: number[],
): number[][] {
  const n = prices.length;
  if (n < 3 || grid.length === 0) {
    const rows: number[][] = [];
    for (let h = 1; h <= horizon; h++) rows.push(Z.map((z) => z * BASE_SIGMA * Math.sqrt(h)));
    return rows;
  }
  const recent = meanAbsStep(prices, n - RECENT_WINDOW, n);
  const overall = meanAbsStep(prices, 0, n);
  let vol = 1;
  if (USE_VOL && overall > 0) vol = Math.min(VOL_HI, Math.max(VOL_LO, recent / overall));
  const scale = recent > 1e-6 ? recent : 1e-6; // market's own step size, for the pooled shape

  const own = gridDeltas(prices, grid);
  const lastH = grid[grid.length - 1] as number;
  // Largest grid index with this market's own data; used to scale short histories.
  let ownMax = own.length - 1;
  while (ownMax > 0 && own[ownMax] === null) ownMax--;
  const widen = (k: number): number => (k === 2 ? 1 : k === 0 || k === 4 ? W_OUT : W_IN);

  /**
   * Median + centered spread at grid index j. The median (drift) and the
   * spread are kept apart: a short history's median extrapolates linearly
   * in h while the spread grows like sqrt(h), and neither may invert the
   * other (scaling a whole quantile vector would break monotonicity).
   */
  const compAt = (j: number): { med: number; sp: number[] } => {
    let med: number;
    let sp: number[];
    if (own[j] !== null) {
      const o = own[j] as number[];
      med = o[2] as number;
      sp = o.map((v) => v - med);
    } else {
      const g = grid[j] as number;
      const g0 = grid[ownMax] as number;
      const o0 = own[ownMax] as number[];
      const med0 = o0[2] as number;
      med = med0 * (g / g0);
      const fs = Math.sqrt(g / g0);
      sp = o0.map((v) => (v - med0) * fs);
    }
    const p = pool[j] as number[];
    const pmed = p[2] as number;
    const out: number[] = [];
    for (let k = 0; k < 5; k++) {
      const poolSp = ((p[k] as number) - pmed) * scale;
      const s = (sp[k] as number) * (1 - POOL) + poolSp * POOL;
      out.push(s * widen(k) * vol);
    }
    return { med: med * DRIFT, sp: out };
  };
  const toDeltas = (c: { med: number; sp: number[] }): number[] =>
    c.sp.map((v) => c.med + v);

  const rows: number[][] = [];
  for (let h = 1; h <= horizon; h++) {
    let d: number[];
    if (h <= lastH) {
      const s = Math.sqrt(h);
      let j = 0;
      while (j + 1 < grid.length && Math.sqrt(grid[j + 1] as number) <= s) j++;
      if (j + 1 >= grid.length || (grid[j] as number) === h) {
        d = toDeltas(compAt(j));
      } else {
        const s0 = Math.sqrt(grid[j] as number);
        const s1 = Math.sqrt(grid[j + 1] as number);
        const f = (s - s0) / (s1 - s0);
        const a = compAt(j);
        const b = compAt(j + 1);
        const med = a.med + (b.med - a.med) * f;
        d = a.sp.map((v, k) => med + v + ((b.sp[k] as number) - v) * f);
      }
    } else {
      // Beyond the data: drift grows linearly, spread like sqrt(h).
      const fs = Math.sqrt(h / lastH);
      const fd = h / lastH;
      const base = compAt(grid.length - 1);
      d = base.sp.map((v, k) => base.med * fd + v * fs);
    }
    rows.push(d);
  }
  return rows;
}

mkdirSync(OUTPUT_DIR, { recursive: true });
const inputs = readdirSync(INPUT_DIR).filter((f) => f.endsWith('.in')).sort();
for (const file of inputs) {
  const id = file.slice(0, -'.in'.length);
  const c = JSON.parse(readFileSync(join(INPUT_DIR, file), 'utf8')) as CaseIn;
  if (c.quantiles.length !== TAU.length || c.quantiles.some((t, j) => t !== TAU[j])) {
    throw new Error(`${file} asks for quantiles ${JSON.stringify(c.quantiles)}, not ${JSON.stringify(TAU)}`);
  }
  const H = c.horizon_minutes;
  // Grid usable for the pooled shape: limited by the LONGEST history.
  const maxN = Math.max(...c.markets.map((m) => m.prices.length));
  const grid = GRID.filter((g) => g <= Math.max(1, maxN - 2));
  const pool = grid.length > 0 ? pooledShape(c.markets, grid) : [];
  const forecasts = c.markets.map((m) => {
    const deltas = forecastDeltas(m.prices, H, pool, grid);
    const x = m.prices[m.prices.length - 1] as number;
    const rows = deltas.map((d) => {
      const r = d.map((v) => price6(x + v));
      return [...r].sort((a, b) => a - b); // enforce non-decreasing
    });
    return { id: m.id, q: rows };
  });
  const out = { forecasts };
  writeFileSync(join(OUTPUT_DIR, `${id}.out`), `${JSON.stringify(out)}\n`);
}
