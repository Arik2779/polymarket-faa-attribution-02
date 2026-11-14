# Polymarket FAA outage attribution — 4-hour price forecaster

Entry for make.fun's `polymarket-faa-attribution-02`: forecast each market's
Yes price, minute by minute, 4 hours ahead of each checkpoint, as five
quantiles (5/25/50/75/95%).

## How it works

`main.ts` (Node 22, standard library only) reads a case file, and for every
market and every horizon h = 1..240 minutes:

- takes the empirical distribution of that market's own h-minute price
  changes over its history (up to 72 hours), weighted toward recent windows
  (8-hour half-life) — prices mostly sit flat and then jump, so the real
  change distribution beats a Gaussian random walk;
- keeps the median at the checkpoint price (realized 4-hour prices are
  essentially a martingale; historical trends do not persist over the
  window) and uses only the empirical spread;
- widens the 5%/95% tails a little (pinball loss punishes a too-narrow
  outer quantile far more than a too-wide one) and scales the spread by
  how volatile the market has been lately versus its whole history.

Local practice score (mean pinball loss x100, against Polymarket's public
price history): **0.2818**, versus the house baseline's 0.363589
(about 22% better). Per case: p00 0.476, p01 0.304, p02 0.382, p03 0.274,
p04 0.208, p05 0.115.

## Run

```sh
mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out node main.ts
```

Reads every `*.in` in the input directory and writes one `<id>.out` per
case. Deterministic: two runs produce byte-identical output.
