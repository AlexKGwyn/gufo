# Qwen3.8 Flash-Next benchmarks

AMD Strix Halo `gfx1151`, 128 GB unified memory. Unsloth `UD-Q4_K_XL` target and
shared-Q8_0 MTP sidecar; Gufo uses adaptive MTP. HTTP, greedy, thinking off.
Gufo pp/tg, concurrency and memory: October 8–9, 2026, production Nix build.
Loading and reference results: September 22–23.
llama.cpp uses `b11069` for AR and `6fcaa16f` for MTP.

Positive gain favors Gufo.
[Quality and measurement details](QUALITY.md#benchmark-method) · [Model identities](artifacts/model-identities.json)

- [Single Strix Halo](#single-strix-halo): Gufo against llama.cpp.
- [Two Strix Halos over RDMA](#two-strix-halos-over-rdma): Gufo RDMA against
  Gufo on one host, deeper contexts, and the full Q8 model.

## Single Strix Halo

Gufo single-user tg: September 27, 2026 (`f797b5b`); pp and other results:
September 22–23. Concurrency uses the unchanged short-context attention path.
llama.cpp uses `b11069` for AR and `6fcaa16f` for MTP. Gufo's prefill matches
the hosts' performance power mode (120 W sustained).

### Single user, autoregressive

Approximately pp2048 / tg128; depth is the cached prefix in tokens.
Context capacities differ between engines; see the measurement details.

<!-- bench:single-ar -->
| Flash-Next Q4 AR<br>Depth (tokens) | Gufo pp (tok/s) | llama.cpp pp (tok/s) | Gain | Gufo tg (tok/s) | llama.cpp tg (tok/s) | Gain |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 1700.52 | 489.59 | +247.3% | 25.95 | 22.20 | +16.9% |
| 4,096 | 1618.60 | 455.24 | +255.5% | 25.84 | 21.21 | +21.8% |
| 8,192 | 1611.53 | 428.85 | +275.8% | 25.96 | 20.40 | +27.3% |
| 12,288 | 1602.99 | 400.21 | +300.5% | 25.94 | 19.64 | +32.1% |
| 16,384 | 1597.41 | 375.89 | +325.0% | 25.90 | 18.95 | +36.7% |
| 32,768 | 1585.98 | 301.31 | +426.4% | 25.78 | 16.54 | +55.9% |
| 65,536 | 1509.10 | 221.94 | +580.0% | 25.40 | 11.62 | +118.6% |
| 131,072 | 1478.66 | 144.78 | +921.3% | 24.92 | 7.98 | +212.3% |
<!-- /bench -->

![Single user, autoregressive](artifacts/charts/single-ar.svg)

### Single user, MTP

pp is the highest measured rate per engine and depth across mixed/repetitive
text, including Gufo predictor catch-up.

<!-- bench:single-mtp -->
| Flash-Next Q4 MTP<br>Depth (tokens) | Gufo pp (tok/s) | llama.cpp pp (tok/s) | Gain pp | Gufo tg mixed (tok/s) | llama.cpp tg mixed (tok/s) | Gain mixed | Gufo tg repetitive (tok/s) | llama.cpp tg repetitive (tok/s) | Gain repetitive |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 1642.85 | 468.76 | +250.5% | 31.32 | 31.73 | -1.3% | 60.39 | 48.12 | +25.5% |
| 4,096 | 1575.60 | 423.79 | +271.8% | 32.48 | 34.48 | -5.8% | 57.66 | 45.71 | +26.1% |
| 8,192 | 1573.54 | 394.98 | +298.4% | 32.38 | 32.20 | +0.6% | 49.28 | 44.31 | +11.2% |
| 12,288 | 1559.07 | 365.91 | +326.1% | 34.68 | 32.32 | +7.3% | 42.15 | 43.53 | -3.2% |
| 16,384 | 1547.52 | 341.98 | +352.5% | 35.09 | 32.07 | +9.4% | 50.58 | 43.66 | +15.8% |
| 32,768 | 1541.35 | 277.61 | +455.2% | 34.20 | 25.41 | +34.6% | 44.54 | 39.28 | +13.4% |
| 65,536 | 1466.98 | 208.04 | +605.1% | 33.49 | 19.65 | +70.4% | 44.45 | 27.75 | +60.2% |
| 131,072 | 1436.89 | 135.42 | +961.1% | 33.07 | 14.63 | +126.0% | 44.26 | 20.96 | +111.2% |
<!-- /bench -->

![Single user, MTP](artifacts/charts/single-mtp.svg)

### Multiple users, autoregressive

Same pp2048 prose prompt as single-user d0, tg128, context 4096 per user.
All sessions prefilled before timed decoding; throughput sums individual rates.
llama.cpp re-evaluates its four-token checkpoint tail.

<!-- bench:multi-ar -->
| Flash-Next Q4 AR<br>Users | Gufo AR (tok/s) | llama.cpp AR (tok/s) | Gain |
| ---: | ---: | ---: | ---: |
| 1 | 25.84 | 22.34 | +15.7% |
| 2 | 45.77 | 37.08 | +23.4% |
| 4 | 76.33 | 54.82 | +39.2% |
| 6 | 95.75 | 65.34 | +46.5% |
| 8 | 108.68 | 68.79 | +58.0% |
<!-- /bench -->

![Multiple users, autoregressive](artifacts/charts/multi-ar.svg)

### Multiple users, MTP

Same pp2048 mixed/repetitive prompts as single-user d0, tg128, context 4096
per user. All sessions prefilled before timed decoding; rates sum individual
request decode rates. C1 cross-checks the single-user table.

<!-- bench:multi-mtp -->
| Flash-Next Q4 MTP<br>Users | Gufo mixed (tok/s) | llama.cpp mixed (tok/s) | Gain | Gufo repetitive (tok/s) | llama.cpp repetitive (tok/s) | Gain |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 31.28 | 31.77 | -1.5% | 60.10 | 46.93 | +28.1% |
| 2 | 52.22 | 42.79 | +22.0% | 84.52 | 52.57 | +60.8% |
| 4 | 76.16 | 48.78 | +56.1% | 131.10 | 48.23 | +171.8% |
| 6 | 96.01 | 52.59 | +82.6% | 145.38 | 48.51 | +199.7% |
| 8 | 107.77 | 61.92 | +74.0% | 162.98 | 58.13 | +180.4% |
<!-- /bench -->

![Multiple users, MTP](artifacts/charts/multi-mtp.svg)

### Loading time

C1, context capacity 262144, MTP. Cold target/sidecar files to HTTP readiness.

<!-- bench:loading -->
| Flash-Next Q4<br>Target | Gufo ready (s) | llama.cpp ready (s) | Gain |
| --- | ---: | ---: | ---: |
| Q4 | 15.45 | 117.83 | +662.7% |
<!-- /bench -->

![Loading time](artifacts/charts/loading.svg)

### Memory occupation

C1, context capacity 133121, AR. Peak memory reported by HIP.

<!-- bench:memory -->
| Flash-Next Q4 AR<br>Workload | Gufo GiB | llama.cpp GiB | Gain |
| --- | ---: | ---: | ---: |
| pp2048 + tg128 | 85.56 | 84.84 | -0.8% |
| 16K prefix, pp4096 + tg128 | 86.71 | 85.31 | -1.6% |
<!-- /bench -->

![Memory occupation](artifacts/charts/memory.svg)

## Two Strix Halos over RDMA

Gufo RDMA runs one model on two such hosts: every layer is split between them,
and they exchange partial sums over RDMA ([how it works](TP2.md)).
The workloads are those of the one-host tables; "Gufo" is the same build on
one host, and gain is Gufo RDMA over it. With twice the memory, single users
also reach depths one host cannot (context capacity 262144). The Q4 single-user
comparisons were measured on September 29, 2026 (`ea76571`) in the balanced power mode (85 W sustained),
so prefill is lower than in the one-host tables above; decode is not
power-bound.

### Single user, autoregressive

<!-- bench:single-ar-tp2 -->
| Flash-Next Q4 AR<br>Depth (tokens) | Gufo pp (tok/s) | Gufo RDMA pp (tok/s) | Gain | Gufo tg (tok/s) | Gufo RDMA tg (tok/s) | Gain |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 1203.95 | 1541.16 | +28.0% | 26.21 | 34.19 | +30.4% |
| 4,096 | 1243.96 | 1480.06 | +19.0% | 25.81 | 34.54 | +33.8% |
| 32,768 | 1134.13 | 1479.06 | +30.4% | 25.79 | 33.92 | +31.5% |
| 65,536 | 1060.00 | 1431.50 | +35.0% | 25.37 | 33.78 | +33.1% |
| 131,072 | 979.47 | 1320.03 | +34.8% | 25.27 | 32.83 | +29.9% |
| 258,048 | — | 1167.63 | — | — | 31.72 | — |
<!-- /bench -->

![Single user, autoregressive, one host and RDMA](artifacts/charts/single-ar-tp2.svg)

### Single user, MTP

<!-- bench:single-mtp-tp2 -->
| Flash-Next Q4 MTP<br>Depth (tokens) | Gufo pp (tok/s) | Gufo RDMA pp (tok/s) | Gain pp | Gufo tg mixed (tok/s) | Gufo RDMA tg mixed (tok/s) | Gain mixed | Gufo tg repetitive (tok/s) | Gufo RDMA tg repetitive (tok/s) | Gain repetitive |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 1334.06 | 1505.96 | +12.9% | 33.00 | 41.20 | +24.8% | 56.25 | 75.58 | +34.4% |
| 4,096 | 1282.50 | 1457.15 | +13.6% | 31.51 | 46.44 | +47.4% | 54.69 | 58.86 | +7.6% |
| 32,768 | 1121.21 | 1432.81 | +27.8% | 33.03 | 45.44 | +37.6% | 39.11 | 53.66 | +37.2% |
| 65,536 | 1089.34 | 1395.77 | +28.1% | 32.51 | 45.05 | +38.6% | 39.45 | 56.53 | +43.3% |
| 131,072 | 1070.54 | 1292.44 | +20.7% | 33.12 | 42.65 | +28.8% | 44.97 | 59.36 | +32.0% |
| 258,048 | — | 1129.95 | — | — | 23.22 | — | — | 50.16 | — |
<!-- /bench -->

![Single user, MTP, one host and RDMA](artifacts/charts/single-mtp-tp2.svg)

### Multiple users, autoregressive

October 2, 2026 (`c19211e`), same production build on one host and two hosts
over USB4 RoCE v2 with write striping. Both hosts use balanced mode and the
recorded fan curves. Same pp2048 prose prompt, tg128, context 4096 per user;
every session prefills before timed decoding. Rates sum individual decode rates.

<!-- bench:multi-ar-tp2 -->
| Flash-Next Q4 AR<br>Users | Gufo AR (tok/s) | Gufo RDMA AR (tok/s) | Gain |
| ---: | ---: | ---: | ---: |
| 1 | 26.59 | 32.93 | +23.8% |
| 2 | 47.17 | 46.66 | -1.1% |
| 4 | 75.60 | 91.60 | +21.2% |
| 6 | 93.61 | 107.83 | +15.2% |
| 8 | 105.12 | 129.10 | +22.8% |
<!-- /bench -->

![Multiple users, autoregressive, one host and RDMA](artifacts/charts/multi-ar-tp2.svg)

### Multiple users, MTP

Same configuration and preparation as the AR table, with mixed/repetitive
prompts. All 129 prepared completions across both topologies and all modes
match fresh AR controls; C1 MTP output and draft counts match fresh MTP controls.

<!-- bench:multi-mtp-tp2 -->
| Flash-Next Q4 MTP<br>Users | Gufo mixed (tok/s) | Gufo RDMA mixed (tok/s) | Gain | Gufo repetitive (tok/s) | Gufo RDMA repetitive (tok/s) | Gain |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 31.99 | 36.34 | +13.6% | 60.62 | 69.70 | +15.0% |
| 2 | 49.66 | 67.00 | +34.9% | 91.28 | 104.74 | +14.7% |
| 4 | 75.41 | 89.64 | +18.9% | 120.15 | 150.81 | +25.5% |
| 6 | 90.30 | 97.63 | +8.1% | 134.06 | 164.76 | +22.9% |
| 8 | 102.63 | 118.40 | +15.4% | 143.90 | 183.70 | +27.7% |
<!-- /bench -->

C1 repetitive RDMA is **69.70 ± 3.57 tok/s** (mean ± sample standard
deviation, four warmed cohorts); all valid samples are retained. Its timing
variability limits a precise gain claim. Other cells use one warmed cohort.

![Multiple users, MTP, one host and RDMA](artifacts/charts/multi-mtp-tp2.svg)

### Full Q8 model

The full Q8_0 target does not fit one host. Single user, pp2048/tg128 by depth
as above; prefill on the left axis, generation on the right.

<!-- bench:tp2-q8 -->
| Flash-Next Q8 RDMA<br>Depth (tokens) | pp (tok/s) | tg AR (tok/s) | tg MTP mixed (tok/s) | tg MTP repetitive (tok/s) |
| ---: | ---: | ---: | ---: | ---: |
| 0 | 916.15 | 31.13 | 41.97 | 68.12 |
| 4,096 | 907.62 | 31.14 | 40.09 | 58.64 |
| 32,768 | 871.49 | 30.77 | 41.06 | 48.29 |
| 65,536 | 690.42 | 30.53 | 41.39 | 51.99 |
| 131,072 | 823.21 | 30.03 | 37.45 | 53.61 |
| 258,048 | 774.02 | 28.77 | 38.04 | 53.21 |
<!-- /bench -->

![Flash-Next Q8 RDMA](artifacts/charts/tp2-q8.svg)
