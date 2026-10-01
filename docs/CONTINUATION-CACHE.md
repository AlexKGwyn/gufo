# Continuation Cache

Status: describes `main` as of 2026-10-01

This is what the `kvcache` issue label refers to. The name is a convenient
shorthand rather than a description: the cache does not store attention
key/value tensors. It stores opaque, model-private continuation snapshots, and
almost every design consequence below follows from that one fact.

Operator-facing behaviour and flags are specified in
[the server contract](SERVER.md#hip-execution). This document explains the
model behind them, so that changes to the cache can be argued about from stated
invariants instead of rediscovered.

Claims below are marked **(measured)** where they come from running the server
and **(source)** where they come only from reading the implementation.

## What the cache stores

`ContinuationCache` owns `ContinuationState` (mutable, per-session) and
`ContinuationSnapshot` (immutable, retained). Both are produced and consumed by
the model runner; the cache deliberately cannot inspect or copy their bytes.

For Qwen3.8-27B that payload happens to be attention KV. For
Qwen3.8-Flash-Next it also includes GDN recurrent state, the indexer, position
and sampling state. The cache does not know the difference, which is the point:
one retention, admission, eviction and persistence mechanism serves every
model.

What it buys: a single cache implementation, correct for hybrid-recurrent
models, with no per-model retention logic.

What it forfeits: **a retained snapshot cannot be truncated to an arbitrary
token.** The cache can only reuse a checkpoint that is an exact prefix of the
new prompt. It computes and reports the longest common prefix it found
(`cache_common_prefix_tokens`), but it cannot resume from a position it never
captured.

That limitation is a property of the abstraction, not of the models. The same
full re-prefill occurs on Qwen3.8-27B, which is plain attention and could in
principle be truncated **(measured)**. Worth knowing before concluding that a
mid-history edit is unavoidable for a hybrid model: llama.cpp resumes those
same edits on Flash-Next by keeping periodic checkpoints along the prompt
rather than by truncating recurrent state **(measured)**. See #331.

## The unit of retention is a checkpoint, not a conversation

A request retains up to two checkpoints:

- the **reused frontier**, frozen before prefill mutates the state;
- the request's own boundary or complete prompt.

Two per conversation, against an entry table of `sessions x 2`
(`continuation_cache.cpp`, constructor). So the number of *conversations* the
cache can hold is approximately `--sessions`, not twice it **(source, matching
measurement)**.

This is the most commonly misread part of the configuration. `--sessions` is
usually chosen for request concurrency, but it also bounds how many distinct
conversations stay resumable. Exceeding it does not degrade gradually: with
round-robin traffic the entry about to be needed is always the one just
evicted, so reuse collapses from ~95% to 0% when conversations exceed sessions
by one **(measured)**. See #341.

The server reports the limit at startup:

```text
event=snapshot_cache_configured sessions=2 snapshot_entries=4 retained_conversations=2 capacity_bytes=99007139840
```

## Invariants that constrain the design

These are the constraints that make otherwise-odd behaviour necessary. Change
them only deliberately.

**A snapshot can only be captured where the state actually sits.** The runner
serialises the state as it is; there is no way to capture position *N* while
the state is at position *M*. Retaining a checkpoint at a stable boundary
therefore requires prefill to *stop* at that boundary. A warm continuation that
wants both to reuse a frontier and to retain its own boundary cannot do the
whole suffix in one pass. This is the reason the capture sequence exists at
all, and the reason #335 is not fixable by simply moving the capture point.

**The reused frontier must be frozen before prefill.** Prefill mutates the
leased state in place, so a frontier that a peer request might branch from has
to be captured first. This is what the clamp in
`TextRunnerPool::Request::Impl`'s constructor is for; it is not an oversight.

**Eviction cannot skip a victim.** `RemoveEntry` returning false abandons the
eviction loop rather than trying the next candidate, in both tiers
**(source)**. In practice the store recovers once the underlying condition
clears **(measured)**; see the closed #346 for why the obvious worry about this
does not hold.

**Admission is advisory, never fatal.** A refused snapshot is a skipped
optimisation. Requests must complete normally when capture or reservation
fails.

## Two tiers

**In-memory** (`ContinuationCache`, always on). Exact-prefix reuse and live
continuations within one process lifetime. It does not learn a boundary between
two prompts that merely share a prefix.

**On disk** (`ContinuationDiskStore`, opt-in via `--cache-disk`). Restart-safe,
and additionally *learns* shared-prefix boundaries: when several prompts share
a long prefix and then diverge, it can capture a checkpoint at the divergence
point so later conversations resume from it. This is why a system prompt shared
across conversations is reusable with `--cache-disk` and not without
**(measured)**. The boundary is not learned on first sight; in one run it
became usable from the fifth conversation **(measured)**. See #267.

Disk entries are evicted by **global LRU on last access**, with no awareness of
conversation, prefix subtree, or cost to rebuild **(source, matching
measurement)**. One active run whose checkpoints are all recent will displace
every other conversation in age order. See #275.

## Budgets

Three separate limits, each able to bind first.

| Limit | Default | Set by |
| --- | --- | --- |
| Retained snapshot bytes (RAM) | `MemAvailable / 2` | not configurable |
| Disk cache bytes | 8 GiB | `--cache-disk-bytes` |
| Disk staging bytes | smallest of 1 GiB, `MemAvailable / 8`, the disk budget | `--cache-disk-staging-bytes` |

The RAM budget (`HostSnapshotBudgetBytes`, `text_model_runner.cpp`) is sampled
*after* the model and sessions are allocated, and halved. It therefore shrinks
as `--context` and `--sessions` grow, which is exactly when snapshots are
largest. On Flash-Next at 262144 context a single snapshot reached 5.70 GB
against a 7.76 GB budget, so a second could not be retained **(measured)**. See
#343.

Snapshot size scales with retained tokens and differs sharply by model: roughly
0.5 GB at 5k tokens on Flash-Next, and 3.7 GB at 24k tokens on Qwen3.8-27B
**(measured)**. The disk staging default of 1 GiB is therefore below a single
27B checkpoint at moderate depth, which silently reduces `--cache-disk` to a
no-op unless raised. See #259.

## What invalidates reuse

Anything that changes the token prefix. In practice:

- **Tool definitions** — adding, removing or reordering them changes the prompt
  head and invalidates everything after it.
- **System prompt** — including anything volatile embedded in it, such as a
  timestamp or working directory.
- **Response schema** — `response_format` is rendered into the prompt.
- **Images** — changing, removing or moving an earlier image invalidates
  checkpoints after it; appending a new one does not.
- **Reasoning the client cannot replay** — thinking is on by default for Qwen,
  and `reasoning_content` is not part of the OpenAI schema, so an ordinary
  client omits it when sending the conversation back. The retained checkpoint
  then stops advancing for the rest of the session **(measured)**. See #335.

Sampling parameters do **not** invalidate reuse: `temperature`, `top_p`,
`seed`, `max_tokens`, penalties and `stop` all reuse the full prefix
**(measured)**. Neither does streaming: streamed and non-streamed requests
reuse identically and interoperate within a conversation **(measured)**.

`cache_prompt: false` bypasses lookup for one request; the result can still
populate the cache.

## Reading the logs

| Line | Meaning |
| --- | --- |
| `event=snapshot_cache_configured` | retained capacity at startup |
| `event=snapshot action=removed reason=entry_capacity` | a retained prefix was evicted because every entry was taken; the server holds fewer conversations than the workload rotates through |
| `event=snapshot action=skipped reason=byte_capacity` | a checkpoint did not fit the RAM budget |
| `event=disk_cache action=removed reason=lru` | a disk entry was evicted to stay inside `--cache-disk-bytes` |
| `event=disk_cache action=skipped reason=staging_capacity` | a checkpoint exceeded `--cache-disk-staging-bytes` and was never written |

Per-request outcomes appear in the completion log and in `usage.gufo`:
`cache_hit`, `cache_miss_reason`, `cache_common_prefix_tokens`,
`cache_checkpoint_tokens`, plus `prompt_n` and `cache_n` in llama.cpp-compatible
`timings`. Miss reasons are `no_checkpoint`, `prefix_changed`, `input_changed`
and `disabled`.

A large `cache_common_prefix_tokens` on a miss means the server found a long
agreeing prefix and could not resume from it, which is the signature of a
mid-history edit or an evicted checkpoint rather than a genuinely new prompt.

## Sizing guidance

- Set `--sessions` from the number of conversations in rotation, not from
  request concurrency. One side request on `--sessions 1` evicts the main
  conversation **(measured)**.
- With `--cache-disk`, raise `--cache-disk-staging-bytes` above a single
  checkpoint for the model and context in use, or nothing is written.
- Expect the first few conversations sharing a prefix to miss while the disk
  boundary is learned.
- Have clients return `reasoning_content`, or disable thinking, until #335 is
  resolved.

## Verifying changes

- `tools/serving/check-continuation.py` covers cancellation, reasoning replay,
  images and restart persistence.
- Per-model serving tests cover reuse within one conversation.
- `tests/cli/continuation_cache_test.cpp`,
  `tests/cli/continuation_disk_store_test.cpp` and
  `tests/cli/text_model_runner_test.cpp` cover retention, admission, eviction
  and the event sinks without a model.

Prefix reuse across turns, conversations and clients is not yet covered by a
checked-in harness.

## Known gaps

Tracked under #336: #337, #338, #339, #340, #341, #343, and the upstream
reports #331, #335, #259, #267, #275, #300, #318.
