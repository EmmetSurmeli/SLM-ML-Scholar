# Milestone 1 engineering and mathematical audit

Audit date: 2026-07-23

Scope: every tracked source file, test, configuration, document, experiment,
ignore rule, and repository metadata entry present before Milestone 2.

## Checks performed

1. **Chronological isolation.** Traced `prepare_bigram_dataset` from raw text
   splitting through tokenization and pair construction. Confirmed the
   tokenizer is fit on `text[:split_index]` only, validation is encoded
   afterward, and a validation-only character raises rather than entering the
   vocabulary.
2. **Boundary exclusion.** Verified examples are constructed independently as
   `train_tokens[:-1] -> train_tokens[1:]` and
   `validation_tokens[:-1] -> validation_tokens[1:]`. The pair from
   `text[split_index - 1]` to `text[split_index]` is absent from both sets.
3. **Vocabulary/inference contract.** Confirmed characters are unique and
   sorted by Unicode code point, vocabulary JSON requires the same ordering,
   and encoding seed or inference text with an unseen character raises an
   indexed error. This no-unknown-token limitation is stated in the README.
4. **Softmax/cross-entropy.** Re-derived max-shifted softmax and indexed
   log-sum-exp cross-entropy. Tested logits of magnitude \(10^{300}\) and
   opposite-sign `float64` limits. Probabilities remain finite and normalized;
   a mathematically unrepresentable loss may correctly become positive
   infinity.
5. **Mean-loss scaling.** Re-derived
   \((p-\operatorname{onehot}(y))/B\) and verified the division occurs only in
   `softmax_cross_entropy_backward`. Duplicating every example in a batch
   leaves the weight gradient unchanged.
6. **Repeated rows.** Verified `np.add.at(grad_weights, input_ids,
   grad_logits)` sums every occurrence. A repeated-ID fixture produces the
   hand-computed row gradient.
7. **Weight decay.** Verified the legacy SGD implementation matches its
   documented coupled L2 update
   \(W \leftarrow W-\eta(\nabla W+\lambda W)\), using the pre-update parameter
   in the decay term.
8. **Finite differences.** Inspected every mutation path and confirmed each
   scalar perturbation is restored in a `finally` block. A bit-exact snapshot
   comparison after exhaustive checking passes.
9. **Persistence.** Reloaded weights and compared logits bit-for-bit. The
   audit found that the original checkpoint inferred vocabulary size from the
   weight shape but did not store an explicit model configuration.
10. **Reproducibility.** Re-created samplers and generation calls with equal
    seeds and compared multiple draws/output strings exactly. The training
    script uses independent deterministic RNG streams for updates and each
    evaluation split.
11. **Perplexity.** Verified NaN rejection, `-inf -> 0`, and positive infinity
    or excessive finite loss capping at `float64` maximum rather than calling
    an overflowing exponential.
12. **Validation.** Exercised wrong ranks, mismatched shapes, empty arrays,
    out-of-range IDs, unknown characters, non-finite values, invalid numeric
    ranges, booleans masquerading as integers, missing configuration fields,
    and malformed JSON.
13. **Repository hygiene.** Compared `git ls-files` with ignored status.
    Corpora, processed data, run artifacts, NPZ checkpoints, Python caches,
    pytest/Ruff caches, virtual environments, coverage data, and `.DS_Store`
    are not tracked. Only `.gitkeep` placeholders are tracked under ignored
    data/output directories.
14. **Claims review.** The README explicitly calls the bigram an educational
    one-character-context baseline, says it is neither an SLM nor a paper
    assistant, reports only one measured fixture, and prohibits unsupported
    performance/capability claims.

## Issues discovered and fixes made

### 1. Checkpoint configuration was implicit

The version-1 NPZ held only a format version and weights. Although vocabulary
size is recoverable from the square matrix, the checkpoint did not explicitly
state its reconstruction contract.

Fix: checkpoint version 2 stores a sorted `model_config_json` containing
vocabulary size and dtype. Loading validates this metadata against the weight
matrix. The loader retains version-1 compatibility.

### 2. Some floating-point APIs silently coerced inputs

Loss, generation, and bigram-backward paths used `np.asarray(...,
dtype=np.float64)`. Integer logits or gradients could therefore be accepted,
and a caller could not observe a dtype mismatch.

Fix: these paths now require floating arrays, accept only `float32` or
`float64` where applicable, preserve input dtype in generalized losses, and
reject mismatched gradient/parameter dtypes. Legacy SGD received the same
explicit policy.

### 3. Classification loss was restricted to rank two

The original loss was correct for `(batch, classes)` but would not directly
support future language-model logits.

Fix: the indexed implementation now treats the final axis as classes and
supports `(batch, sequence, vocabulary)` or any non-empty leading shape.
Targets must equal the leading shape. Flattening is internal, no one-hot tensor
is allocated, and averaging occurs over all target positions exactly once.
Existing bigram behavior remains unchanged.

### 4. Evaluation mode was not restored generically

`evaluate` always called `model.train()` afterward, even if its caller supplied
an already-evaluation-mode model.

Fix: the prior mode is recorded and restored in a `finally` block.

### 5. Extreme opposite-sign finite logits could emit an overflow warning

Subtracting `-max_float - max_float` represents as `-inf`; this is the correct
softmax limiting value but can emit an expected subtraction warning.

Fix: only that expected overflow warning is suppressed around max subtraction.
Invalid arithmetic still raises, and output finiteness remains validated.

## Tests added

Milestone 1 audit coverage now includes:

- exact source-side pair lists and explicit boundary-pair absence
- duplicate-batch invariance for one-time mean scaling
- hand-computed legacy coupled weight decay
- bit-exact parameter restoration after finite differences
- evaluation-mode preservation
- perplexity overflow/NaN behavior
- malformed configuration types and ranges
- opposite `float64` limits
- float32 preservation and integer-logit rejection
- versioned checkpoint configuration

These tests run alongside all original Milestone 1 tests and the Milestone 2
suite.

## Unresolved limitations

- Character tokenization has no unknown token. A validation-only or
  inference-only character is rejected deliberately.
- The chronological boundary pair is discarded, which loses one example but
  guarantees no target crosses the split.
- Minibatches sample with replacement and validation loss is a reproducible
  estimate, not an exhaustive epoch metric.
- Bigram checkpoints still do not contain optimizer/RNG resume state. Training
  configuration and artifact paths remain in `run_summary.json`.
- The model has one character of context and no semantic or paper-grounding
  capability.
- `float64` cross-entropy can become infinity when the true loss exceeds the
  representable range; this is reported rather than disguised.

## Verification commands and results

Baseline-before-change commands:

```bash
python3 -m ruff check .
python3 -m ruff format --check .
python3 -m pytest -q
python3 experiments/train_bigram.py --config configs/bigram_small.json
```

Baseline result: Ruff passed, `27 passed in 0.15s`, and the 300-step smoke run
reproduced best validation loss `1.5488034950125846` with perplexity
`4.705836256201863` at step 300.

Final post-Milestone-2 verification used:

```bash
python3 -m ruff check .
python3 -m ruff format --check .
python3 -m pytest -q
python3 experiments/train_bigram.py --config configs/bigram_small.json
python3 experiments/train_mlp_xor.py
git status --short --ignored
git ls-files
```

Final results:

- Ruff lint: all checks passed.
- Ruff format check: 44 files already formatted.
- Pytest: `90 passed in 0.31s`.
- Bigram smoke run: identical best validation loss
  `1.5488034950125846`, perplexity `4.705836256201863`, and best step 300.
- Bigram checkpoint: version 2 configuration reloaded as
  `{"vocabulary_size": 23, "dtype": "float64"}`.
- XOR: loss `0.6865651569496622 -> 7.506981534793376e-06`,
  predictions `[0, 1, 1, 0]`, four of four correct.
- MLP checkpoint: configuration, 42 parameters, and predictions reloaded
  exactly.
- Adam checkpoint: first/second moments and `step_count == 1000` reloaded.
- Git hygiene: generated outputs and every observed cache are ignored; the
  tracked-file query found no corpus, checkpoint, generated output, cache,
  `.DS_Store`, or virtual-environment entry.
