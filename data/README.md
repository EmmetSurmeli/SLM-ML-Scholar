# Local data

Place user-supplied strict UTF-8 corpora in `data/raw/`. Derived local artifacts
may go in `data/processed/`. Empty and malformed UTF-8 files are rejected
explicitly.

The repository intentionally does not download or commit datasets. Before using
any corpus, verify its license and record its provenance. Both data directories
are ignored except for their placeholder files.

Controlled language-model experiments split raw text chronologically before
fitting a tokenizer. Character vocabularies and BPE merge rules see training
text only; the fixed byte tokenizer has no fitting phase. Train and validation
are then encoded independently, so no token or shifted target crosses the split
boundary.

The default text policy performs no Unicode or whitespace normalization. Corpus
metadata records the logical source, raw UTF-8 SHA-256 hash, character and byte
counts, split policy, tokenizer type/hash, vocabulary size, and isolated token
counts. Checkpoints store identity metadata, not full corpus contents.

Generated tokenizer JSON, corpora, checkpoints, and experiment summaries belong
under ignored paths unless a deliberately tiny test fixture is reviewed for
commit.
