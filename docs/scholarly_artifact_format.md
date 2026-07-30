# Scholarly artifact format

Package 1.1.1 writes scholarly artifacts as UTF-8 JSON using format version 1.
Writes are atomic; pickle and object arrays are not used.

Every artifact contains:

- `artifact_type`
- `artifact_format_version`
- `package_version`
- `index_sha256`
- `document_hashes`
- complete `analysis_configuration`
- JSON `payload`
- `artifact_sha256`

Supported types are paper analysis, role-filtered paper-analysis view, notation
glossary, equation analysis, methodology, experiments, structured summary,
reproduction checklist, paper comparison, and research-gap worksheet.

Loading is transactional. The loader reconstructs the full outer object before
returning it and rejects:

- unknown format versions or a malformed recorded package version;
- malformed keys or non-canonical values;
- artifact hash changes;
- an index hash mismatch;
- missing source documents;
- source-document hash changes;
- citation ranges outside a document or section;
- citation text/hash/line/page metadata drift.

`save_analysis` and `load_analysis` additionally reconstruct the complete typed
`PaperAnalysis`. Other artifacts load through `load_artifact` and preserve
their validated JSON payload exactly.

Generated artifacts belong under `outputs/`, which is ignored by Git.
