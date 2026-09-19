# Final NuRec time provenance hardening

The scanner is bounded and reproducible. It reports requested, found, missing, successfully read, failed, and errored metadata files. A read error is never inserted into the verified candidate set and therefore cannot produce `FOUND`.

Identity semantics are explicit:

- `SOURCE_IDENTITY` is verified only when an explicit source field matches the expected PhysicalAI clip.
- `GENERIC_IDENTITY` (`clip_id`) remains `CANDIDATE_ONLY`.
- target and unknown identity fields do not establish source lineage.

External Hugging Face dataset presence is recorded with `evidence_source=EXTERNAL_INSPECTION` and `verified_by_this_script=false`. Public repository evidence is labeled `MANUAL_REVIEW_RECORDED_IN_REPOSITORY`; the report does not claim exhaustive automated public search.

The scientific result is unchanged: source lineage, if found, is separate from time lineage. `NCORE_TO_NUREC_MAPPING=UNRESOLVED`, `TIME_ALIGNMENT_VERIFIED=false`, and `OFFSET_US=null`. The forensic status is `EXHAUSTED_WITH_PUBLIC_EVIDENCE`, with remaining blocker `NCORE_TO_NUREC_TIME_TRANSFORM_NOT_PUBLICLY_PROVEN`.
