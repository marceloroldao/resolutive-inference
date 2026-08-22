# Publication checklist — 0.2.0rc1

## Release identity

- [x] README identifies `0.2.0rc1` as an experimental publication candidate.
- [x] `pyproject.toml` package version is `0.2.0rc1`.
- [x] `CITATION.cff` version is `0.2.0-rc1`.
- [x] API documentation identifies the same release-candidate line.
- [x] Release notes exist and document limitations.

## Reproducibility and validation

- [x] Python CI covers 3.10, 3.11 and 3.12.
- [x] Ruff and pytest pass in CI.
- [x] Server latency benchmark is repository-native.
- [x] Single-process load benchmark records throughput, latency and errors.
- [x] Gaussian HMM and Student-t HMM controls use the same generated observations and known parameters.
- [x] Benchmark claims are explicitly limited to tested synthetic conditions.

## API and engineering scope

- [x] PC/server is explicitly the primary release target.
- [x] HTTP/WebSocket contract is documented.
- [x] Authentication and persistence configuration are documented.
- [x] API/session limits are documented.
- [x] `v1.0` compatibility is explicitly not frozen.
- [x] ESP32/MCU claims are withheld until target-hardware measurements exist.

## Scientific claims

- [x] No claim of universal superiority over HMMs or neural networks.
- [x] Synthetic known-parameter comparison is labeled as such.
- [x] Real-world external-dataset validation is identified as future work.
- [x] Runtime/load measurements are not represented as production SLAs.

## Licensing and citation

- [x] `LICENSE` contains RRNCL 1.0 terms.
- [x] README states that the project is source-available, not OSI-approved open source.
- [x] `CITATION.cff` identifies the author, repository and custom license reference.
- [x] Citation instructions request the exact release or commit.

## Before creating the public release/tag

- [ ] Confirm the final release commit has green CI after documentation merge.
- [ ] Create the `v0.2.0rc1` tag/release from the audited `main` commit.
- [ ] Use `RELEASE_NOTES.md` as the basis for the GitHub release description.
- [ ] If depositing on Zenodo, verify imported metadata before publishing the record and add the resulting DOI to the repository only after Zenodo assigns it.

## Gate decision

Technical and documentation gates are satisfied for an **experimental release candidate**, subject to final green CI on the release commit. This checklist does not approve `v1.0`; stable API and real-world validation remain separate gates.
