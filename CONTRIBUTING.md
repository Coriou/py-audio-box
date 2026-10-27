# Contributing

Keep changes small and explain the affected behavior and how you checked it.
Preserve existing copyright and third-party notices. Contributions to project
source are provided under AGPL-3.0-only; contribute only material you have the
right to provide under those terms. No copyright assignment is requested.

Do not commit credentials, personal recordings, model weights, caches or unrelated
qualification evidence. Use synthetic fixtures and explicit file lists.

The Python project targets Python 3.11. Use the repository's locked test environment;
run `python -m pytest tests -m "not integration and not redis_integration and not manual_b_probe" --ignore=tests/local --ignore=tests/remote --ignore=tests/release/test_dispatcher.py --ignore=tests/release/test_packaged_lifecycle.py --ignore=tests/release/test_admission.py -p no:cacheprovider -q`
for the ordinary unit selection (`Dockerfile.tests` and `.github/workflows/unit.yml`).
Run focused tests first, for example
`python -m unittest -v tests.test_source_distribution tests.test_docs_onboarding`.
Report missing dependencies and unrun checks explicitly. Integration, Docker,
model execution and listening qualification are separate checks.

For Studio, use its pinned Node/npm versions in `studio/package.json`;
`cd studio && npm test` checks tests, `npm run build` checks types/build, and
`npm run dev` starts the local development server.

Contract changes must keep `contracts/v1/`, synthetic fixtures and client behavior
consistent. `python -m contracts.generate` regenerates deterministic fixtures;
inspect its diff rather than overwriting unrelated work. Keep generated contract
documentation, CLI examples and onboarding drift checks in sync with changed
operations. Run `tests/contracts` and `tests/test_docs_onboarding.py` in the locked
environment, and preserve archive documentation/member closure when adding links.
See [source distribution guidance](docs/SOURCE-LICENSE.md) for release source duties.
