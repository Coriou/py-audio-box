# Lot 0B editor fixture

Requires Node **25.2.1**, npm **11.6.2** (UI-only pins; no service toolchain decision).

```sh
npm ci --ignore-scripts
npm run dev
```

Open http://127.0.0.1:5178. The server binds loopback. Synthetic shared media only.
`npm run build` typechecks and emits the fixture in `dist/`; `npm run dev` serves source.

Adapter checks invoke the existing Python contract oracle, not models. From repository root:

```sh
python3 -m venv studio/.test-venv
studio/.test-venv/bin/pip install -r studio/evaluation/requirements-oracle.txt
CONTRACT_PYTHON="$PWD/studio/.test-venv/bin/python" npm test --prefix studio
npm run build --prefix studio
```

The optional Waveform Playlist probe is `/evaluation/playlist.html` under the dev server; click Load fixture, then `await window.probeZero()` in browser diagnostics to reproduce gain-zero conversion. It is excluded from the production entry and retained as an evaluation-only dev dependency.

See `docs/agent-interface/phase-0/EDITOR-0B.md` for decision, evidence, interop and limitations. Browser storage is a local fixture transport, not the Phase 0C service or a database. Reset fixture only resets this fixture's local browser history.
