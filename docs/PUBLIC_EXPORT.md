# Public export verification

Original source, tests and copied resource inputs were hash-checked unchanged. No installed executable, desktop shortcut, original profile or user setting was modified. Only this export and external validation artifacts were written.

429 tests discovered: 102 passed, 327 explicitly skipped because their modules transitively depend on excluded private screenshots. The conservative module-level exclusion also skips some self-contained tests within those modules; it is not a claim that they require private data individually. All test source is retained. No assertion was removed. One test expectation was updated for the public-only application-data directory. Three public resource tests were added.

Validation ran with the existing installed Python dependencies, importing application and Tetrarch source from this export. Stockfish was supplied from a temporary external validation copy; it is not included and no validation path is saved here. ONNX loaded from this export, inference ran, and session reuse was checked. The classic and FFA windows opened and rendered through Qt offscreen. This is not a manual native Windows screen-capture session or a clean-machine installation test.

Private screenshots, screenshot annotations, old logs/docs, profiles, backups, release files, research checkouts, browser data, personal decorations, compiler tools and the original virtual environment are absent. Decorations are neutral licensed chess graphics. Runtime, capture, ONNX, classic UCI, FFA, color selection, PGN and game-review code remain present. Public settings use ChessAnalyzerPublic to avoid touching the installed app.

Automated privacy checks searched all export files, including binary bytes, for personal home paths, the original account name, recognizable private-key headers and common token prefixes. Manual inspection covered selected source, asset provenance and license notices. This is not a guarantee against every possible secret format. Mandatory third-party copyright attributions remain.

External requirements: install pinned Python dependencies and obtain Stockfish separately. Tetrarch DLL/model and ONNX model are bundled with notices. A fresh dependency installation, a second Windows machine, full public image-fixture validation and native live capture remain unverified. No Git repository was initialized, no installer was rebuilt, and nothing was published or transmitted to GitHub.
