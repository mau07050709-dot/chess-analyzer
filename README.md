# Local Chess Analyzer

A Windows desktop application for local chess study, built with Python, PySide6, ONNX Runtime and asynchronous chess engines.

## Features

- Classic chess: editable FEN, Stockfish UCI analysis, five ranked arrows, white-perspective scores and a vertical evaluation bar.
- PGN import, main-line navigation, full-game review and configurable move-quality heuristics.
- Screen-region capture, local ONNX piece recognition, manual review and continuous legal-position tracking with orientation/turn recovery.
- Four-player Free-for-All: cross-shaped board, Tetrarch analysis, gray-army handling and persistent color selection independent of the real turn. Choose **Mi color / Evaluar a** to analyze Red, Blue, Yellow or Green and place it below.
- Dark theme, nonblocking workers, cancellation and stale-result rejection.
- Classic analysis supports an overlay, configurable board visuals and exploring variations while tracking a position. Live capture includes updated board-style and last-move recognition.

## Windows setup

Use 64-bit Windows 10/11 and Python 3.12. The bundled Tetrarch DLL is Windows x64 CPU code. Other systems have not been validated.

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pip install --no-deps -e .
```

Download Stockfish from https://stockfishchess.org/download/ and extract a version compatible with your CPU. It is intentionally not bundled. Select its executable with **Elegir motor** or set:

```powershell
$env:STOCKFISH_EXECUTABLE = 'C:/Tools/Stockfish/stockfish.exe'
.\.venv\Scripts\python.exe -m ajedrez.ui
```

CLI: `.\.venv\Scripts\python.exe -m ajedrez --help`. The ONNX model and licensed Tetrarch runtime/network are included. No cloud recognition, API key or GPU is needed. `.env.example` documents environment variables; dotenv is not automatically loaded. Public-copy app data is isolated in `%LOCALAPPDATA%/ChessAnalyzerPublic`.

## Tests

```powershell
.\.venv\Scripts\python.exe tools/run_public_tests.py
```

Private screenshots and their metadata are deliberately excluded. The runner reports test modules depending on them as skipped, without claiming those regressions passed. Test source is retained. Run standard `python -m unittest discover -s tests` only after supplying your own licensed fixtures; missing images otherwise cause failures. No accounts, browser automation or real games are used by the public runner. Physical mouse input is isolated in injected-frame tests.

## Architecture

`src/ajedrez`: engine/UCI sessions, recognition, capture, legal state tracking, PGN/game review, FFA adapters and Qt UI. `models`: ONNX model/license. `engines/tetrarch`: MIT runtime, Python/C source and FFA network. `tests`: regression source. `tools`: source entry and public verification runner.

## Limits and responsible use

Recognition depends on board themes, scaling, image quality and visible pieces. An isolated image cannot prove turn or historical castling rights. Tetrarch provides one actual FFA recommendation using paranoid utility; it is not Stockfish and does not model every Chess.com rule or random retired-king behavior exactly. Full-game classifications are documented heuristics, not an exact proprietary Game Review replica. Use for study, training and local analysis, respecting platform rules against engine assistance in live competitive games.

This export replaces personal decoration with licensed neutral chess artwork. No private captures, history, credentials, browser profiles, installation files or release executables are included. Tests use installed dependencies; a fresh-machine dependency installation is still required. Original private project and desktop application are outside this export.

## License

Application: GPL-3.0-or-later. See LICENSE, LICENSE_NOTICE.md and THIRD_PARTY_NOTICES.md. Preserve all upstream notices. Stockfish binary redistribution and installer creation require their own source/license compliance; this repository is a source distribution, not a signed Windows installer.
