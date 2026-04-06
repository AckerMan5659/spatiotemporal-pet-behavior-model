# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Layout

- This repo is a split application, not a JS monorepo or Python workspace. Run commands from the relevant subdirectory.
- `frontend/` is a Vite + React 18 + TypeScript SPA.
- `backend/` is a Flask + Waitress video inference server that handles camera ingestion, inference, MJPEG streaming, recording, and the API consumed by the frontend.
- `go2rtc/` contains the local RTSP relay executable/config used by the backend's RTSP defaults.

## Common Commands

### Frontend (`frontend/`)

- Install dependencies: `npm install`
- Start the dev server: `npm run dev`
- Build production assets: `npm run build`
- Vite dev server runs on port `3000`; production build output goes to `frontend/build`.
- No frontend lint script is configured in `frontend/package.json`.
- No frontend test runner or single-test command is configured.

### Backend (`backend/`)

- Main backend entrypoint: `python yolo_infer_xiaomi_win_deploy.py`
- Common manual install path: `python -m pip install -r requirements.txt` from `backend/`
- Model export / OpenVINO conversion utility: `python OpenVINO_convert.py`
- The backend serves HTTP on port `9527`.
- Useful runtime flags on the backend entrypoint include `--source`, `--device`, `--hwaccel`, `--num_cams`, `--max_fps`, and per-camera `--rtsp_url_N` / `--cam_index_N` overrides.

### Dependency Notes

- There are two Python dependency manifests with different pins: root `requirements.txt` and `backend/requirements.txt`.
- The Windows bootstrap script `ppc_dasboard_env.bat` installs from the root `requirements.txt`; backend-only work often references `backend/requirements.txt`.
- Verify which manifest is intended before changing Python dependencies.

### Testing

- No repository-standard automated test suite, lint workflow, or single-test command is currently defined for either the frontend or backend.

## Architecture Overview

### Frontend

- `frontend/src/App.tsx` is the application shell. It does not use React Router; page changes are handled with local component state.
- The main screens are dashboard, pet data, monitoring/alerts, settings, fullscreen live monitoring, and a fullscreen analytics view.
- `frontend/src/services/api.ts` is the frontend/backend contract. It hardcodes the backend base URL as `http://127.0.0.1:9527` and wraps the API used across the UI.
- The frontend relies heavily on polling backend state rather than local data stores. `Dashboard.tsx`, `MonitoringAlerts.tsx`, `LiveMonitoring.tsx`, and `DataVisualization.tsx` all derive UI from repeated `/stats` fetches plus config/record APIs.
- Camera IDs are displayed in the UI as `CAM-01`, `CAM-02`, etc., but backend camera IDs are zero-based integers. `getVideoFeedUrl` in `frontend/src/services/api.ts` handles this translation.
- Frontend text and behavior label mappings live in `frontend/src/utils/translations.ts`. If backend behavior classes change, review the frontend's hardcoded behavior lists and translations together.

### Backend

- `backend/yolo_infer_xiaomi_win_deploy.py` is the core orchestrator. It owns hardware profiling, strategy selection, multiprocessing lifecycle, Flask routes, MJPEG streaming, SQLite persistence, and recording.
- Startup selects a hardware tier via `HardwareProfiler` and `StrategyFactory`, then chooses either:
  - parallel mode: one worker process per active camera
  - sequential mode: one centralized inference engine for lower-end hardware
- Active camera selection is dynamic. The backend stores the desired camera list in shared multiprocessing state and exposes it via `/api/active_cams`; in parallel mode the parent process starts/stops camera workers accordingly.
- Per-camera inference flow spans several files:
  - detection/tracking happens in `backend/yolo_infer_xiaomi_win_deploy.py`
  - `backend/core/bowl_manager.py` classifies detected bowls as food or water from image color
  - `backend/core/rule_engine.py` derives motion- and overlap-based candidate states
  - `backend/modules/action_recognizer.py` loads the behavior model selected by `backend/config.yaml` and runs either ONNX Runtime or PyTorch inference
  - `backend/core/fusion_agent.py` fuses rule outputs with model probabilities and smooths state transitions with an FSM
- `backend/config.yaml` is the source of truth for recognition mode (`normal` vs `abnormal`), class names, model path, recognizer settings, and detector settings. Check it before changing behavior labels, class counts, or model-loading logic.
- The backend publishes a shared state structure per camera with `stats`, `logs`, and `active_states`; the frontend screens all build from this shape.

### Recording and Data Flow

- Triggered recordings are written to date-based `records/...` folders with both video clips and thumbnails, and metadata is indexed in SQLite `records.db`.
- The Flask API surface defined in `backend/yolo_infer_xiaomi_win_deploy.py` includes:
  - `/stats`
  - `/api/active_cams`
  - `/api/config/<cam_id>`
  - `/api/records`
  - `/api/records/<record_id>`
  - `/records/<path>`
  - `/video_feed/<cam_id>`

### RTSP / Relay Expectations

- Backend RTSP defaults point at a local relay on `rtsp://127.0.0.1:8554/...`.
- The repo includes `go2rtc/` for that relay, and `go2rtc/go2rtc.yaml` is environment-specific machine config.
- Treat `go2rtc/go2rtc.yaml` as sensitive local configuration; do not echo credentials/tokens back into docs or commits.

## Working Notes For This Repo

- Usually ignore or avoid editing generated/runtime directories and artifacts unless the task is specifically about them: `.venv/`, `frontend/node_modules/`, model binaries under `backend/model/`, `records/`, and `records.db`.
- `ppc_dasboard_env.bat` is not a normal project run command. It is a Windows bootstrap script that requests admin elevation, installs Python/Node, changes npm/pip mirrors, and performs machine setup. Do not run it unless the user explicitly asks for environment bootstrap work.
- When changing anything that touches the frontend/backend contract, check both `frontend/src/services/api.ts` and the Flask routes in `backend/yolo_infer_xiaomi_win_deploy.py`.