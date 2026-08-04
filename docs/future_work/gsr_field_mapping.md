# GSR Field Mapping Reference — Future Work

> **Status: REFERENCE ONLY — Not active code.**
> This document captures the expected field-by-field mapping between a real Game State Reconstruction (GSR) pipeline output and this project's internal data models. It is intended as a reference for when real GSR tracking data becomes available from the separate, in-progress GSR capstone project.

---

## Field Mapping: Typical Real GSR Output → Internal Data Model

### Per-Frame Structure

| # | Real GSR Field (Typical) | Internal Model Field | Match? | Notes |
|:--|:---|:---|:---|:---|
| 1 | `frame_id` / `frame_num` / `frame` | `FrameData.frame_index` (int) | ⚠️ Name varies | Parser must accept all common names. |
| 2 | `timestamp` (may be absent; derived from frame_id / fps) | `FrameData.timestamp_sec` (float) | ⚠️ Often missing | Real GSR rarely provides timestamp; must be computed as `frame_index / fps`. |
| 3 | *(not present in real GSR)* | `FrameData.timestamp_str` (str, `"MM:SS"`) | ❌ Never in real data | Always derived; no change needed. |

### Player Fields

| # | Real GSR Field (Typical) | Internal Model Field | Match? | Notes |
|:--|:---|:---|:---|:---|
| 4 | `track_id` / `tracker_id` / `id` | `TrackedPlayer.track_id` (int) | ⚠️ Name varies | Real data may use `id` or `tracker_id`. |
| 5 | `team` / `team_label` / `club` / `team_id` | `TrackedPlayer.team_id` (str) | ⚠️ Name + values vary | Real GSR often uses `"home"` / `"away"` or `0` / `1` instead of `"Team A"` / `"Team B"`. May also include `"referee"` or `"goalkeeper"`. |
| 6 | `jersey_number` / `jersey` / `shirt_number` / `number` | `TrackedPlayer.jersey_number` (Optional[int]) | ⚠️ Name varies; often noisy | Real jersey recognition has ~60-70% accuracy; expect `null`, `-1`, or wrong numbers. |
| 7 | `pitch_x` / `x_pitch` / `x_projected` | `TrackedPlayer.x` (float, meters) | ⚠️ Name + coordinate system varies | Real GSR may use pixel coordinates (bbox center) instead of pitch-projected meters. Parser MUST check which coordinate space is used. |
| 8 | `pitch_y` / `y_pitch` / `y_projected` | `TrackedPlayer.y` (float, meters) | ⚠️ Same as above | If pixel coords, needs homography transform. If pitch coords, may use different origin (center vs corner). |
| 9 | `speed` / `velocity` | `TrackedPlayer.speed` (float, m/s) | ❌ Often missing | Must be derived from positional delta between frames. |
| 10 | `bbox` / `bounding_box` / `[x1,y1,x2,y2]` | *(not in model)* | ❌ Extra field | Present in real GSR, ignored by event extraction (uses pitch coords only). |
| 11 | `confidence` / `detection_score` | *(not in model)* | ❌ Extra field | Could filter low-quality detections. |

### Ball Fields

| # | Real GSR Field (Typical) | Internal Model Field | Match? | Notes |
|:--|:---|:---|:---|:---|
| 12 | Ball may be inline with players (role=`"ball"`) or separate key | `FrameData.ball` (`TrackedBall`) | ⚠️ Structure varies | Parser must handle both formats. |
| 13 | `ball.x` / `ball.pitch_x` | `TrackedBall.x` (float) | ⚠️ Same coord issues as players | |
| 14 | `ball.z` / `ball.height` | `TrackedBall.z` (float) | ❌ Usually missing | Rarely estimated in 2D pipelines. |
| 15 | `ball.speed` | `TrackedBall.speed` (float) | ❌ Usually missing | Must be derived. |

### File Format Variations

| Format | Structure | Likelihood |
|:---|:---|:---|
| **Single JSON** (like current synthetic) | `{"frames": [...]}` | Common for custom pipelines |
| **JSONL / NDJSON** | One JSON object per line, one per frame | Common for streaming output |
| **CSV** | One row per player-per-frame | Very common (SoccerNet format) |

---

## Required Parser Changes (When Real Data Arrives)

1. Auto-detect format (JSON vs JSONL vs CSV) and field name variants
2. Normalize team labels (`"home"`/`"away"` → `"Team A"`/`"Team B"`)
3. Handle missing `timestamp_sec` (derive from `frame_index / fps`)
4. Derive `speed` from positional deltas if not present
5. Handle missing/noisy jersey numbers gracefully (keep as `None`)
6. Ignore extra fields (`bbox`, `confidence`, `homography`) without breaking

The downstream pipeline (preprocessor, heuristics, chunker, RAG) should require **zero changes** — the parser produces identical `FrameData` / `TrackedPlayer` / `TrackedBall` instances regardless of input format.
