# TODO

## Reliability & Connection

- [x] **Fix liveness check using wrong command** — Replaced `get_polyhaven_status` with dedicated `ping` command.
- [x] **Add client socket timeout** — Client recv uses 30s timeout (heartbeat-driven).
- [x] **Cap the JSON receive buffer** — Replaced with length-prefixed framing (50MB max message size).
- [x] **Add message framing** — Both sides use 4-byte big-endian length prefix via `send_message`/`recv_message`.
- [x] **Add send retry logic** — `send_command` retries once on connection errors before raising.
- [x] **Add connection heartbeat** — Addon sends heartbeat every 10s via `bpy.app.timers`.
- [x] **Add request timeouts to all HTTP calls** — `timeout=30` for metadata, `timeout=60` for downloads.

## Code Execution Safety

- [x] **Add execution timeout to `execute_code`** — Uses `threading.Timer` watchdog with configurable timeout (default 30s, max 120s).
- [x] **Capture stderr** — Both stdout and stderr captured via `redirect_stdout` + `redirect_stderr`.
- [x] **Truncate large output** — Output capped at 64KB with `[truncated]` marker.

## New Blender Tools

### Animation & Timeline
- [x] Insert/delete keyframes on object properties (location, rotation, scale, custom)
- [x] Get/set frame range, current frame, FPS
- [ ] Playback control (play, pause, jump to frame)
- [x] Read keyframe data for an object

### Modifiers
- [x] Add/remove/reorder modifiers on objects
- [x] Get modifier stack info
- [x] Configure modifier parameters (subdivision level, boolean operation, array count, etc.)
- [x] Apply modifier (bake to mesh)

### Materials & Shader Nodes
- [x] Create materials and assign to objects/faces
- [x] Get/set principled BSDF parameters (color, roughness, metallic, IOR, transmission, etc.)
- [ ] Create and connect shader nodes programmatically
- [x] List material node tree structure

### Lighting
- [x] Create lights (point, sun, spot, area)
- [x] Get/set light properties (power, color, size, shadow settings)
- [x] List all lights in scene with properties

### Camera
- [x] Create cameras
- [x] Get/set camera properties (focal length, sensor size, DOF, clip distances)
- [x] Set active camera
- [x] Frame selected objects in camera view

### Render Settings
- [x] Get/set render engine (Eevee, Cycles, Workbench)
- [x] Get/set resolution, samples, output format
- [x] Configure denoising
- [x] Trigger render and return result image

### Collections & Organization
- [x] Create/delete/rename collections
- [x] Link/unlink objects to collections
- [x] Get collection hierarchy
- [x] Toggle collection visibility

### Mesh Operations
- [x] Get vertex group info
- [x] Get/set shape keys
- [x] UV unwrap with preset methods (smart project, cube projection, etc.)
- [x] Get UV layer info

### Constraints
- [x] Add/remove constraints (copy location, track to, child of, etc.)
- [x] Configure constraint parameters
- [x] Get constraint stack for an object

## Scene Inspection Improvements

- [x] **Expand `get_scene_info`** — Now returns render settings, active camera, world/HDRI info, collection hierarchy, light summary, with offset/limit pagination.
- [x] **Expand `get_object_info`** — Now returns modifiers, constraints, parent/children, custom properties, and type-specific data (light, camera, armature).

## Error Handling Cleanup

- [x] Replace bare `except: pass` blocks with specific exception types and logging.
- [x] Add structured error context — Errors prefixed with `[cmd_type]`, handler execution timed and logged at INFO.
- [x] Consistent temp file cleanup — Standardized on explicit `os.unlink` in `finally` blocks. Replaced `tempfile._cleanup()`.

## UX

- [x] **Connection status indicator** — Panel shows client connected/disconnected and last command status.
- [x] **Progress feedback for long operations** — Panel shows current operation status with SORTTIME icon during downloads and generation.
- [x] **Structured logging** — All `print()` replaced with `logging.getLogger("BlenderMCP")`.
- [x] **Persistent log file** — Optional `RotatingFileHandler` (5MB, 3 backups) via addon preferences.
- [x] **Asset search pagination** — PolyHaven supports offset/limit params. Sketchfab reports next page availability.

## Security

- [x] **Add optional socket authentication** — Auth token in addon panel, `BLENDER_AUTH_TOKEN` env var on server side. Auth handshake required as first message when token is set.
- [x] **Connection access logging** — All command execution logged at INFO with timestamps and duration.
