# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

BlenderMCP bridges Claude and Blender via the Model Context Protocol (MCP). It has two components:

1. **Blender Addon** (`addon.py`) — A TCP socket server running inside Blender that receives JSON commands and executes them in Blender's main thread
2. **MCP Server** (`src/blender_mcp/server.py`) — A FastMCP server that exposes tools to Claude and forwards commands to the Blender addon over TCP

Communication flow: Claude → MCP Server → TCP socket (length-prefixed JSON) → Blender Addon → `bpy` API

## Development Commands

```bash
# Install dependencies
uv sync

# Run the MCP server locally
uv run blender-mcp

# Install in editable mode for development
uv pip install -e .
```

There is no test suite, linter configuration, or CI pipeline in this project.

## Architecture

### Two-Process Design

The MCP server and Blender addon run in separate processes, communicating via JSON over TCP sockets (default: `localhost:9876`).

- **MCP Server** uses `mcp[cli]` (FastMCP) to register tools. A global `BlenderConnection` instance (`get_blender_connection()`) manages the persistent TCP socket with automatic reconnection retry. All tools share this connection. Liveness is checked via `ping` command.
- **Blender Addon** uses `bpy.app.timers.register()` to schedule all Blender API calls on the main thread (required for thread safety). A command dispatcher routes incoming JSON messages to handler functions. Uses structured logging via `logging.getLogger("BlenderMCP")`.

### Command Protocol

Messages use length-prefixed framing: `[4-byte big-endian length][JSON payload]`. Helper functions `send_message(sock, data_dict)` and `recv_message(sock, timeout)` in both `addon.py` and `server.py` handle this.

Request: `{"type": "command_name", "params": {...}}`
Response: `{"status": "success"|"error", "result": {...}}` or `{"status": "error", "message": "..."}`

Built-in commands: `ping` (liveness check), `reload_addon` (hot-reload from disk), `install_addon` (install/update from path). The addon sends `{"type": "heartbeat"}` every 10 seconds to detect dead connections.

Socket timeout: 180 seconds for commands, 30 seconds for client recv (heartbeat-driven).

### Integration Architecture

The addon supports optional integrations (PolyHaven, Sketchfab, Hyper3D Rodin, Hunyuan3D). Each is toggled via Blender scene properties and conditionally registers its command handlers. The MCP server mirrors this with corresponding tools that check integration status before use.

### Telemetry

`telemetry.py` + `telemetry_decorator.py` implement a non-blocking background telemetry system using Supabase. The `@telemetry_tool()` decorator wraps MCP tools. Consent level (from addon settings) controls what data is collected.

### Key Environment Variables

- `BLENDER_HOST` / `BLENDER_PORT` — Override socket connection target (defaults: `localhost`, `9876`)
- `BLENDER_AUTH_TOKEN` — Optional authentication token for socket connections. When set, the MCP server sends an auth handshake on connect. The addon must have the matching token configured in its panel.
- `DISABLE_TELEMETRY=true` — Completely disable telemetry

### Entry Points

- `pyproject.toml` defines `blender-mcp = "blender_mcp.server:main"` as the CLI entry point
- `main.py` is an alternative entry that imports and calls the same `main()`
- `addon.py` registers with Blender via `bl_info` dict and `register()`/`unregister()` functions

### MCP Tool Inventory

**Core tools** (always available):
- `get_scene_info` — Scene overview with render settings, camera, world, collections, lights, paginated objects
- `get_object_info` — Detailed object info with modifiers, constraints, parent/children, type-specific data
- `get_viewport_screenshot` — Capture 3D viewport as image
- `execute_blender_code` — Execute Python in Blender with timeout and stderr capture

**Read-only query tools:**
- `get_lights` — List all lights with properties
- `get_render_settings` — Current render engine, resolution, samples, FPS
- `get_collections` — Recursive collection hierarchy
- `get_mesh_info` — Vertex/edge/polygon counts, UV layers, vertex groups, shape keys
- `get_animation_info` — Frame range, keyframed objects, animated properties
- `get_keyframes` — Detailed fcurve/keyframe data for an object
- `get_modifiers` — Modifier stack with serialized parameters
- `get_material_info` — Material node tree, links, Principled BSDF parameters
- `get_constraints` — Constraint stack with serialized parameters

**Creation & mutation tools:**
- `create_light` / `set_light_property` — Create and configure lights
- `create_camera` / `set_camera_property` / `set_active_camera` — Create and configure cameras
- `create_collection` / `delete_collection` / `move_to_collection` / `set_collection_visibility` — Manage collections
- `create_material` / `assign_material` / `set_material_property` — Create and configure PBR materials
- `set_frame_range` / `insert_keyframe` / `delete_keyframe` — Animation timeline and keyframes

**Complex tools:**
- `add_modifier` / `remove_modifier` / `set_modifier_params` / `apply_modifier` — Full modifier workflow
- `add_constraint` / `remove_constraint` / `set_constraint_params` — Full constraint workflow
- `uv_unwrap` / `set_vertex_group` — Mesh operations
- `set_render_settings` / `render_image` — Configure and trigger renders
- `frame_selected` — Position camera to frame objects

**Utility tools:**
- `reload_addon` / `install_addon` — Hot-reload or install addon from disk

**Integration tools** (conditionally available):
- PolyHaven: `get_polyhaven_status`, `get_polyhaven_categories`, `search_polyhaven_assets`, `download_polyhaven_asset`, `set_texture`
- Sketchfab: `get_sketchfab_status`, `search_sketchfab_models`, `get_sketchfab_model_preview`, `download_sketchfab_model`
- Hyper3D Rodin: `get_hyper3d_status`, `generate_hyper3d_model_via_text`, `generate_hyper3d_model_via_images`, `poll_rodin_job_status`, `import_generated_asset`
- Hunyuan3D: `get_hunyuan3d_status`, `generate_hunyuan3d_model`, `poll_hunyuan_job_status`, `import_generated_asset_hunyuan`
