from mcp.server.fastmcp import FastMCP, Context, Image
import socket
import json
import struct
import asyncio
import logging
import tempfile
from dataclasses import dataclass
from contextlib import asynccontextmanager
from typing import AsyncIterator, Dict, Any, List
import os
from pathlib import Path
import base64
from urllib.parse import urlparse


logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("BlenderMCPServer")

MAX_MESSAGE_SIZE = 50 * 1024 * 1024  # 50MB

def send_message(sock, data_dict):
    """Send a length-prefixed JSON message over a socket."""
    payload = json.dumps(data_dict).encode('utf-8')
    header = struct.pack('>I', len(payload))
    sock.sendall(header + payload)

def recv_message(sock, timeout=180.0):
    """Receive a length-prefixed JSON message from a socket.
    Returns the parsed dict, or raises on error/timeout."""
    sock.settimeout(timeout)

    header = b''
    while len(header) < 4:
        chunk = sock.recv(4 - len(header))
        if not chunk:
            raise ConnectionError("Connection closed while reading message header")
        header += chunk

    msg_len = struct.unpack('>I', header)[0]
    if msg_len > MAX_MESSAGE_SIZE:
        raise ValueError(f"Message size {msg_len} exceeds maximum {MAX_MESSAGE_SIZE}")

    data = b''
    while len(data) < msg_len:
        chunk = sock.recv(min(msg_len - len(data), 65536))
        if not chunk:
            raise ConnectionError("Connection closed while reading message body")
        data += chunk

    return json.loads(data.decode('utf-8'))

DEFAULT_HOST = "localhost"
DEFAULT_PORT = 9876

@dataclass
class BlenderConnection:
    host: str
    port: int
    sock: socket.socket = None  # Changed from 'socket' to 'sock' to avoid naming conflict
    
    def connect(self) -> bool:
        """Connect to the Blender addon socket server"""
        if self.sock:
            return True

        try:
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.sock.connect((self.host, self.port))
            logger.info(f"Connected to Blender at {self.host}:{self.port}")

            auth_token = os.getenv("BLENDER_AUTH_TOKEN", "")
            if auth_token:
                send_message(self.sock, {"type": "auth", "token": auth_token})
                response = recv_message(self.sock, timeout=10.0)
                if response.get("status") != "success":
                    logger.error(f"Authentication failed: {response.get('message', 'Unknown error')}")
                    self.sock.close()
                    self.sock = None
                    return False
                logger.info("Authenticated with Blender addon")

            return True
        except Exception as e:
            logger.error(f"Failed to connect to Blender: {str(e)}")
            self.sock = None
            return False
    
    def disconnect(self):
        """Disconnect from the Blender addon"""
        if self.sock:
            try:
                self.sock.close()
            except Exception as e:
                logger.error(f"Error disconnecting from Blender: {str(e)}")
            finally:
                self.sock = None

    def send_command(self, command_type: str, params: Dict[str, Any] = None) -> Dict[str, Any]:
        """Send a command to Blender and return the response"""
        if not self.sock and not self.connect():
            raise ConnectionError("Not connected to Blender")

        command = {"type": command_type, "params": params or {}}
        last_error = None

        for attempt in range(2):
            try:
                if attempt > 0:
                    logger.info(f"Retrying command: {command_type} (attempt {attempt + 1})")
                    self.disconnect()
                    if not self.connect():
                        raise ConnectionError("Reconnection failed")

                logger.info(f"Sending command: {command_type}")
                send_message(self.sock, command)
                response = recv_message(self.sock, timeout=180.0)
                logger.info(f"Response received, status: {response.get('status', 'unknown')}")

                if response.get("status") == "error":
                    raise Exception(response.get("message", "Unknown error from Blender"))
                return response.get("result", {})

            except (ConnectionError, BrokenPipeError, ConnectionResetError) as e:
                last_error = e
                logger.warning(f"Connection error on attempt {attempt + 1}: {e}")
                self.sock = None
                if attempt == 0:
                    continue
            except socket.timeout:
                self.sock = None
                raise Exception("Timeout waiting for Blender response - try simplifying your request")
            except json.JSONDecodeError as e:
                self.sock = None
                raise Exception(f"Invalid response from Blender: {e}")
            except Exception:
                self.sock = None
                raise

        raise Exception(f"Connection to Blender lost after retry: {last_error}")

@asynccontextmanager
async def server_lifespan(server: FastMCP) -> AsyncIterator[Dict[str, Any]]:
    """Manage server startup and shutdown lifecycle"""
    try:
        logger.info("BlenderMCP server starting up")

        try:
            blender = get_blender_connection()
            logger.info("Successfully connected to Blender on startup")
        except Exception as e:
            logger.warning(f"Could not connect to Blender on startup: {str(e)}")
            logger.warning("Make sure the Blender addon is running before using Blender resources or tools")

        yield {}
    finally:
        global _blender_connection
        if _blender_connection:
            logger.info("Disconnecting from Blender on shutdown")
            _blender_connection.disconnect()
            _blender_connection = None
        logger.info("BlenderMCP server shut down")

mcp = FastMCP(
    "BlenderMCP",
    lifespan=server_lifespan
)

_blender_connection = None

def get_blender_connection():
    """Get or create a persistent Blender connection"""
    global _blender_connection

    if _blender_connection is not None:
        try:
            _blender_connection.send_command("ping")
            return _blender_connection
        except Exception as e:
            logger.warning(f"Existing connection is no longer valid: {e}")
            try:
                _blender_connection.disconnect()
            except Exception:
                pass
            _blender_connection = None

    host = os.getenv("BLENDER_HOST", DEFAULT_HOST)
    port = int(os.getenv("BLENDER_PORT", DEFAULT_PORT))
    _blender_connection = BlenderConnection(host=host, port=port)
    if not _blender_connection.connect():
        logger.error("Failed to connect to Blender")
        _blender_connection = None
        raise Exception("Could not connect to Blender. Make sure the Blender addon is running.")
    logger.info("Created new persistent connection to Blender")

    return _blender_connection


@mcp.tool()
def get_scene_info(ctx: Context, offset: int = 0, limit: int = 50) -> str:
    """Get detailed information about the current Blender scene.

    Parameters:
    - offset: Starting index for object list pagination (default: 0)
    - limit: Maximum number of objects to return (default: 50)
    """
    try:
        blender = get_blender_connection()
        result = blender.send_command("get_scene_info", {"offset": offset, "limit": limit})
        return json.dumps(result, indent=2)
    except Exception as e:
        logger.error(f"Error getting scene info from Blender: {str(e)}")
        return f"Error getting scene info: {str(e)}"

@mcp.tool()
def get_object_info(ctx: Context, object_name: str) -> str:
    """
    Get detailed information about a specific object in the Blender scene.
    
    Parameters:
    - object_name: The name of the object to get information about
    """
    try:
        blender = get_blender_connection()
        result = blender.send_command("get_object_info", {"name": object_name})
        return json.dumps(result, indent=2)
    except Exception as e:
        logger.error(f"Error getting object info from Blender: {str(e)}")
        return f"Error getting object info: {str(e)}"

@mcp.tool()
def get_viewport_screenshot(ctx: Context, max_size: int = 800) -> Image:
    """
    Capture a screenshot of the current Blender 3D viewport.
    
    Parameters:
    - max_size: Maximum size in pixels for the largest dimension (default: 800)
    
    Returns the screenshot as an Image.
    """
    try:
        blender = get_blender_connection()

        temp_dir = tempfile.gettempdir()
        temp_path = os.path.join(temp_dir, f"blender_screenshot_{os.getpid()}.png")

        result = blender.send_command("get_viewport_screenshot", {
            "max_size": max_size,
            "filepath": temp_path,
            "format": "png"
        })
        
        if "error" in result:
            raise Exception(result["error"])
        
        if not os.path.exists(temp_path):
            raise Exception("Screenshot file was not created")

        with open(temp_path, 'rb') as f:
            image_bytes = f.read()

        os.remove(temp_path)
        
        return Image(data=image_bytes, format="png")
        
    except Exception as e:
        logger.error(f"Error capturing screenshot: {str(e)}")
        raise Exception(f"Screenshot failed: {str(e)}")


@mcp.tool()
def execute_blender_code(ctx: Context, code: str, timeout: int = 30) -> str:
    """
    Execute arbitrary Python code in Blender. Make sure to do it step-by-step by breaking it into smaller chunks.

    Parameters:
    - code: The Python code to execute
    - timeout: Maximum execution time in seconds (default: 30, max: 120)
    """
    try:
        blender = get_blender_connection()
        result = blender.send_command("execute_code", {"code": code, "timeout": timeout})
        output = "Code executed successfully."
        if result.get('result'):
            output += f"\nOutput: {result['result']}"
        if result.get('stderr'):
            output += f"\nStderr: {result['stderr']}"
        if result.get('warning'):
            output += f"\nWarning: {result['warning']}"
        return output
    except Exception as e:
        logger.error(f"Error executing code: {str(e)}")
        return f"Error executing code: {str(e)}"


@mcp.tool()
def get_lights(ctx: Context) -> str:
    """
    Get a list of all lights in the scene with their properties.

    Returns light names, types, energy, color, shadow settings, and transforms.
    """
    try:
        blender = get_blender_connection()
        result = blender.send_command("get_lights")
        return json.dumps(result, indent=2)
    except Exception as e:
        logger.error(f"Error getting lights: {str(e)}")
        return f"Error getting lights: {str(e)}"


@mcp.tool()
def get_render_settings(ctx: Context) -> str:
    """
    Get the current render settings including engine, resolution, samples, FPS, and output format.
    """
    try:
        blender = get_blender_connection()
        result = blender.send_command("get_render_settings")
        return json.dumps(result, indent=2)
    except Exception as e:
        logger.error(f"Error getting render settings: {str(e)}")
        return f"Error getting render settings: {str(e)}"


@mcp.tool()
def get_collections(ctx: Context) -> str:
    """
    Get the collection hierarchy of the current scene.

    Returns a recursive tree of collections with their objects, visibility, and children.
    """
    try:
        blender = get_blender_connection()
        result = blender.send_command("get_collections")
        return json.dumps(result, indent=2)
    except Exception as e:
        logger.error(f"Error getting collections: {str(e)}")
        return f"Error getting collections: {str(e)}"


@mcp.tool()
def get_mesh_info(ctx: Context, object_name: str) -> str:
    """
    Get detailed mesh data for a mesh object.

    Parameters:
    - object_name: Name of the mesh object

    Returns vertex/edge/polygon counts, UV layers, vertex groups, and shape keys.
    """
    try:
        blender = get_blender_connection()
        result = blender.send_command("get_mesh_info", {"name": object_name})
        return json.dumps(result, indent=2)
    except Exception as e:
        logger.error(f"Error getting mesh info: {str(e)}")
        return f"Error getting mesh info: {str(e)}"


@mcp.tool()
def get_animation_info(ctx: Context) -> str:
    """
    Get animation information for the current scene.

    Returns frame range, FPS, current frame, and a list of keyframed objects with their animated property paths.
    """
    try:
        blender = get_blender_connection()
        result = blender.send_command("get_animation_info")
        return json.dumps(result, indent=2)
    except Exception as e:
        logger.error(f"Error getting animation info: {str(e)}")
        return f"Error getting animation info: {str(e)}"


@mcp.tool()
def get_keyframes(ctx: Context, object_name: str, property_path: str = None) -> str:
    """
    Get keyframe data for a specific object.

    Parameters:
    - object_name: Name of the object to read keyframes from
    - property_path: Optional filter by property path (e.g. "location", "rotation_euler")

    Returns fcurve data with frame numbers, values, and interpolation types.
    """
    try:
        blender = get_blender_connection()
        params = {"name": object_name}
        if property_path:
            params["property_path"] = property_path
        result = blender.send_command("get_keyframes", params)
        return json.dumps(result, indent=2)
    except Exception as e:
        logger.error(f"Error getting keyframes: {str(e)}")
        return f"Error getting keyframes: {str(e)}"


@mcp.tool()
def get_modifiers(ctx: Context, object_name: str) -> str:
    """
    Get the modifier stack for an object.

    Parameters:
    - object_name: Name of the object

    Returns modifier names, types, visibility settings, and serialized parameters.
    """
    try:
        blender = get_blender_connection()
        result = blender.send_command("get_modifiers", {"name": object_name})
        return json.dumps(result, indent=2)
    except Exception as e:
        logger.error(f"Error getting modifiers: {str(e)}")
        return f"Error getting modifiers: {str(e)}"


@mcp.tool()
def get_material_info(ctx: Context, name: str) -> str:
    """
    Get detailed material information including node tree structure.

    Parameters:
    - name: Name of the material or object (if object, uses active material)

    Returns node names/types/locations, links between nodes, and Principled BSDF parameters if present.
    """
    try:
        blender = get_blender_connection()
        result = blender.send_command("get_material_info", {"name": name})
        return json.dumps(result, indent=2)
    except Exception as e:
        logger.error(f"Error getting material info: {str(e)}")
        return f"Error getting material info: {str(e)}"


@mcp.tool()
def get_constraints(ctx: Context, object_name: str) -> str:
    """
    Get the constraint stack for an object.

    Parameters:
    - object_name: Name of the object

    Returns constraint names, types, influence, mute state, targets, and serialized parameters.
    """
    try:
        blender = get_blender_connection()
        result = blender.send_command("get_constraints", {"name": object_name})
        return json.dumps(result, indent=2)
    except Exception as e:
        logger.error(f"Error getting constraints: {str(e)}")
        return f"Error getting constraints: {str(e)}"


@mcp.tool()
def create_light(
    ctx: Context,
    light_type: str = "POINT",
    name: str = "Light",
    location: list = None,
    rotation: list = None,
    energy: float = 1000.0,
    color: list = None,
    size: float = 0.25,
    spot_angle: float = 45.0,
    shadow: bool = True
) -> str:
    """
    Create a new light in the scene.

    Parameters:
    - light_type: Type of light (POINT, SUN, SPOT, AREA)
    - name: Name for the light object
    - location: [x, y, z] position (default: [0, 0, 3])
    - rotation: [x, y, z] rotation in radians
    - energy: Light power/energy
    - color: [r, g, b] color values 0-1
    - size: Shadow soft size
    - spot_angle: Spot cone angle in degrees (SPOT only)
    - shadow: Enable shadows
    """
    try:
        blender = get_blender_connection()
        params = {"light_type": light_type, "name": name, "energy": energy, "size": size, "spot_angle": spot_angle, "shadow": shadow}
        if location: params["location"] = location
        if rotation: params["rotation"] = rotation
        if color: params["color"] = color
        result = blender.send_command("create_light", params)
        return json.dumps(result, indent=2)
    except Exception as e:
        return f"Error creating light: {str(e)}"


@mcp.tool()
def set_light_property(
    ctx: Context,
    light_name: str,
    energy: float = None,
    color: list = None,
    size: float = None,
    shadow: bool = None,
    location: list = None,
    rotation: list = None
) -> str:
    """
    Set properties on an existing light.

    Parameters:
    - light_name: Name of the light object
    - energy: Light power/energy
    - color: [r, g, b] color values
    - size: Shadow soft size
    - shadow: Enable/disable shadows
    - location: [x, y, z] position
    - rotation: [x, y, z] rotation in radians
    """
    try:
        blender = get_blender_connection()
        params = {"light_name": light_name}
        if energy is not None: params["energy"] = energy
        if color is not None: params["color"] = color
        if size is not None: params["size"] = size
        if shadow is not None: params["shadow"] = shadow
        if location is not None: params["location"] = location
        if rotation is not None: params["rotation"] = rotation
        result = blender.send_command("set_light_property", params)
        return json.dumps(result, indent=2)
    except Exception as e:
        return f"Error setting light property: {str(e)}"


@mcp.tool()
def create_camera(
    ctx: Context,
    name: str = "Camera",
    location: list = None,
    rotation: list = None,
    focal_length: float = 50.0,
    sensor_width: float = 36.0,
    set_active: bool = True
) -> str:
    """
    Create a new camera in the scene.

    Parameters:
    - name: Name for the camera
    - location: [x, y, z] position
    - rotation: [x, y, z] rotation in radians
    - focal_length: Lens focal length in mm
    - sensor_width: Sensor width in mm
    - set_active: Set as the active scene camera
    """
    try:
        blender = get_blender_connection()
        params = {"name": name, "focal_length": focal_length, "sensor_width": sensor_width, "set_active": set_active}
        if location: params["location"] = location
        if rotation: params["rotation"] = rotation
        result = blender.send_command("create_camera", params)
        return json.dumps(result, indent=2)
    except Exception as e:
        return f"Error creating camera: {str(e)}"


@mcp.tool()
def set_camera_property(
    ctx: Context,
    camera_name: str,
    focal_length: float = None,
    clip_start: float = None,
    clip_end: float = None,
    dof_enabled: bool = None,
    dof_focus_distance: float = None,
    dof_aperture_fstop: float = None,
    location: list = None,
    rotation: list = None
) -> str:
    """
    Set properties on an existing camera.

    Parameters:
    - camera_name: Name of the camera object
    - focal_length: Lens focal length in mm
    - clip_start: Near clipping distance
    - clip_end: Far clipping distance
    - dof_enabled: Enable/disable depth of field
    - dof_focus_distance: DOF focus distance
    - dof_aperture_fstop: DOF aperture f-stop
    - location: [x, y, z] position
    - rotation: [x, y, z] rotation in radians
    """
    try:
        blender = get_blender_connection()
        params = {"camera_name": camera_name}
        for key, val in [("focal_length", focal_length), ("clip_start", clip_start), ("clip_end", clip_end),
                         ("dof_enabled", dof_enabled), ("dof_focus_distance", dof_focus_distance),
                         ("dof_aperture_fstop", dof_aperture_fstop), ("location", location), ("rotation", rotation)]:
            if val is not None:
                params[key] = val
        result = blender.send_command("set_camera_property", params)
        return json.dumps(result, indent=2)
    except Exception as e:
        return f"Error setting camera property: {str(e)}"


@mcp.tool()
def set_active_camera(ctx: Context, camera_name: str) -> str:
    """
    Set the active camera for the scene.

    Parameters:
    - camera_name: Name of the camera object to make active
    """
    try:
        blender = get_blender_connection()
        result = blender.send_command("set_active_camera", {"camera_name": camera_name})
        return json.dumps(result, indent=2)
    except Exception as e:
        return f"Error setting active camera: {str(e)}"


@mcp.tool()
def create_collection(ctx: Context, name: str, parent_name: str = None) -> str:
    """
    Create a new collection in the scene.

    Parameters:
    - name: Name for the new collection
    - parent_name: Optional parent collection name (uses scene collection if omitted)
    """
    try:
        blender = get_blender_connection()
        params = {"name": name}
        if parent_name: params["parent_name"] = parent_name
        result = blender.send_command("create_collection", params)
        return json.dumps(result, indent=2)
    except Exception as e:
        return f"Error creating collection: {str(e)}"


@mcp.tool()
def delete_collection(ctx: Context, collection_name: str, delete_objects: bool = False) -> str:
    """
    Delete a collection.

    Parameters:
    - collection_name: Name of the collection to delete
    - delete_objects: If True, also delete all objects in the collection
    """
    try:
        blender = get_blender_connection()
        result = blender.send_command("delete_collection", {"collection_name": collection_name, "delete_objects": delete_objects})
        return json.dumps(result, indent=2)
    except Exception as e:
        return f"Error deleting collection: {str(e)}"


@mcp.tool()
def move_to_collection(ctx: Context, object_name: str, collection_name: str, unlink_from_current: bool = True) -> str:
    """
    Move an object to a different collection.

    Parameters:
    - object_name: Name of the object to move
    - collection_name: Target collection name
    - unlink_from_current: Remove from current collections first (default: True)
    """
    try:
        blender = get_blender_connection()
        result = blender.send_command("move_to_collection", {
            "object_name": object_name, "collection_name": collection_name, "unlink_from_current": unlink_from_current
        })
        return json.dumps(result, indent=2)
    except Exception as e:
        return f"Error moving to collection: {str(e)}"


@mcp.tool()
def set_collection_visibility(ctx: Context, collection_name: str, hide_viewport: bool = None, hide_render: bool = None) -> str:
    """
    Set visibility for a collection.

    Parameters:
    - collection_name: Name of the collection
    - hide_viewport: Hide in viewport
    - hide_render: Hide in render
    """
    try:
        blender = get_blender_connection()
        params = {"collection_name": collection_name}
        if hide_viewport is not None: params["hide_viewport"] = hide_viewport
        if hide_render is not None: params["hide_render"] = hide_render
        result = blender.send_command("set_collection_visibility", params)
        return json.dumps(result, indent=2)
    except Exception as e:
        return f"Error setting collection visibility: {str(e)}"


@mcp.tool()
def create_material(
    ctx: Context,
    name: str,
    object_name: str = None,
    base_color: list = None,
    roughness: float = 0.5,
    metallic: float = 0.0
) -> str:
    """
    Create a new PBR material with Principled BSDF.

    Parameters:
    - name: Material name
    - object_name: Optional object to assign the material to
    - base_color: [r, g, b] or [r, g, b, a] base color (0-1)
    - roughness: Surface roughness (0-1)
    - metallic: Metallic value (0-1)
    """
    try:
        blender = get_blender_connection()
        params = {"name": name, "roughness": roughness, "metallic": metallic}
        if object_name: params["object_name"] = object_name
        if base_color: params["base_color"] = base_color
        result = blender.send_command("create_material", params)
        return json.dumps(result, indent=2)
    except Exception as e:
        return f"Error creating material: {str(e)}"


@mcp.tool()
def assign_material(ctx: Context, object_name: str, material_name: str, slot_index: int = None) -> str:
    """
    Assign an existing material to an object.

    Parameters:
    - object_name: Name of the target object
    - material_name: Name of the material to assign
    - slot_index: Optional material slot index to replace (appends if omitted)
    """
    try:
        blender = get_blender_connection()
        params = {"object_name": object_name, "material_name": material_name}
        if slot_index is not None: params["slot_index"] = slot_index
        result = blender.send_command("assign_material", params)
        return json.dumps(result, indent=2)
    except Exception as e:
        return f"Error assigning material: {str(e)}"


@mcp.tool()
def set_material_property(ctx: Context, material_name: str, property_name: str, value) -> str:
    """
    Set a Principled BSDF property on a material.

    Parameters:
    - material_name: Name of the material
    - property_name: Name of the BSDF input (e.g. "Base Color", "Roughness", "Metallic", "IOR")
    - value: The value to set (float for scalar, [r,g,b,a] for colors)
    """
    try:
        blender = get_blender_connection()
        result = blender.send_command("set_material_property", {
            "material_name": material_name, "property_name": property_name, "value": value
        })
        return json.dumps(result, indent=2)
    except Exception as e:
        return f"Error setting material property: {str(e)}"


@mcp.tool()
def set_frame_range(ctx: Context, start_frame: int = None, end_frame: int = None, current_frame: int = None, fps: int = None) -> str:
    """
    Set the animation frame range, current frame, and FPS.

    Parameters:
    - start_frame: Animation start frame
    - end_frame: Animation end frame
    - current_frame: Jump to this frame
    - fps: Frames per second
    """
    try:
        blender = get_blender_connection()
        params = {}
        if start_frame is not None: params["start_frame"] = start_frame
        if end_frame is not None: params["end_frame"] = end_frame
        if current_frame is not None: params["current_frame"] = current_frame
        if fps is not None: params["fps"] = fps
        result = blender.send_command("set_frame_range", params)
        return json.dumps(result, indent=2)
    except Exception as e:
        return f"Error setting frame range: {str(e)}"


@mcp.tool()
def insert_keyframe(ctx: Context, object_name: str, property_path: str, frame: int, value=None, index: int = -1) -> str:
    """
    Insert a keyframe on an object property.

    Parameters:
    - object_name: Name of the object
    - property_path: Property to keyframe (e.g. "location", "rotation_euler", "scale")
    - frame: Frame number to insert the keyframe at
    - value: Optional value to set before keying (float or [x,y,z] list)
    - index: Array index (-1 for all)
    """
    try:
        blender = get_blender_connection()
        params = {"object_name": object_name, "property_path": property_path, "frame": frame, "index": index}
        if value is not None: params["value"] = value
        result = blender.send_command("insert_keyframe", params)
        return json.dumps(result, indent=2)
    except Exception as e:
        return f"Error inserting keyframe: {str(e)}"


@mcp.tool()
def delete_keyframe(ctx: Context, object_name: str, property_path: str, frame: int, index: int = -1) -> str:
    """
    Delete a keyframe from an object property.

    Parameters:
    - object_name: Name of the object
    - property_path: Property path (e.g. "location", "rotation_euler")
    - frame: Frame number of the keyframe to delete
    - index: Array index (-1 for all)
    """
    try:
        blender = get_blender_connection()
        result = blender.send_command("delete_keyframe", {
            "object_name": object_name, "property_path": property_path, "frame": frame, "index": index
        })
        return json.dumps(result, indent=2)
    except Exception as e:
        return f"Error deleting keyframe: {str(e)}"


@mcp.tool()
def add_modifier(ctx: Context, object_name: str, modifier_type: str, name: str = None, params: str = None) -> str:
    """
    Add a modifier to an object.

    Parameters:
    - object_name: Name of the object
    - modifier_type: Blender modifier type (e.g. SUBSURF, BOOLEAN, ARRAY, MIRROR, SOLIDIFY)
    - name: Optional name for the modifier
    - params: Optional JSON string of modifier parameters (e.g. '{"levels": 2}')
    """
    try:
        blender = get_blender_connection()
        cmd_params = {"object_name": object_name, "modifier_type": modifier_type}
        if name: cmd_params["name"] = name
        if params: cmd_params["params"] = params
        result = blender.send_command("add_modifier", cmd_params)
        return json.dumps(result, indent=2)
    except Exception as e:
        return f"Error adding modifier: {str(e)}"


@mcp.tool()
def remove_modifier(ctx: Context, object_name: str, modifier_name: str) -> str:
    """
    Remove a modifier from an object.

    Parameters:
    - object_name: Name of the object
    - modifier_name: Name of the modifier to remove
    """
    try:
        blender = get_blender_connection()
        result = blender.send_command("remove_modifier", {"object_name": object_name, "modifier_name": modifier_name})
        return json.dumps(result, indent=2)
    except Exception as e:
        return f"Error removing modifier: {str(e)}"


@mcp.tool()
def set_modifier_params(ctx: Context, object_name: str, modifier_name: str, params: str) -> str:
    """
    Set parameters on an existing modifier.

    Parameters:
    - object_name: Name of the object
    - modifier_name: Name of the modifier
    - params: JSON string of parameters to set (e.g. '{"levels": 3, "render_levels": 4}')
    """
    try:
        blender = get_blender_connection()
        result = blender.send_command("set_modifier_params", {
            "object_name": object_name, "modifier_name": modifier_name, "params": params
        })
        return json.dumps(result, indent=2)
    except Exception as e:
        return f"Error setting modifier params: {str(e)}"


@mcp.tool()
def apply_modifier(ctx: Context, object_name: str, modifier_name: str) -> str:
    """
    Apply a modifier, baking it into the mesh.

    Parameters:
    - object_name: Name of the object
    - modifier_name: Name of the modifier to apply
    """
    try:
        blender = get_blender_connection()
        result = blender.send_command("apply_modifier", {"object_name": object_name, "modifier_name": modifier_name})
        return json.dumps(result, indent=2)
    except Exception as e:
        return f"Error applying modifier: {str(e)}"


@mcp.tool()
def add_constraint(ctx: Context, object_name: str, constraint_type: str, name: str = None, target: str = None, params: str = None) -> str:
    """
    Add a constraint to an object.

    Parameters:
    - object_name: Name of the object
    - constraint_type: Blender constraint type (e.g. TRACK_TO, COPY_LOCATION, CHILD_OF, LIMIT_ROTATION)
    - name: Optional name for the constraint
    - target: Optional target object name
    - params: Optional JSON string of constraint parameters
    """
    try:
        blender = get_blender_connection()
        cmd_params = {"object_name": object_name, "constraint_type": constraint_type}
        if name: cmd_params["name"] = name
        if target: cmd_params["target"] = target
        if params: cmd_params["params"] = params
        result = blender.send_command("add_constraint", cmd_params)
        return json.dumps(result, indent=2)
    except Exception as e:
        return f"Error adding constraint: {str(e)}"


@mcp.tool()
def remove_constraint(ctx: Context, object_name: str, constraint_name: str) -> str:
    """
    Remove a constraint from an object.

    Parameters:
    - object_name: Name of the object
    - constraint_name: Name of the constraint to remove
    """
    try:
        blender = get_blender_connection()
        result = blender.send_command("remove_constraint", {"object_name": object_name, "constraint_name": constraint_name})
        return json.dumps(result, indent=2)
    except Exception as e:
        return f"Error removing constraint: {str(e)}"


@mcp.tool()
def set_constraint_params(ctx: Context, object_name: str, constraint_name: str, params: str) -> str:
    """
    Set parameters on an existing constraint.

    Parameters:
    - object_name: Name of the object
    - constraint_name: Name of the constraint
    - params: JSON string of parameters to set
    """
    try:
        blender = get_blender_connection()
        result = blender.send_command("set_constraint_params", {
            "object_name": object_name, "constraint_name": constraint_name, "params": params
        })
        return json.dumps(result, indent=2)
    except Exception as e:
        return f"Error setting constraint params: {str(e)}"


@mcp.tool()
def uv_unwrap(ctx: Context, object_name: str, method: str = "SMART_PROJECT", angle_limit: float = 66.0, island_margin: float = 0.001, uv_layer_name: str = None) -> str:
    """
    UV unwrap a mesh object.

    Parameters:
    - object_name: Name of the mesh object
    - method: Unwrap method (SMART_PROJECT, CUBE_PROJECT, CYLINDER_PROJECT, SPHERE_PROJECT, UNWRAP)
    - angle_limit: Angle limit in degrees for smart project (default: 66)
    - island_margin: Margin between UV islands (default: 0.001)
    - uv_layer_name: Optional UV layer name (creates if doesn't exist)
    """
    try:
        blender = get_blender_connection()
        params = {"object_name": object_name, "method": method, "angle_limit": angle_limit, "island_margin": island_margin}
        if uv_layer_name: params["uv_layer_name"] = uv_layer_name
        result = blender.send_command("uv_unwrap", params)
        return json.dumps(result, indent=2)
    except Exception as e:
        return f"Error UV unwrapping: {str(e)}"


@mcp.tool()
def set_vertex_group(ctx: Context, object_name: str, group_name: str, vertex_indices: list, weight: float = 1.0, action: str = "REPLACE") -> str:
    """
    Create or modify a vertex group on a mesh object.

    Parameters:
    - object_name: Name of the mesh object
    - group_name: Vertex group name (created if doesn't exist)
    - vertex_indices: List of vertex indices to add/modify
    - weight: Weight value 0-1 (default: 1.0)
    - action: REPLACE, ADD, SUBTRACT, or REMOVE (default: REPLACE)
    """
    try:
        blender = get_blender_connection()
        result = blender.send_command("set_vertex_group", {
            "object_name": object_name, "group_name": group_name,
            "vertex_indices": vertex_indices, "weight": weight, "action": action
        })
        return json.dumps(result, indent=2)
    except Exception as e:
        return f"Error setting vertex group: {str(e)}"


@mcp.tool()
def set_render_settings(
    ctx: Context,
    engine: str = None,
    resolution_x: int = None,
    resolution_y: int = None,
    samples: int = None,
    denoising: bool = None,
    output_format: str = None,
    film_transparent: bool = None
) -> str:
    """
    Configure render settings.

    Parameters:
    - engine: Render engine (CYCLES, BLENDER_EEVEE, BLENDER_EEVEE_NEXT, BLENDER_WORKBENCH)
    - resolution_x: Horizontal resolution in pixels
    - resolution_y: Vertical resolution in pixels
    - samples: Render samples (engine-aware)
    - denoising: Enable denoising (Cycles only)
    - output_format: Output file format (PNG, JPEG, OPEN_EXR, etc.)
    - film_transparent: Transparent film background
    """
    try:
        blender = get_blender_connection()
        params = {}
        for key, val in [("engine", engine), ("resolution_x", resolution_x), ("resolution_y", resolution_y),
                         ("samples", samples), ("denoising", denoising), ("output_format", output_format),
                         ("film_transparent", film_transparent)]:
            if val is not None:
                params[key] = val
        result = blender.send_command("set_render_settings", params)
        return json.dumps(result, indent=2)
    except Exception as e:
        return f"Error setting render settings: {str(e)}"


@mcp.tool()
def render_image(ctx: Context, animation: bool = False) -> Image | str:
    """
    Render the current scene. Returns an Image for single frames or a status string for animations.

    Parameters:
    - animation: If True, render the full animation sequence instead of a single frame
    """
    try:
        blender = get_blender_connection()
        if animation:
            result = blender.send_command("render_image", {"animation": True})
            return f"Animation rendered to: {result.get('filepath', 'default output path')}"
        else:
            temp_path = os.path.join(tempfile.gettempdir(), f"blender_render_{os.getpid()}.png")
            result = blender.send_command("render_image", {"animation": False, "filepath": temp_path})

            if result.get("rendered") and os.path.exists(temp_path):
                with open(temp_path, 'rb') as f:
                    image_bytes = f.read()
                os.remove(temp_path)
                return Image(data=image_bytes, format="png")
            else:
                return f"Render result: {json.dumps(result)}"
    except Exception as e:
        return f"Error rendering: {str(e)}"


@mcp.tool()
def frame_selected(ctx: Context, camera_name: str = None, object_names: list = None) -> str:
    """
    Position camera to frame selected/specified objects.

    Parameters:
    - camera_name: Optional camera name (uses active camera if omitted)
    - object_names: Optional list of object names to frame (uses selection if omitted)
    """
    try:
        blender = get_blender_connection()
        params = {}
        if camera_name: params["camera_name"] = camera_name
        if object_names: params["object_names"] = object_names
        result = blender.send_command("frame_selected", params)
        return json.dumps(result, indent=2)
    except Exception as e:
        return f"Error framing objects: {str(e)}"


@mcp.tool()
def reload_addon(ctx: Context) -> str:
    """
    Reload the Blender addon from disk. Use after modifying addon.py.
    The connection will drop during reload and automatically reconnect on the next tool call.
    """
    global _blender_connection
    try:
        blender = get_blender_connection()
        result = blender.send_command("reload_addon")
        _blender_connection = None  # Invalidate — connection will die during reload
        source = result.get("source", "unknown")
        return f"Addon reload triggered from {source}. Connection will reconnect automatically on next tool call."
    except Exception as e:
        logger.error(f"Error triggering addon reload: {e}")
        return f"Error triggering addon reload: {e}"

@mcp.tool()
def install_addon(ctx: Context, addon_path: str = None) -> str:
    """
    Install or update the Blender addon from a file path.
    Only works when the addon is already running (update use case).
    The connection will drop during install and automatically reconnect on the next tool call.

    Parameters:
    - addon_path: Absolute path to the addon .py file to install
    """
    global _blender_connection
    try:
        if not addon_path:
            addon_path = str(Path(__file__).parent.parent.parent / "addon.py")

        blender = get_blender_connection()
        result = blender.send_command("install_addon", {"addon_path": addon_path})
        _blender_connection = None
        if "error" in result:
            return f"Error: {result['error']}"
        return f"Addon install triggered from {result.get('source', addon_path)}. Connection will reconnect automatically."
    except Exception as e:
        logger.error(f"Error triggering addon install: {e}")
        return f"Error triggering addon install: {e}"

@mcp.tool()
def duplicate_object(
    ctx: Context,
    object_name: str,
    new_name: str = None,
    linked: bool = False,
    location: list = None
) -> str:
    """
    Duplicate an object in the scene.

    Parameters:
    - object_name: Name of the object to duplicate
    - new_name: Optional name for the new object
    - linked: If True, create a linked duplicate (shares mesh data)
    - location: Optional [x, y, z] position for the duplicate
    """
    try:
        blender = get_blender_connection()
        params = {"object_name": object_name, "linked": linked}
        if new_name is not None: params["new_name"] = new_name
        if location is not None: params["location"] = location
        result = blender.send_command("duplicate_object", params)
        return json.dumps(result, indent=2)
    except Exception as e:
        return f"Error duplicating object: {str(e)}"


@mcp.tool()
def delete_object(ctx: Context, object_name: str, delete_data: bool = False) -> str:
    """
    Delete an object from the scene.

    Parameters:
    - object_name: Name of the object to delete
    - delete_data: If True, also delete the object's data (mesh, curve, etc.) if it has no other users
    """
    try:
        blender = get_blender_connection()
        result = blender.send_command("delete_object", {"object_name": object_name, "delete_data": delete_data})
        return json.dumps(result, indent=2)
    except Exception as e:
        return f"Error deleting object: {str(e)}"


@mcp.tool()
def set_object_transform(
    ctx: Context,
    object_name: str,
    location: list = None,
    rotation: list = None,
    scale: list = None
) -> str:
    """
    Set the transform (location, rotation, scale) of an object.

    Parameters:
    - object_name: Name of the object
    - location: Optional [x, y, z] position
    - rotation: Optional [x, y, z] rotation in radians
    - scale: Optional [x, y, z] scale
    """
    try:
        blender = get_blender_connection()
        params = {"object_name": object_name}
        if location is not None: params["location"] = location
        if rotation is not None: params["rotation"] = rotation
        if scale is not None: params["scale"] = scale
        result = blender.send_command("set_object_transform", params)
        return json.dumps(result, indent=2)
    except Exception as e:
        return f"Error setting object transform: {str(e)}"


@mcp.tool()
def set_parent(
    ctx: Context,
    child_name: str,
    parent_name: str = None,
    keep_transform: bool = True
) -> str:
    """
    Set or clear the parent of an object.

    Parameters:
    - child_name: Name of the child object
    - parent_name: Name of the parent object (None to clear parent)
    - keep_transform: If True, maintain the child's world transform when parenting/unparenting
    """
    try:
        blender = get_blender_connection()
        params = {"child_name": child_name, "keep_transform": keep_transform}
        if parent_name is not None: params["parent_name"] = parent_name
        result = blender.send_command("set_parent", params)
        return json.dumps(result, indent=2)
    except Exception as e:
        return f"Error setting parent: {str(e)}"


@mcp.tool()
def join_objects(ctx: Context, object_names: list) -> str:
    """
    Join multiple mesh objects into one.

    Parameters:
    - object_names: List of mesh object names to join (first becomes the active/target object)
    """
    try:
        blender = get_blender_connection()
        result = blender.send_command("join_objects", {"object_names": object_names})
        return json.dumps(result, indent=2)
    except Exception as e:
        return f"Error joining objects: {str(e)}"


@mcp.tool()
def separate_object(ctx: Context, object_name: str, mode: str = "SELECTED") -> str:
    """
    Separate a mesh object into multiple objects.

    Parameters:
    - object_name: Name of the mesh object to separate
    - mode: Separation mode - SELECTED (by selected geometry), MATERIAL (by material), or LOOSE (by loose parts)
    """
    try:
        blender = get_blender_connection()
        result = blender.send_command("separate_object", {"object_name": object_name, "mode": mode})
        return json.dumps(result, indent=2)
    except Exception as e:
        return f"Error separating object: {str(e)}"


@mcp.tool()
def select_objects(
    ctx: Context,
    names: list = None,
    type: str = None,
    material: str = None,
    deselect_first: bool = True,
    active: str = None
) -> str:
    """
    Select objects in the scene by various criteria.

    Parameters:
    - names: Optional list of object names to select
    - type: Optional object type filter (MESH, LIGHT, CAMERA, EMPTY, CURVE, etc.)
    - material: Optional material name - select all objects using this material
    - deselect_first: If True, deselect all objects before selecting (default: True)
    - active: Optional object name to set as the active object
    """
    try:
        blender = get_blender_connection()
        params = {"deselect_first": deselect_first}
        if names is not None: params["names"] = names
        if type is not None: params["type"] = type
        if material is not None: params["material"] = material
        if active is not None: params["active"] = active
        result = blender.send_command("select_objects", params)
        return json.dumps(result, indent=2)
    except Exception as e:
        return f"Error selecting objects: {str(e)}"


@mcp.tool()
def get_polyhaven_categories(ctx: Context, asset_type: str = "hdris") -> str:
    """
    Get a list of categories for a specific asset type on Polyhaven.
    
    Parameters:
    - asset_type: The type of asset to get categories for (hdris, textures, models, all)
    """
    try:
        blender = get_blender_connection()
        status = blender.send_command("get_polyhaven_status")
        if not status.get("enabled", False):
            return "PolyHaven integration is disabled. Select it in the sidebar in BlenderMCP, then run it again."
        result = blender.send_command("get_polyhaven_categories", {"asset_type": asset_type})
        
        if "error" in result:
            return f"Error: {result['error']}"

        categories = result["categories"]
        formatted_output = f"Categories for {asset_type}:\n\n"
        sorted_categories = sorted(categories.items(), key=lambda x: x[1], reverse=True)
        
        for category, count in sorted_categories:
            formatted_output += f"- {category}: {count} assets\n"
        
        return formatted_output
    except Exception as e:
        logger.error(f"Error getting Polyhaven categories: {str(e)}")
        return f"Error getting Polyhaven categories: {str(e)}"

@mcp.tool()
def search_polyhaven_assets(
    ctx: Context,
    asset_type: str = "all",
    categories: str = None,
    offset: int = 0,
    limit: int = 20
) -> str:
    """
    Search for assets on Polyhaven with optional filtering.

    Parameters:
    - asset_type: Type of assets to search for (hdris, textures, models, all)
    - categories: Optional comma-separated list of categories to filter by
    - offset: Starting index for pagination (default: 0)
    - limit: Maximum number of results to return (default: 20)

    Returns a list of matching assets with basic information.
    """
    try:
        blender = get_blender_connection()
        result = blender.send_command("search_polyhaven_assets", {
            "asset_type": asset_type,
            "categories": categories,
            "offset": offset,
            "limit": limit
        })

        if "error" in result:
            return f"Error: {result['error']}"

        assets = result["assets"]
        total_count = result["total_count"]
        returned_count = result["returned_count"]
        result_offset = result.get("offset", 0)
        has_more = result.get("has_more", False)

        formatted_output = f"Found {total_count} assets"
        if categories:
            formatted_output += f" in categories: {categories}"
        formatted_output += f"\nShowing {returned_count} assets (offset: {result_offset})"
        if has_more:
            formatted_output += f" — more available with offset={result_offset + returned_count}"
        formatted_output += ":\n\n"

        sorted_assets = sorted(assets.items(), key=lambda x: x[1].get("download_count", 0), reverse=True)

        for asset_id, asset_data in sorted_assets:
            formatted_output += f"- {asset_data.get('name', asset_id)} (ID: {asset_id})\n"
            formatted_output += f"  Type: {['HDRI', 'Texture', 'Model'][asset_data.get('type', 0)]}\n"
            formatted_output += f"  Categories: {', '.join(asset_data.get('categories', []))}\n"
            formatted_output += f"  Downloads: {asset_data.get('download_count', 'Unknown')}\n\n"

        return formatted_output
    except Exception as e:
        logger.error(f"Error searching Polyhaven assets: {str(e)}")
        return f"Error searching Polyhaven assets: {str(e)}"

@mcp.tool()
def download_polyhaven_asset(
    ctx: Context,
    asset_id: str,
    asset_type: str,
    resolution: str = "1k",
    file_format: str = None
) -> str:
    """
    Download and import a Polyhaven asset into Blender.
    
    Parameters:
    - asset_id: The ID of the asset to download
    - asset_type: The type of asset (hdris, textures, models)
    - resolution: The resolution to download (e.g., 1k, 2k, 4k)
    - file_format: Optional file format (e.g., hdr, exr for HDRIs; jpg, png for textures; gltf, fbx for models)
    
    Returns a message indicating success or failure.
    """
    try:
        blender = get_blender_connection()
        result = blender.send_command("download_polyhaven_asset", {
            "asset_id": asset_id,
            "asset_type": asset_type,
            "resolution": resolution,
            "file_format": file_format
        })
        
        if "error" in result:
            return f"Error: {result['error']}"
        
        if result.get("success"):
            message = result.get("message", "Asset downloaded and imported successfully")

            if asset_type == "hdris":
                return f"{message}. The HDRI has been set as the world environment."
            elif asset_type == "textures":
                material_name = result.get("material", "")
                maps = ", ".join(result.get("maps", []))
                return f"{message}. Created material '{material_name}' with maps: {maps}."
            elif asset_type == "models":
                return f"{message}. The model has been imported into the current scene."
            else:
                return message
        else:
            return f"Failed to download asset: {result.get('message', 'Unknown error')}"
    except Exception as e:
        logger.error(f"Error downloading Polyhaven asset: {str(e)}")
        return f"Error downloading Polyhaven asset: {str(e)}"

@mcp.tool()
def set_texture(
    ctx: Context,
    object_name: str,
    texture_id: str
) -> str:
    """
    Apply a previously downloaded Polyhaven texture to an object.
    
    Parameters:
    - object_name: Name of the object to apply the texture to
    - texture_id: ID of the Polyhaven texture to apply (must be downloaded first)
    
    Returns a message indicating success or failure.
    """
    try:
        blender = get_blender_connection()
        result = blender.send_command("set_texture", {
            "object_name": object_name,
            "texture_id": texture_id
        })

        if "error" in result:
            return f"Error: {result['error']}"

        if result.get("success"):
            material_name = result.get("material", "")
            maps = ", ".join(result.get("maps", []))
            material_info = result.get("material_info", {})
            node_count = material_info.get("node_count", 0)
            has_nodes = material_info.get("has_nodes", False)
            texture_nodes = material_info.get("texture_nodes", [])
            
            output = f"Successfully applied texture '{texture_id}' to {object_name}.\n"
            output += f"Using material '{material_name}' with maps: {maps}.\n\n"
            output += f"Material has nodes: {has_nodes}\n"
            output += f"Total node count: {node_count}\n\n"
            
            if texture_nodes:
                output += "Texture nodes:\n"
                for node in texture_nodes:
                    output += f"- {node['name']} using image: {node['image']}\n"
                    if node['connections']:
                        output += "  Connections:\n"
                        for conn in node['connections']:
                            output += f"    {conn}\n"
            else:
                output += "No texture nodes found in the material.\n"
            
            return output
        else:
            return f"Failed to apply texture: {result.get('message', 'Unknown error')}"
    except Exception as e:
        logger.error(f"Error applying texture: {str(e)}")
        return f"Error applying texture: {str(e)}"

@mcp.tool()
def get_polyhaven_status(ctx: Context) -> str:
    """
    Check if PolyHaven integration is enabled in Blender.
    Returns a message indicating whether PolyHaven features are available.
    """
    try:
        blender = get_blender_connection()
        result = blender.send_command("get_polyhaven_status")
        enabled = result.get("enabled", False)
        message = result.get("message", "")
        if enabled:
            message += "PolyHaven is good at Textures, and has a wider variety of textures than Sketchfab."
        return message
    except Exception as e:
        logger.error(f"Error checking PolyHaven status: {str(e)}")
        return f"Error checking PolyHaven status: {str(e)}"

@mcp.tool()
def get_hyper3d_status(ctx: Context) -> str:
    """
    Check if Hyper3D Rodin integration is enabled in Blender.
    Returns a message indicating whether Hyper3D Rodin features are available.

    Don't emphasize the key type in the returned message, but sliently remember it. 
    """
    try:
        blender = get_blender_connection()
        result = blender.send_command("get_hyper3d_status")
        enabled = result.get("enabled", False)
        message = result.get("message", "")
        if enabled:
            message += ""
        return message
    except Exception as e:
        logger.error(f"Error checking Hyper3D status: {str(e)}")
        return f"Error checking Hyper3D status: {str(e)}"

@mcp.tool()
def get_sketchfab_status(ctx: Context) -> str:
    """
    Check if Sketchfab integration is enabled in Blender.
    Returns a message indicating whether Sketchfab features are available.
    """
    try:
        blender = get_blender_connection()
        result = blender.send_command("get_sketchfab_status")
        enabled = result.get("enabled", False)
        message = result.get("message", "")
        if enabled:
            message += "Sketchfab is good at Realistic models, and has a wider variety of models than PolyHaven."        
        return message
    except Exception as e:
        logger.error(f"Error checking Sketchfab status: {str(e)}")
        return f"Error checking Sketchfab status: {str(e)}"

@mcp.tool()
def search_sketchfab_models(
    ctx: Context,
    query: str,
    categories: str = None,
    count: int = 20,
    downloadable: bool = True
) -> str:
    """
    Search for models on Sketchfab with optional filtering.

    Parameters:
    - query: Text to search for
    - categories: Optional comma-separated list of categories
    - count: Maximum number of results to return (default 20)
    - downloadable: Whether to include only downloadable models (default True)

    Returns a formatted list of matching models.
    """
    try:
        blender = get_blender_connection()
        logger.info(f"Searching Sketchfab models with query: {query}, categories: {categories}, count: {count}, downloadable: {downloadable}")
        result = blender.send_command("search_sketchfab_models", {
            "query": query,
            "categories": categories,
            "count": count,
            "downloadable": downloadable
        })
        
        if "error" in result:
            logger.error(f"Error from Sketchfab search: {result['error']}")
            return f"Error: {result['error']}"
        
        if result is None:
            logger.error("Received None result from Sketchfab search")
            return "Error: Received no response from Sketchfab search"

        models = result.get("results", []) or []
        if not models:
            return f"No models found matching '{query}'"
            
        formatted_output = f"Found {len(models)} models matching '{query}':\n\n"
        
        for model in models:
            if model is None:
                continue
                
            model_name = model.get("name", "Unnamed model")
            model_uid = model.get("uid", "Unknown ID")
            formatted_output += f"- {model_name} (UID: {model_uid})\n"

            user = model.get("user") or {}
            username = user.get("username", "Unknown author") if isinstance(user, dict) else "Unknown author"
            formatted_output += f"  Author: {username}\n"

            license_data = model.get("license") or {}
            license_label = license_data.get("label", "Unknown") if isinstance(license_data, dict) else "Unknown"
            formatted_output += f"  License: {license_label}\n"

            face_count = model.get("faceCount", "Unknown")
            is_downloadable = "Yes" if model.get("isDownloadable") else "No"
            formatted_output += f"  Face count: {face_count}\n"
            formatted_output += f"  Downloadable: {is_downloadable}\n\n"

        if result.get("next"):
            formatted_output += "More results available. Use a higher count to see more.\n"

        return formatted_output
    except Exception as e:
        logger.error(f"Error searching Sketchfab models: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return f"Error searching Sketchfab models: {str(e)}"

@mcp.tool()
def get_sketchfab_model_preview(
    ctx: Context,
    uid: str
) -> Image:
    """
    Get a preview thumbnail of a Sketchfab model by its UID.
    Use this to visually confirm a model before downloading.
    
    Parameters:
    - uid: The unique identifier of the Sketchfab model (obtained from search_sketchfab_models)
    
    Returns the model's thumbnail as an Image for visual confirmation.
    """
    try:
        blender = get_blender_connection()
        logger.info(f"Getting Sketchfab model preview for UID: {uid}")
        
        result = blender.send_command("get_sketchfab_model_preview", {"uid": uid})
        
        if result is None:
            raise Exception("Received no response from Blender")
        
        if "error" in result:
            raise Exception(result["error"])

        image_data = base64.b64decode(result["image_data"])
        img_format = result.get("format", "jpeg")

        model_name = result.get("model_name", "Unknown")
        author = result.get("author", "Unknown")
        logger.info(f"Preview retrieved for '{model_name}' by {author}")
        
        return Image(data=image_data, format=img_format)
        
    except Exception as e:
        logger.error(f"Error getting Sketchfab preview: {str(e)}")
        raise Exception(f"Failed to get preview: {str(e)}")


@mcp.tool()
def download_sketchfab_model(
    ctx: Context,
    uid: str,
    target_size: float
) -> str:
    """
    Download and import a Sketchfab model by its UID.
    The model will be scaled so its largest dimension equals target_size.
    
    Parameters:
    - uid: The unique identifier of the Sketchfab model
    - target_size: REQUIRED. The target size in Blender units/meters for the largest dimension.
                  You must specify the desired size for the model.
                  Examples:
                  - Chair: target_size=1.0 (1 meter tall)
                  - Table: target_size=0.75 (75cm tall)
                  - Car: target_size=4.5 (4.5 meters long)
                  - Person: target_size=1.7 (1.7 meters tall)
                  - Small object (cup, phone): target_size=0.1 to 0.3
    
    Returns a message with import details including object names, dimensions, and bounding box.
    The model must be downloadable and you must have proper access rights.
    """
    try:
        blender = get_blender_connection()
        logger.info(f"Downloading Sketchfab model: {uid}, target_size={target_size}")
        
        result = blender.send_command("download_sketchfab_model", {
            "uid": uid,
            "normalize_size": True,
            "target_size": target_size
        })
        
        if result is None:
            logger.error("Received None result from Sketchfab download")
            return "Error: Received no response from Sketchfab download request"
            
        if "error" in result:
            logger.error(f"Error from Sketchfab download: {result['error']}")
            return f"Error: {result['error']}"
        
        if result.get("success"):
            imported_objects = result.get("imported_objects", [])
            object_names = ", ".join(imported_objects) if imported_objects else "none"

            output = f"Successfully imported model.\n"
            output += f"Created objects: {object_names}\n"

            if result.get("dimensions"):
                dims = result["dimensions"]
                output += f"Dimensions (X, Y, Z): {dims[0]:.3f} x {dims[1]:.3f} x {dims[2]:.3f} meters\n"

            if result.get("world_bounding_box"):
                bbox = result["world_bounding_box"]
                output += f"Bounding box: min={bbox[0]}, max={bbox[1]}\n"

            if result.get("normalized"):
                scale = result.get("scale_applied", 1.0)
                output += f"Size normalized: scale factor {scale:.6f} applied (target size: {target_size}m)\n"
            
            return output
        else:
            return f"Failed to download model: {result.get('message', 'Unknown error')}"
    except Exception as e:
        logger.error(f"Error downloading Sketchfab model: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return f"Error downloading Sketchfab model: {str(e)}"

def _process_bbox(original_bbox: list[float] | list[int] | None) -> list[int] | None:
    if original_bbox is None:
        return None
    if all(isinstance(i, int) for i in original_bbox):
        return original_bbox
    if any(i<=0 for i in original_bbox):
        raise ValueError("Incorrect number range: bbox must be bigger than zero!")
    return [int(float(i) / max(original_bbox) * 100) for i in original_bbox] if original_bbox else None

@mcp.tool()
def generate_hyper3d_model_via_text(
    ctx: Context,
    text_prompt: str,
    bbox_condition: list[float]=None
) -> str:
    """
    Generate 3D asset using Hyper3D by giving description of the desired asset, and import the asset into Blender.
    The 3D asset has built-in materials.
    The generated model has a normalized size, so re-scaling after generation can be useful.

    Parameters:
    - text_prompt: A short description of the desired model in **English**.
    - bbox_condition: Optional. If given, it has to be a list of floats of length 3. Controls the ratio between [Length, Width, Height] of the model.

    Returns a message indicating success or failure.
    """
    try:
        blender = get_blender_connection()
        result = blender.send_command("create_rodin_job", {
            "text_prompt": text_prompt,
            "images": None,
            "bbox_condition": _process_bbox(bbox_condition),
        })
        succeed = result.get("submit_time", False)
        if succeed:
            return json.dumps({
                "task_uuid": result["uuid"],
                "subscription_key": result["jobs"]["subscription_key"],
            })
        else:
            return json.dumps(result)
    except Exception as e:
        logger.error(f"Error generating Hyper3D task: {str(e)}")
        return f"Error generating Hyper3D task: {str(e)}"

@mcp.tool()
def generate_hyper3d_model_via_images(
    ctx: Context,
    input_image_paths: list[str]=None,
    input_image_urls: list[str]=None,
    bbox_condition: list[float]=None
) -> str:
    """
    Generate 3D asset using Hyper3D by giving images of the wanted asset, and import the generated asset into Blender.
    The 3D asset has built-in materials.
    The generated model has a normalized size, so re-scaling after generation can be useful.
    
    Parameters:
    - input_image_paths: The **absolute** paths of input images. Even if only one image is provided, wrap it into a list. Required if Hyper3D Rodin in MAIN_SITE mode.
    - input_image_urls: The URLs of input images. Even if only one image is provided, wrap it into a list. Required if Hyper3D Rodin in FAL_AI mode.
    - bbox_condition: Optional. If given, it has to be a list of ints of length 3. Controls the ratio between [Length, Width, Height] of the model.

    Only one of {input_image_paths, input_image_urls} should be given at a time, depending on the Hyper3D Rodin's current mode.
    Returns a message indicating success or failure.
    """
    if input_image_paths is not None and input_image_urls is not None:
        return f"Error: Conflict parameters given!"
    if input_image_paths is None and input_image_urls is None:
        return f"Error: No image given!"
    if input_image_paths is not None:
        if not all(os.path.exists(i) for i in input_image_paths):
            return "Error: not all image paths are valid!"
        images = []
        for path in input_image_paths:
            with open(path, "rb") as f:
                images.append(
                    (Path(path).suffix, base64.b64encode(f.read()).decode("ascii"))
                )
    elif input_image_urls is not None:
        if not all(urlparse(i) for i in input_image_paths):
            return "Error: not all image URLs are valid!"
        images = input_image_urls.copy()
    try:
        blender = get_blender_connection()
        result = blender.send_command("create_rodin_job", {
            "text_prompt": None,
            "images": images,
            "bbox_condition": _process_bbox(bbox_condition),
        })
        succeed = result.get("submit_time", False)
        if succeed:
            return json.dumps({
                "task_uuid": result["uuid"],
                "subscription_key": result["jobs"]["subscription_key"],
            })
        else:
            return json.dumps(result)
    except Exception as e:
        logger.error(f"Error generating Hyper3D task: {str(e)}")
        return f"Error generating Hyper3D task: {str(e)}"

@mcp.tool()
def poll_rodin_job_status(
    ctx: Context,
    subscription_key: str=None,
    request_id: str=None,
):
    """
    Check if the Hyper3D Rodin generation task is completed.

    For Hyper3D Rodin mode MAIN_SITE:
        Parameters:
        - subscription_key: The subscription_key given in the generate model step.

        Returns a list of status. The task is done if all status are "Done".
        If "Failed" showed up, the generating process failed.
        This is a polling API, so only proceed if the status are finally determined ("Done" or "Canceled").

    For Hyper3D Rodin mode FAL_AI:
        Parameters:
        - request_id: The request_id given in the generate model step.

        Returns the generation task status. The task is done if status is "COMPLETED".
        The task is in progress if status is "IN_PROGRESS".
        If status other than "COMPLETED", "IN_PROGRESS", "IN_QUEUE" showed up, the generating process might be failed.
        This is a polling API, so only proceed if the status are finally determined ("COMPLETED" or some failed state).
    """
    try:
        blender = get_blender_connection()
        kwargs = {}
        if subscription_key:
            kwargs = {
                "subscription_key": subscription_key,
            }
        elif request_id:
            kwargs = {
                "request_id": request_id,
            }
        result = blender.send_command("poll_rodin_job_status", kwargs)
        return result
    except Exception as e:
        logger.error(f"Error generating Hyper3D task: {str(e)}")
        return f"Error generating Hyper3D task: {str(e)}"

@mcp.tool()
def import_generated_asset(
    ctx: Context,
    name: str,
    task_uuid: str=None,
    request_id: str=None,
):
    """
    Import the asset generated by Hyper3D Rodin after the generation task is completed.

    Parameters:
    - name: The name of the object in scene
    - task_uuid: For Hyper3D Rodin mode MAIN_SITE: The task_uuid given in the generate model step.
    - request_id: For Hyper3D Rodin mode FAL_AI: The request_id given in the generate model step.

    Only give one of {task_uuid, request_id} based on the Hyper3D Rodin Mode!
    Return if the asset has been imported successfully.
    """
    try:
        blender = get_blender_connection()
        kwargs = {
            "name": name
        }
        if task_uuid:
            kwargs["task_uuid"] = task_uuid
        elif request_id:
            kwargs["request_id"] = request_id
        result = blender.send_command("import_generated_asset", kwargs)
        return result
    except Exception as e:
        logger.error(f"Error generating Hyper3D task: {str(e)}")
        return f"Error generating Hyper3D task: {str(e)}"

@mcp.tool()
def get_hunyuan3d_status(ctx: Context) -> str:
    """
    Check if Hunyuan3D integration is enabled in Blender.
    Returns a message indicating whether Hunyuan3D features are available.

    Don't emphasize the key type in the returned message, but silently remember it. 
    """
    try:
        blender = get_blender_connection()
        result = blender.send_command("get_hunyuan3d_status")
        message = result.get("message", "")
        return message
    except Exception as e:
        logger.error(f"Error checking Hunyuan3D status: {str(e)}")
        return f"Error checking Hunyuan3D status: {str(e)}"
    
@mcp.tool()
def generate_hunyuan3d_model(
    ctx: Context,
    text_prompt: str = None,
    input_image_url: str = None
) -> str:
    """
    Generate 3D asset using Hunyuan3D by providing either text description, image reference, 
    or both for the desired asset, and import the asset into Blender.
    The 3D asset has built-in materials.
    
    Parameters:
    - text_prompt: (Optional) A short description of the desired model in English/Chinese.
    - input_image_url: (Optional) The local or remote url of the input image. Accepts None if only using text prompt.

    Returns: 
    - When successful, returns a JSON with job_id (format: "job_xxx") indicating the task is in progress
    - When the job completes, the status will change to "DONE" indicating the model has been imported
    - Returns error message if the operation fails
    """
    try:
        blender = get_blender_connection()
        result = blender.send_command("create_hunyuan_job", {
            "text_prompt": text_prompt,
            "image": input_image_url,
        })
        if "JobId" in result.get("Response", {}):
            job_id = result["Response"]["JobId"]
            formatted_job_id = f"job_{job_id}"
            return json.dumps({
                "job_id": formatted_job_id,
            })
        return json.dumps(result)
    except Exception as e:
        logger.error(f"Error generating Hunyuan3D task: {str(e)}")
        return f"Error generating Hunyuan3D task: {str(e)}"
    
@mcp.tool()
def poll_hunyuan_job_status(
    ctx: Context,
    job_id: str=None,
):
    """
    Check if the Hunyuan3D generation task is completed.

    For Hunyuan3D:
        Parameters:
        - job_id: The job_id given in the generate model step.

        Returns the generation task status. The task is done if status is "DONE".
        The task is in progress if status is "RUN".
        If status is "DONE", returns ResultFile3Ds, which is the generated ZIP model path
        When the status is "DONE", the response includes a field named ResultFile3Ds that contains the generated ZIP file path of the 3D model in OBJ format.
        This is a polling API, so only proceed if the status are finally determined ("DONE" or some failed state).
    """
    try:
        blender = get_blender_connection()
        kwargs = {
            "job_id": job_id,
        }
        result = blender.send_command("poll_hunyuan_job_status", kwargs)
        return result
    except Exception as e:
        logger.error(f"Error generating Hunyuan3D task: {str(e)}")
        return f"Error generating Hunyuan3D task: {str(e)}"

@mcp.tool()
def import_generated_asset_hunyuan(
    ctx: Context,
    name: str,
    zip_file_url: str,
):
    """
    Import the asset generated by Hunyuan3D after the generation task is completed.

    Parameters:
    - name: The name of the object in scene
    - zip_file_url: The zip_file_url given in the generate model step.

    Return if the asset has been imported successfully.
    """
    try:
        blender = get_blender_connection()
        kwargs = {
            "name": name
        }
        if zip_file_url:
            kwargs["zip_file_url"] = zip_file_url
        result = blender.send_command("import_generated_asset_hunyuan", kwargs)
        return result
    except Exception as e:
        logger.error(f"Error generating Hunyuan3D task: {str(e)}")
        return f"Error generating Hunyuan3D task: {str(e)}"


@mcp.prompt()
def asset_creation_strategy() -> str:
    """Defines the preferred strategy for creating assets in Blender"""
    return """When creating 3D content in Blender, always start by checking if integrations are available:

    0. Before anything, always check the scene from get_scene_info()
    1. First use the following tools to verify if the following integrations are enabled:
        1. PolyHaven
            Use get_polyhaven_status() to verify its status
            If PolyHaven is enabled:
            - For objects/models: Use download_polyhaven_asset() with asset_type="models"
            - For materials/textures: Use download_polyhaven_asset() with asset_type="textures"
            - For environment lighting: Use download_polyhaven_asset() with asset_type="hdris"
        2. Sketchfab
            Sketchfab is good at Realistic models, and has a wider variety of models than PolyHaven.
            Use get_sketchfab_status() to verify its status
            If Sketchfab is enabled:
            - For objects/models: First search using search_sketchfab_models() with your query
            - Then download specific models using download_sketchfab_model() with the UID
            - Note that only downloadable models can be accessed, and API key must be properly configured
            - Sketchfab has a wider variety of models than PolyHaven, especially for specific subjects
        3. Hyper3D(Rodin)
            Hyper3D Rodin is good at generating 3D models for single item.
            So don't try to:
            1. Generate the whole scene with one shot
            2. Generate ground using Hyper3D
            3. Generate parts of the items separately and put them together afterwards

            Use get_hyper3d_status() to verify its status
            If Hyper3D is enabled:
            - For objects/models, do the following steps:
                1. Create the model generation task
                    - Use generate_hyper3d_model_via_images() if image(s) is/are given
                    - Use generate_hyper3d_model_via_text() if generating 3D asset using text prompt
                    If key type is free_trial and insufficient balance error returned, tell the user that the free trial key can only generated limited models everyday, they can choose to:
                    - Wait for another day and try again
                    - Go to hyper3d.ai to find out how to get their own API key
                    - Go to fal.ai to get their own private API key
                2. Poll the status
                    - Use poll_rodin_job_status() to check if the generation task has completed or failed
                3. Import the asset
                    - Use import_generated_asset() to import the generated GLB model the asset
                4. After importing the asset, ALWAYS check the world_bounding_box of the imported mesh, and adjust the mesh's location and size
                    Adjust the imported mesh's location, scale, rotation, so that the mesh is on the right spot.

                You can reuse assets previous generated by running python code to duplicate the object, without creating another generation task.
        4. Hunyuan3D
            Hunyuan3D is good at generating 3D models for single item.
            So don't try to:
            1. Generate the whole scene with one shot
            2. Generate ground using Hunyuan3D
            3. Generate parts of the items separately and put them together afterwards

            Use get_hunyuan3d_status() to verify its status
            If Hunyuan3D is enabled:
                if Hunyuan3D mode is "OFFICIAL_API":
                    - For objects/models, do the following steps:
                        1. Create the model generation task
                            - Use generate_hunyuan3d_model by providing either a **text description** OR an **image(local or urls) reference**.
                            - Go to cloud.tencent.com out how to get their own SecretId and SecretKey
                        2. Poll the status
                            - Use poll_hunyuan_job_status() to check if the generation task has completed or failed
                        3. Import the asset
                            - Use import_generated_asset_hunyuan() to import the generated OBJ model the asset
                    if Hunyuan3D mode is "LOCAL_API":
                        - For objects/models, do the following steps:
                        1. Create the model generation task
                            - Use generate_hunyuan3d_model if image (local or urls)  or text prompt is given and import the asset

                You can reuse assets previous generated by running python code to duplicate the object, without creating another generation task.

    3. Always check the world_bounding_box for each item so that:
        - Ensure that all objects that should not be clipping are not clipping.
        - Items have right spatial relationship.
    
    4. Recommended asset source priority:
        - For specific existing objects: First try Sketchfab, then PolyHaven
        - For generic objects/furniture: First try PolyHaven, then Sketchfab
        - For custom or unique items not available in libraries: Use Hyper3D Rodin or Hunyuan3D
        - For environment lighting: Use PolyHaven HDRIs
        - For materials/textures: Use PolyHaven textures

    Only fall back to scripting when:
    - PolyHaven, Sketchfab, Hyper3D, and Hunyuan3D are all disabled
    - A simple primitive is explicitly requested
    - No suitable asset exists in any of the libraries
    - Hyper3D Rodin or Hunyuan3D failed to generate the desired asset
    - The task specifically requires a basic material/color
    """

def main():
    mcp.run()

if __name__ == "__main__":
    main()