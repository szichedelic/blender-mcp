# Code created by Siddharth Ahuja: www.github.com/ahujasid © 2025

import re
import bpy
import mathutils
import json
import threading
import socket
import time
import requests
import tempfile
import traceback
import os
import shutil
import zipfile
import logging
import struct
from bpy.props import IntProperty, BoolProperty
import io
from datetime import datetime
import hashlib, hmac, base64
import os.path as osp
from contextlib import redirect_stdout, redirect_stderr, suppress

logger = logging.getLogger("BlenderMCP")
logger.setLevel(logging.DEBUG)
if not logger.handlers:
    _console_handler = logging.StreamHandler()
    _console_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    logger.addHandler(_console_handler)

bl_info = {
    "name": "Blender MCP",
    "author": "BlenderMCP",
    "version": (1, 2),
    "blender": (3, 0, 0),
    "location": "View3D > Sidebar > BlenderMCP",
    "description": "Connect Blender to Claude via MCP",
    "category": "Interface",
}

RODIN_FREE_TRIAL_KEY = "k9TcfFoEhNd9cCPP2guHAHHHkctZHIRhZDywZ1euGUXwihbYLpOjQhofby80NJez"

# Add User-Agent as required by Poly Haven API
REQ_HEADERS = requests.utils.default_headers()
REQ_HEADERS.update({"User-Agent": "blender-mcp"})

MAX_OUTPUT_SIZE = 64 * 1024  # 64KB max output from code execution
MAX_EXEC_TIMEOUT = 120  # seconds max for code execution


def send_message(sock, data_dict):
    """Send a length-prefixed JSON message over a socket."""
    payload = json.dumps(data_dict).encode('utf-8')
    header = struct.pack('>I', len(payload))
    sock.sendall(header + payload)


def recv_message(sock, timeout=30.0):
    """Receive a length-prefixed JSON message from a socket."""
    old_timeout = sock.gettimeout()
    sock.settimeout(timeout)
    try:
        header = b''
        while len(header) < 4:
            chunk = sock.recv(4 - len(header))
            if not chunk:
                raise ConnectionError("Connection closed while reading message header")
            header += chunk
        msg_len = struct.unpack('>I', header)[0]
        data = b''
        while len(data) < msg_len:
            chunk = sock.recv(min(msg_len - len(data), 8192))
            if not chunk:
                raise ConnectionError("Connection closed while reading message body")
            data += chunk
        return json.loads(data.decode('utf-8'))
    finally:
        sock.settimeout(old_timeout)


class BlenderMCPServer:
    def __init__(self, host='localhost', port=9876):
        self.host = host
        self.port = port
        self.running = False
        self.socket = None
        self.server_thread = None

    def start(self):
        if self.running:
            logger.info("Server is already running")
            return

        self.running = True
        self._auth_token = bpy.context.scene.blendermcp_auth_token

        try:
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.socket.bind((self.host, self.port))
            self.socket.listen(1)

            self.server_thread = threading.Thread(target=self._server_loop)
            self.server_thread.daemon = True
            self.server_thread.start()

            logger.info(f"BlenderMCP server started on {self.host}:{self.port}")
        except Exception as e:
            logger.error(f"Failed to start server: {str(e)}")
            self.stop()

    def stop(self):
        self.running = False

        if self.socket:
            try:
                self.socket.close()
            except:
                pass
            self.socket = None

        if self.server_thread:
            try:
                if self.server_thread.is_alive():
                    self.server_thread.join(timeout=1.0)
            except:
                pass
            self.server_thread = None

        logger.info("BlenderMCP server stopped")

    def _server_loop(self):
        """Main server loop in a separate thread"""
        logger.info("Server thread started")
        self.socket.settimeout(1.0)  # Timeout to allow for stopping

        while self.running:
            try:
                try:
                    client, address = self.socket.accept()
                    logger.info(f"Connected to client: {address}")

                    client_thread = threading.Thread(
                        target=self._handle_client,
                        args=(client,)
                    )
                    client_thread.daemon = True
                    client_thread.start()
                except socket.timeout:
                    # Just check running condition
                    continue
                except Exception as e:
                    logger.error(f"Error accepting connection: {str(e)}")
                    time.sleep(0.5)
            except Exception as e:
                logger.error(f"Error in server loop: {str(e)}")
                if not self.running:
                    break
                time.sleep(0.5)

        logger.info("Server thread stopped")

    def _handle_client(self, client):
        """Handle connected client"""
        logger.info("Client handler started")
        client.settimeout(None)  # No timeout
        self._client_socket = client

        # Socket authentication check
        if self._auth_token:
            try:
                auth_msg = recv_message(client, timeout=10.0)
                if auth_msg.get("type") != "auth" or auth_msg.get("token") != self._auth_token:
                    send_message(client, {"status": "error", "message": "Authentication failed"})
                    logger.warning(f"Authentication failed from {client.getpeername()}")
                    return
                send_message(client, {"status": "success", "result": {"authenticated": True}})
                logger.info("Client authenticated successfully")
            except Exception as e:
                logger.warning(f"Auth handshake failed: {e}")
                return

        try:
            while self.running:
                try:
                    command = recv_message(client, timeout=30.0)
                    if command is None:
                        continue

                    def execute_wrapper(cmd=command, cli=client):
                        try:
                            response = self.execute_command(cmd)
                            try:
                                send_message(cli, response)
                            except:
                                logger.warning("Failed to send response - client disconnected")
                        except Exception as e:
                            logger.error(f"Error executing command: {str(e)}")
                            traceback.print_exc()
                            try:
                                send_message(cli, {"status": "error", "message": str(e)})
                            except:
                                pass
                        return None

                    bpy.app.timers.register(execute_wrapper, first_interval=0.0)
                except socket.timeout:
                    continue
                except Exception as e:
                    logger.error(f"Error receiving data: {str(e)}")
                    break
        except Exception as e:
            logger.error(f"Error in client handler: {str(e)}")
        finally:
            try:
                client.close()
            except:
                pass
            logger.info("Client handler stopped")

    def _set_operation_status(self, status):
        """Update the UI status label for long-running operations."""
        def _update(s=status):
            bpy.context.scene.blendermcp_current_operation = s
            return None
        bpy.app.timers.register(_update, first_interval=0.0)

    def execute_command(self, command):
        """Execute a command in the main Blender thread"""
        cmd_type = command.get("type", "unknown")
        try:
            return self._execute_command_internal(command)

        except Exception as e:
            logger.error(f"[{cmd_type}] Error executing command: {str(e)}")
            traceback.print_exc()
            return {"status": "error", "message": f"[{cmd_type}] {str(e)}"}

    def _execute_command_internal(self, command):
        """Internal command execution with proper context"""
        cmd_type = command.get("type")
        params = command.get("params", {})

        if cmd_type == "get_polyhaven_status":
            return {"status": "success", "result": self.get_polyhaven_status()}

        handlers = {
            "get_scene_info": self.get_scene_info,
            "get_object_info": self.get_object_info,
            "get_lights": self.get_lights,
            "get_render_settings": self.get_render_settings,
            "get_collections": self.get_collections,
            "get_mesh_info": self.get_mesh_info,
            "get_animation_info": self.get_animation_info,
            "get_keyframes": self.get_keyframes,
            "get_modifiers": self.get_modifiers,
            "get_material_info": self.get_material_info,
            "get_constraints": self.get_constraints,
            "create_light": self.create_light,
            "set_light_property": self.set_light_property,
            "create_camera": self.create_camera,
            "set_camera_property": self.set_camera_property,
            "set_active_camera": self.set_active_camera,
            "create_collection": self.create_collection,
            "delete_collection": self.delete_collection,
            "move_to_collection": self.move_to_collection,
            "set_collection_visibility": self.set_collection_visibility,
            "create_material": self.create_material,
            "assign_material": self.assign_material,
            "set_material_property": self.set_material_property,
            "set_frame_range": self.set_frame_range,
            "insert_keyframe": self.insert_keyframe,
            "delete_keyframe": self.delete_keyframe,
            "add_modifier": self.add_modifier,
            "remove_modifier": self.remove_modifier,
            "set_modifier_params": self.set_modifier_params,
            "apply_modifier": self.apply_modifier,
            "add_constraint": self.add_constraint,
            "remove_constraint": self.remove_constraint,
            "set_constraint_params": self.set_constraint_params,
            "uv_unwrap": self.uv_unwrap,
            "set_vertex_group": self.set_vertex_group,
            "set_render_settings": self.set_render_settings,
            "render_image": self.render_image,
            "frame_selected": self.frame_selected,
            "get_viewport_screenshot": self.get_viewport_screenshot,
            "execute_code": self.execute_code,
            "duplicate_object": self.duplicate_object,
            "delete_object": self.delete_object,
            "set_object_transform": self.set_object_transform,
            "set_parent": self.set_parent,
            "join_objects": self.join_objects,
            "separate_object": self.separate_object,
            "set_playback": self.set_playback,
            "set_keyframe_interpolation": self.set_keyframe_interpolation,
            "create_shader_node": self.create_shader_node,
            "connect_shader_nodes": self.connect_shader_nodes,
            "blender_undo": self.blender_undo,
            "blender_redo": self.blender_redo,
            "select_objects": self.select_objects,
            "export_scene": self.export_scene,
            "save_scene": self.save_scene,
            "set_auto_save": self.set_auto_save,
            "get_polyhaven_status": self.get_polyhaven_status,
            "get_hyper3d_status": self.get_hyper3d_status,
            "get_sketchfab_status": self.get_sketchfab_status,
            "get_hunyuan3d_status": self.get_hunyuan3d_status,
            "get_meshy_status": self.get_meshy_status,
            "auth": lambda **kw: {"authenticated": True},
        }

        if bpy.context.scene.blendermcp_use_polyhaven:
            polyhaven_handlers = {
                "get_polyhaven_categories": self.get_polyhaven_categories,
                "search_polyhaven_assets": self.search_polyhaven_assets,
                "download_polyhaven_asset": self.download_polyhaven_asset,
                "set_texture": self.set_texture,
            }
            handlers.update(polyhaven_handlers)

        if bpy.context.scene.blendermcp_use_hyper3d:
            polyhaven_handlers = {
                "create_rodin_job": self.create_rodin_job,
                "poll_rodin_job_status": self.poll_rodin_job_status,
                "import_generated_asset": self.import_generated_asset,
            }
            handlers.update(polyhaven_handlers)

        if bpy.context.scene.blendermcp_use_sketchfab:
            sketchfab_handlers = {
                "search_sketchfab_models": self.search_sketchfab_models,
                "get_sketchfab_model_preview": self.get_sketchfab_model_preview,
                "download_sketchfab_model": self.download_sketchfab_model,
            }
            handlers.update(sketchfab_handlers)
        
        if bpy.context.scene.blendermcp_use_hunyuan3d:
            hunyuan_handlers = {
                "create_hunyuan_job": self.create_hunyuan_job,
                "poll_hunyuan_job_status": self.poll_hunyuan_job_status,
                "import_generated_asset_hunyuan": self.import_generated_asset_hunyuan
            }
            handlers.update(hunyuan_handlers)

        if bpy.context.scene.blendermcp_use_meshy:
            meshy_handlers = {
                "create_meshy_job": self.create_meshy_job,
                "poll_meshy_job_status": self.poll_meshy_job_status,
                "import_meshy_asset": self.import_meshy_asset,
            }
            handlers.update(meshy_handlers)

        handler = handlers.get(cmd_type)
        if handler:
            try:
                start_time = time.time()
                logger.info(f"Executing handler for {cmd_type}")
                result = handler(**params)
                duration = time.time() - start_time
                logger.info(f"Handler {cmd_type} completed in {duration:.3f}s")
                return {"status": "success", "result": result}
            except Exception as e:
                logger.error(f"Error in {cmd_type} handler: {str(e)}")
                traceback.print_exc()
                return {"status": "error", "message": str(e)}
        else:
            return {"status": "error", "message": f"Unknown command type: {cmd_type}"}



    def get_scene_info(self, offset=0, limit=50):
        """Get information about the current Blender scene"""
        try:
            logger.info("Getting scene info...")
            scene = bpy.context.scene

            # Simplify the scene info to reduce data size
            scene_info = {
                "name": scene.name,
                "object_count": len(scene.objects),
                "objects": [],
                "materials_count": len(bpy.data.materials),
            }

            render = scene.render
            render_settings = {
                "engine": render.engine,
                "resolution_x": render.resolution_x,
                "resolution_y": render.resolution_y,
                "fps": render.fps,
                "frame_start": scene.frame_start,
                "frame_end": scene.frame_end,
            }
            if render.engine == 'CYCLES':
                try:
                    render_settings["samples"] = scene.cycles.samples
                except AttributeError:
                    pass
            elif render.engine == 'BLENDER_EEVEE' or render.engine == 'BLENDER_EEVEE_NEXT':
                try:
                    render_settings["samples"] = scene.eevee.taa_render_samples
                except AttributeError:
                    pass
            scene_info["render_settings"] = render_settings

            if scene.camera:
                cam = scene.camera
                cam_data = cam.data
                scene_info["active_camera"] = {
                    "name": cam.name,
                    "focal_length": cam_data.lens,
                    "clip_start": cam_data.clip_start,
                    "clip_end": cam_data.clip_end,
                }
            else:
                scene_info["active_camera"] = None

            if scene.world:
                world = scene.world
                world_info = {
                    "name": world.name,
                    "has_hdri": False,
                }
                if world.use_nodes:
                    for node in world.node_tree.nodes:
                        if node.type == 'TEX_ENVIRONMENT':
                            world_info["has_hdri"] = True
                            break
                    # Try to get background color from Background node
                    for node in world.node_tree.nodes:
                        if node.type == 'BACKGROUND':
                            color_input = node.inputs.get('Color')
                            if color_input and not color_input.is_linked:
                                c = color_input.default_value
                                world_info["background_color"] = [round(c[0], 3), round(c[1], 3), round(c[2], 3)]
                            break
                scene_info["world"] = world_info
            else:
                scene_info["world"] = None

            # Collections hierarchy (recursive, capped at 10 levels)
            def _collect_hierarchy(collection, depth=0):
                if depth >= 10:
                    return {"name": collection.name, "objects_count": len(collection.objects), "children": ["..."]}
                children = []
                for child in collection.children:
                    children.append(_collect_hierarchy(child, depth + 1))
                return {
                    "name": collection.name,
                    "objects_count": len(collection.objects),
                    "children": children,
                }

            scene_info["collections"] = _collect_hierarchy(scene.collection)

            lights = []
            for obj in scene.objects:
                if obj.type == 'LIGHT' and len(lights) < 20:
                    light_data = obj.data
                    lights.append({
                        "name": obj.name,
                        "type": light_data.type,
                        "energy": light_data.energy,
                        "color": [round(light_data.color.r, 3), round(light_data.color.g, 3), round(light_data.color.b, 3)],
                    })
            scene_info["lights"] = lights

            all_objects = list(scene.objects)
            paginated_objects = all_objects[offset:offset + limit]
            for obj in paginated_objects:
                obj_info = {
                    "name": obj.name,
                    "type": obj.type,
                    "location": [round(float(obj.location.x), 2),
                                round(float(obj.location.y), 2),
                                round(float(obj.location.z), 2)],
                }
                scene_info["objects"].append(obj_info)

            scene_info["objects_offset"] = offset
            scene_info["objects_limit"] = limit
            scene_info["has_more_objects"] = (offset + limit) < len(all_objects)

            logger.info(f"Scene info collected: {len(scene_info['objects'])} objects")
            return scene_info
        except Exception as e:
            logger.error(f"Error in get_scene_info: {str(e)}")
            traceback.print_exc()
            return {"error": str(e)}

    @staticmethod
    def _get_aabb(obj):
        """ Returns the world-space axis-aligned bounding box (AABB) of an object. """
        if obj.type != 'MESH':
            raise TypeError("Object must be a mesh")

        # Get the bounding box corners in local space
        local_bbox_corners = [mathutils.Vector(corner) for corner in obj.bound_box]

        # Convert to world coordinates
        world_bbox_corners = [obj.matrix_world @ corner for corner in local_bbox_corners]

        # Compute axis-aligned min/max coordinates
        min_corner = mathutils.Vector(map(min, zip(*world_bbox_corners)))
        max_corner = mathutils.Vector(map(max, zip(*world_bbox_corners)))

        return [
            [*min_corner], [*max_corner]
        ]



    def get_object_info(self, name):
        """Get detailed information about a specific object"""
        obj = bpy.data.objects.get(name)
        if not obj:
            raise ValueError(f"Object not found: {name}")

        obj_info = {
            "name": obj.name,
            "type": obj.type,
            "location": [obj.location.x, obj.location.y, obj.location.z],
            "rotation": [obj.rotation_euler.x, obj.rotation_euler.y, obj.rotation_euler.z],
            "scale": [obj.scale.x, obj.scale.y, obj.scale.z],
            "visible": obj.visible_get(),
            "materials": [],
        }

        if obj.type == "MESH":
            bounding_box = self._get_aabb(obj)
            obj_info["world_bounding_box"] = bounding_box

        for slot in obj.material_slots:
            if slot.material:
                obj_info["materials"].append(slot.material.name)

        if obj.type == 'MESH' and obj.data:
            mesh = obj.data
            obj_info["mesh"] = {
                "vertices": len(mesh.vertices),
                "edges": len(mesh.edges),
                "polygons": len(mesh.polygons),
            }

        obj_info["modifiers"] = []
        for mod in obj.modifiers:
            obj_info["modifiers"].append({
                "name": mod.name,
                "type": mod.type,
                "show_viewport": mod.show_viewport,
                "show_render": mod.show_render,
            })

        obj_info["constraints"] = []
        for con in obj.constraints:
            obj_info["constraints"].append({
                "name": con.name,
                "type": con.type,
                "influence": con.influence,
                "mute": con.mute,
            })

        obj_info["parent"] = obj.parent.name if obj.parent else None
        obj_info["children"] = [child.name for child in obj.children]

        custom_props = {}
        for key in obj.keys():
            if key not in ('_RNA_UI', 'cycles'):
                val = obj[key]
                # Convert to JSON-serializable type
                try:
                    json.dumps(val)
                    custom_props[key] = val
                except (TypeError, ValueError):
                    custom_props[key] = str(val)
        obj_info["custom_properties"] = custom_props

        if obj.type == 'LIGHT' and obj.data:
            light = obj.data
            obj_info["light"] = {
                "energy": light.energy,
                "color": [light.color.r, light.color.g, light.color.b],
                "light_type": light.type,
                "shadow_soft_size": light.shadow_soft_size,
            }

        if obj.type == 'CAMERA' and obj.data:
            cam = obj.data
            cam_info = {
                "lens": cam.lens,
                "clip_start": cam.clip_start,
                "clip_end": cam.clip_end,
            }
            if hasattr(cam, 'dof'):
                cam_info["dof_use"] = cam.dof.use_dof
                cam_info["dof_focus_distance"] = cam.dof.focus_distance
                cam_info["dof_aperture_fstop"] = cam.dof.aperture_fstop
            obj_info["camera"] = cam_info

        if obj.type == 'ARMATURE' and obj.data:
            armature = obj.data
            bone_names = [bone.name for bone in armature.bones]
            obj_info["armature"] = {
                "bone_count": len(bone_names),
                "bone_names": bone_names[:50],
            }

        return obj_info

    @staticmethod
    def _serialize_properties(rna_obj):
        """Extract serializable properties from any bpy_struct via bl_rna.properties."""
        result = {}
        for prop in rna_obj.bl_rna.properties:
            if prop.identifier == 'rna_type':
                continue
            try:
                val = getattr(rna_obj, prop.identifier)
                if hasattr(val, 'to_list'):
                    val = val.to_list()
                elif hasattr(val, 'to_dict'):
                    val = val.to_dict()
                json.dumps(val)
                result[prop.identifier] = val
            except (TypeError, AttributeError, ValueError):
                continue
        return result

    @staticmethod
    def _get_principled_bsdf(material):
        """Find Principled BSDF node in material node tree."""
        if not material or not material.use_nodes or not material.node_tree:
            return None
        for node in material.node_tree.nodes:
            if node.type == 'BSDF_PRINCIPLED':
                return node
        return None

    def get_lights(self):
        lights = []
        for obj in bpy.context.scene.objects:
            if obj.type == 'LIGHT' and len(lights) < 50:
                light = obj.data
                lights.append({
                    "name": obj.name,
                    "type": light.type,
                    "energy": light.energy,
                    "color": [light.color.r, light.color.g, light.color.b],
                    "shadow": light.use_shadow,
                    "location": [obj.location.x, obj.location.y, obj.location.z],
                    "rotation": [obj.rotation_euler.x, obj.rotation_euler.y, obj.rotation_euler.z],
                })
        return {"lights": lights, "count": len(lights)}

    def get_render_settings(self):
        scene = bpy.context.scene
        render = scene.render
        result = {
            "engine": render.engine,
            "resolution_x": render.resolution_x,
            "resolution_y": render.resolution_y,
            "resolution_percentage": render.resolution_percentage,
            "fps": scene.render.fps,
            "frame_start": scene.frame_start,
            "frame_end": scene.frame_end,
            "frame_current": scene.frame_current,
            "output_format": render.image_settings.file_format,
            "film_transparent": render.film_transparent,
        }
        if render.engine == 'CYCLES':
            result["samples"] = scene.cycles.samples
            result["preview_samples"] = scene.cycles.preview_samples
            result["denoising"] = scene.cycles.use_denoising
        elif render.engine == 'BLENDER_EEVEE' or render.engine == 'BLENDER_EEVEE_NEXT':
            result["samples"] = scene.eevee.taa_render_samples
            result["denoising"] = False
        camera = scene.camera
        if camera:
            result["active_camera"] = camera.name
        return result

    def get_collections(self):
        def _collect(collection, depth=0):
            if depth > 10:
                return None
            info = {
                "name": collection.name,
                "objects": [obj.name for obj in collection.objects],
                "object_count": len(collection.objects),
                "hide_viewport": collection.hide_viewport,
                "hide_render": collection.hide_render,
                "children": [],
            }
            for child in collection.children:
                child_info = _collect(child, depth + 1)
                if child_info:
                    info["children"].append(child_info)
            return info

        root = bpy.context.scene.collection
        return _collect(root)

    def get_mesh_info(self, name):
        obj = bpy.data.objects.get(name)
        if not obj:
            raise ValueError(f"Object not found: {name}")
        if obj.type != 'MESH':
            raise ValueError(f"Object '{name}' is not a mesh (type: {obj.type})")
        mesh = obj.data
        result = {
            "name": obj.name,
            "vertices": len(mesh.vertices),
            "edges": len(mesh.edges),
            "polygons": len(mesh.polygons),
            "uv_layers": [uv.name for uv in mesh.uv_layers],
            "active_uv": mesh.uv_layers.active.name if mesh.uv_layers.active else None,
            "vertex_groups": [vg.name for vg in obj.vertex_groups],
            "shape_keys": [],
        }
        if mesh.shape_keys:
            for key in mesh.shape_keys.key_blocks:
                result["shape_keys"].append({
                    "name": key.name,
                    "value": key.value,
                    "min": key.slider_min,
                    "max": key.slider_max,
                })
        return result

    def get_animation_info(self):
        scene = bpy.context.scene
        result = {
            "frame_start": scene.frame_start,
            "frame_end": scene.frame_end,
            "frame_current": scene.frame_current,
            "fps": scene.render.fps,
            "keyframed_objects": [],
        }
        seen = set()
        for obj in scene.objects:
            if obj.animation_data and obj.animation_data.action:
                paths = set()
                for fcurve in obj.animation_data.action.fcurves:
                    paths.add(fcurve.data_path)
                if paths and obj.name not in seen:
                    seen.add(obj.name)
                    result["keyframed_objects"].append({
                        "name": obj.name,
                        "animated_properties": list(paths),
                    })
        return result

    def get_keyframes(self, name, property_path=None):
        obj = bpy.data.objects.get(name)
        if not obj:
            raise ValueError(f"Object not found: {name}")
        if not obj.animation_data or not obj.animation_data.action:
            return {"name": name, "fcurves": [], "message": "No animation data"}

        fcurves_data = []
        for i, fcurve in enumerate(obj.animation_data.action.fcurves):
            if i >= 50:
                break
            if property_path and fcurve.data_path != property_path:
                continue
            keyframes = []
            for j, kp in enumerate(fcurve.keyframe_points):
                if j >= 500:
                    keyframes.append({"truncated": True, "total": len(fcurve.keyframe_points)})
                    break
                keyframes.append({
                    "frame": kp.co.x,
                    "value": kp.co.y,
                    "interpolation": kp.interpolation,
                })
            fcurves_data.append({
                "data_path": fcurve.data_path,
                "array_index": fcurve.array_index,
                "keyframes": keyframes,
            })
        return {"name": name, "fcurves": fcurves_data}

    def get_modifiers(self, name):
        obj = bpy.data.objects.get(name)
        if not obj:
            raise ValueError(f"Object not found: {name}")
        modifiers = []
        for mod in obj.modifiers:
            mod_info = {
                "name": mod.name,
                "type": mod.type,
                "show_viewport": mod.show_viewport,
                "show_render": mod.show_render,
            }
            mod_info["params"] = self._serialize_properties(mod)
            modifiers.append(mod_info)
        return {"name": name, "modifiers": modifiers}

    def get_material_info(self, name):
        mat = bpy.data.materials.get(name)
        if not mat:
            obj = bpy.data.objects.get(name)
            if obj and obj.active_material:
                mat = obj.active_material
            else:
                raise ValueError(f"Material not found: {name}")

        result = {"name": mat.name, "use_nodes": mat.use_nodes}
        if not mat.use_nodes or not mat.node_tree:
            return result

        nodes = []
        for i, node in enumerate(mat.node_tree.nodes):
            if i >= 50:
                break
            node_info = {
                "name": node.name,
                "type": node.type,
                "location": [node.location.x, node.location.y],
            }
            inputs = []
            for inp in node.inputs:
                inp_data = {"name": inp.name, "type": inp.type}
                if hasattr(inp, 'default_value'):
                    try:
                        val = inp.default_value
                        if hasattr(val, 'to_list'):
                            val = val.to_list()
                        elif hasattr(val, '__iter__') and not isinstance(val, str):
                            val = list(val)
                        json.dumps(val)
                        inp_data["default_value"] = val
                    except (TypeError, ValueError):
                        pass
                inputs.append(inp_data)
            node_info["inputs"] = inputs
            nodes.append(node_info)

        links = []
        for link in mat.node_tree.links:
            links.append({
                "from_node": link.from_node.name,
                "from_socket": link.from_socket.name,
                "to_node": link.to_node.name,
                "to_socket": link.to_socket.name,
            })

        result["nodes"] = nodes
        result["links"] = links

        bsdf = self._get_principled_bsdf(mat)
        if bsdf:
            bsdf_params = {}
            for inp in bsdf.inputs:
                if hasattr(inp, 'default_value'):
                    try:
                        val = inp.default_value
                        if hasattr(val, 'to_list'):
                            val = val.to_list()
                        elif hasattr(val, '__iter__') and not isinstance(val, str):
                            val = list(val)
                        json.dumps(val)
                        bsdf_params[inp.name] = val
                    except (TypeError, ValueError):
                        pass
            result["principled_bsdf"] = bsdf_params

        return result

    def get_constraints(self, name):
        obj = bpy.data.objects.get(name)
        if not obj:
            raise ValueError(f"Object not found: {name}")
        constraints = []
        for con in obj.constraints:
            con_info = {
                "name": con.name,
                "type": con.type,
                "influence": con.influence,
                "mute": con.mute,
            }
            if hasattr(con, 'target') and con.target:
                con_info["target"] = con.target.name
            con_info["params"] = self._serialize_properties(con)
            constraints.append(con_info)
        return {"name": name, "constraints": constraints}

    # --- Phase 3: Creation / Mutation Handlers ---

    def create_light(self, light_type="POINT", name="Light", location=None, rotation=None, energy=1000.0, color=None, size=0.25, spot_angle=45.0, shadow=True):
        if location is None:
            location = [0, 0, 3]
        if color is None:
            color = [1.0, 1.0, 1.0]
        if rotation is None:
            rotation = [0, 0, 0]

        light_data = bpy.data.lights.new(name=name, type=light_type)
        light_data.energy = energy
        light_data.color = color[:3]
        light_data.use_shadow = shadow

        if light_type == 'SPOT':
            import math
            light_data.spot_size = math.radians(spot_angle)
        if light_type in ('POINT', 'SPOT', 'AREA'):
            light_data.shadow_soft_size = size

        light_obj = bpy.data.objects.new(name=name, object_data=light_data)
        bpy.context.collection.objects.link(light_obj)
        light_obj.location = location
        light_obj.rotation_euler = rotation

        return {
            "name": light_obj.name,
            "type": light_data.type,
            "energy": light_data.energy,
            "location": list(light_obj.location),
        }

    def set_light_property(self, light_name, energy=None, color=None, size=None, shadow=None, location=None, rotation=None):
        obj = bpy.data.objects.get(light_name)
        if not obj or obj.type != 'LIGHT':
            raise ValueError(f"Light not found: {light_name}")
        light = obj.data
        if energy is not None:
            light.energy = energy
        if color is not None:
            light.color = color[:3]
        if size is not None:
            light.shadow_soft_size = size
        if shadow is not None:
            light.use_shadow = shadow
        if location is not None:
            obj.location = location
        if rotation is not None:
            obj.rotation_euler = rotation
        return {"name": obj.name, "updated": True}

    def create_camera(self, name="Camera", location=None, rotation=None, focal_length=50.0, sensor_width=36.0, set_active=True):
        if location is None:
            location = [0, -5, 2]
        if rotation is None:
            import math
            rotation = [math.radians(75), 0, 0]

        cam_data = bpy.data.cameras.new(name=name)
        cam_data.lens = focal_length
        cam_data.sensor_width = sensor_width

        cam_obj = bpy.data.objects.new(name=name, object_data=cam_data)
        bpy.context.collection.objects.link(cam_obj)
        cam_obj.location = location
        cam_obj.rotation_euler = rotation

        if set_active:
            bpy.context.scene.camera = cam_obj

        return {
            "name": cam_obj.name,
            "focal_length": cam_data.lens,
            "location": list(cam_obj.location),
            "is_active": bpy.context.scene.camera == cam_obj,
        }

    def set_camera_property(self, camera_name, focal_length=None, clip_start=None, clip_end=None, dof_enabled=None, dof_focus_distance=None, dof_aperture_fstop=None, location=None, rotation=None):
        obj = bpy.data.objects.get(camera_name)
        if not obj or obj.type != 'CAMERA':
            raise ValueError(f"Camera not found: {camera_name}")
        cam = obj.data
        if focal_length is not None:
            cam.lens = focal_length
        if clip_start is not None:
            cam.clip_start = clip_start
        if clip_end is not None:
            cam.clip_end = clip_end
        if dof_enabled is not None:
            cam.dof.use_dof = dof_enabled
        if dof_focus_distance is not None:
            cam.dof.focus_distance = dof_focus_distance
        if dof_aperture_fstop is not None:
            cam.dof.aperture_fstop = dof_aperture_fstop
        if location is not None:
            obj.location = location
        if rotation is not None:
            obj.rotation_euler = rotation
        return {"name": obj.name, "updated": True}

    def set_active_camera(self, camera_name):
        obj = bpy.data.objects.get(camera_name)
        if not obj or obj.type != 'CAMERA':
            raise ValueError(f"Camera not found: {camera_name}")
        bpy.context.scene.camera = obj
        return {"name": obj.name, "active": True}

    def create_collection(self, name, parent_name=None):
        new_col = bpy.data.collections.new(name)
        if parent_name:
            parent = bpy.data.collections.get(parent_name)
            if not parent:
                raise ValueError(f"Parent collection not found: {parent_name}")
            parent.children.link(new_col)
        else:
            bpy.context.scene.collection.children.link(new_col)
        return {"name": new_col.name, "parent": parent_name}

    def delete_collection(self, collection_name, delete_objects=False):
        col = bpy.data.collections.get(collection_name)
        if not col:
            raise ValueError(f"Collection not found: {collection_name}")
        if delete_objects:
            for obj in list(col.objects):
                bpy.data.objects.remove(obj, do_unlink=True)
        bpy.data.collections.remove(col)
        return {"deleted": collection_name, "objects_deleted": delete_objects}

    def move_to_collection(self, object_name, collection_name, unlink_from_current=True):
        obj = bpy.data.objects.get(object_name)
        if not obj:
            raise ValueError(f"Object not found: {object_name}")
        target = bpy.data.collections.get(collection_name)
        if not target:
            raise ValueError(f"Collection not found: {collection_name}")
        if unlink_from_current:
            for col in obj.users_collection:
                col.objects.unlink(obj)
        target.objects.link(obj)
        return {"object": obj.name, "collection": target.name}

    def set_collection_visibility(self, collection_name, hide_viewport=None, hide_render=None):
        col = bpy.data.collections.get(collection_name)
        if not col:
            raise ValueError(f"Collection not found: {collection_name}")
        if hide_viewport is not None:
            col.hide_viewport = hide_viewport
        if hide_render is not None:
            col.hide_render = hide_render
        return {"name": col.name, "hide_viewport": col.hide_viewport, "hide_render": col.hide_render}

    def create_material(self, name, object_name=None, base_color=None, roughness=0.5, metallic=0.0):
        mat = bpy.data.materials.new(name=name)
        mat.use_nodes = True
        bsdf = self._get_principled_bsdf(mat)
        if bsdf:
            if base_color is not None:
                color = list(base_color)
                while len(color) < 4:
                    color.append(1.0)
                bsdf.inputs['Base Color'].default_value = color[:4]
            bsdf.inputs['Roughness'].default_value = roughness
            bsdf.inputs['Metallic'].default_value = metallic
        if object_name:
            obj = bpy.data.objects.get(object_name)
            if obj and hasattr(obj.data, 'materials'):
                obj.data.materials.append(mat)
        return {"name": mat.name, "assigned_to": object_name}

    def assign_material(self, object_name, material_name, slot_index=None):
        obj = bpy.data.objects.get(object_name)
        if not obj:
            raise ValueError(f"Object not found: {object_name}")
        mat = bpy.data.materials.get(material_name)
        if not mat:
            raise ValueError(f"Material not found: {material_name}")
        if slot_index is not None and slot_index < len(obj.material_slots):
            obj.material_slots[slot_index].material = mat
        else:
            obj.data.materials.append(mat)
        return {"object": obj.name, "material": mat.name}

    def set_material_property(self, material_name, property_name, value):
        mat = bpy.data.materials.get(material_name)
        if not mat:
            raise ValueError(f"Material not found: {material_name}")
        bsdf = self._get_principled_bsdf(mat)
        if not bsdf:
            raise ValueError(f"Material '{material_name}' has no Principled BSDF node")

        name_map = {
            "Subsurface": "Subsurface Weight" if bpy.app.version >= (4, 0, 0) else "Subsurface",
            "Specular": "Specular IOR Level" if bpy.app.version >= (4, 0, 0) else "Specular",
            "Transmission": "Transmission Weight" if bpy.app.version >= (4, 0, 0) else "Transmission",
            "Coat": "Coat Weight" if bpy.app.version >= (4, 0, 0) else "Clearcoat",
            "Sheen": "Sheen Weight" if bpy.app.version >= (4, 0, 0) else "Sheen",
            "Emission": "Emission Color" if bpy.app.version >= (4, 0, 0) else "Emission",
        }
        resolved_name = name_map.get(property_name, property_name)

        if resolved_name not in bsdf.inputs:
            available = [inp.name for inp in bsdf.inputs]
            raise ValueError(f"Property '{resolved_name}' not found. Available: {available}")

        inp = bsdf.inputs[resolved_name]
        if isinstance(value, list) and hasattr(inp.default_value, '__len__'):
            while len(value) < len(inp.default_value):
                value.append(value[-1] if value else 0)
            inp.default_value = value[:len(inp.default_value)]
        else:
            inp.default_value = value
        return {"material": mat.name, "property": resolved_name, "value": value}

    def set_frame_range(self, start_frame=None, end_frame=None, current_frame=None, fps=None):
        scene = bpy.context.scene
        if start_frame is not None:
            scene.frame_start = int(start_frame)
        if end_frame is not None:
            scene.frame_end = int(end_frame)
        if current_frame is not None:
            scene.frame_set(int(current_frame))
        if fps is not None:
            scene.render.fps = int(fps)
        return {
            "frame_start": scene.frame_start,
            "frame_end": scene.frame_end,
            "frame_current": scene.frame_current,
            "fps": scene.render.fps,
        }

    def insert_keyframe(self, object_name, property_path, frame, value=None, index=-1):
        obj = bpy.data.objects.get(object_name)
        if not obj:
            raise ValueError(f"Object not found: {object_name}")

        scene = bpy.context.scene
        original_frame = scene.frame_current
        scene.frame_set(int(frame))

        if value is not None:
            prop = obj.path_resolve(property_path)
            if hasattr(prop, '__len__') and index >= 0:
                prop[index] = value
            elif hasattr(prop, '__len__') and isinstance(value, (list, tuple)):
                for i, v in enumerate(value):
                    prop[i] = v
            else:
                setattr(obj, property_path, value)

        obj.keyframe_insert(data_path=property_path, frame=frame, index=index)
        scene.frame_set(original_frame)
        return {"object": obj.name, "property": property_path, "frame": frame}

    def delete_keyframe(self, object_name, property_path, frame, index=-1):
        obj = bpy.data.objects.get(object_name)
        if not obj:
            raise ValueError(f"Object not found: {object_name}")
        obj.keyframe_delete(data_path=property_path, frame=frame, index=index)
        return {"object": obj.name, "property": property_path, "frame": frame, "deleted": True}

    def add_modifier(self, object_name, modifier_type, name=None, params=None):
        obj = bpy.data.objects.get(object_name)
        if not obj:
            raise ValueError(f"Object not found: {object_name}")
        mod = obj.modifiers.new(name=name or modifier_type, type=modifier_type)
        if params:
            if isinstance(params, str):
                params = json.loads(params)
            for key, val in params.items():
                if hasattr(mod, key):
                    prop = getattr(mod, key)
                    if isinstance(prop, bpy.types.Object.__class__) or key in ('object', 'mirror_object', 'target'):
                        ref = bpy.data.objects.get(val)
                        if ref:
                            setattr(mod, key, ref)
                    else:
                        setattr(mod, key, val)
        return {"object": obj.name, "modifier": mod.name, "type": mod.type}

    def remove_modifier(self, object_name, modifier_name):
        obj = bpy.data.objects.get(object_name)
        if not obj:
            raise ValueError(f"Object not found: {object_name}")
        mod = obj.modifiers.get(modifier_name)
        if not mod:
            raise ValueError(f"Modifier not found: {modifier_name}")
        obj.modifiers.remove(mod)
        return {"object": obj.name, "removed": modifier_name}

    def set_modifier_params(self, object_name, modifier_name, params):
        obj = bpy.data.objects.get(object_name)
        if not obj:
            raise ValueError(f"Object not found: {object_name}")
        mod = obj.modifiers.get(modifier_name)
        if not mod:
            raise ValueError(f"Modifier not found: {modifier_name}")
        if isinstance(params, str):
            params = json.loads(params)
        for key, val in params.items():
            if hasattr(mod, key):
                if key in ('object', 'mirror_object', 'target'):
                    ref = bpy.data.objects.get(val)
                    if ref:
                        setattr(mod, key, ref)
                else:
                    setattr(mod, key, val)
        return {"object": obj.name, "modifier": mod.name, "updated": True}

    def apply_modifier(self, object_name, modifier_name):
        obj = bpy.data.objects.get(object_name)
        if not obj:
            raise ValueError(f"Object not found: {object_name}")
        mod = obj.modifiers.get(modifier_name)
        if not mod:
            raise ValueError(f"Modifier not found: {modifier_name}")

        bpy.context.view_layer.objects.active = obj
        obj.select_set(True)
        if obj.mode != 'OBJECT':
            bpy.ops.object.mode_set(mode='OBJECT')

        bpy.ops.object.modifier_apply(modifier=modifier_name)

        result = {"object": obj.name, "applied": modifier_name}
        if obj.type == 'MESH' and obj.data:
            mesh = obj.data
            result["mesh_stats"] = {
                "vertices": len(mesh.vertices),
                "edges": len(mesh.edges),
                "polygons": len(mesh.polygons),
            }
        return result

    def add_constraint(self, object_name, constraint_type, name=None, target=None, params=None):
        obj = bpy.data.objects.get(object_name)
        if not obj:
            raise ValueError(f"Object not found: {object_name}")
        con = obj.constraints.new(type=constraint_type)
        if name:
            con.name = name
        if target:
            target_obj = bpy.data.objects.get(target)
            if target_obj and hasattr(con, 'target'):
                con.target = target_obj
        if params:
            if isinstance(params, str):
                params = json.loads(params)
            for key, val in params.items():
                if hasattr(con, key):
                    if key == 'target':
                        ref = bpy.data.objects.get(val)
                        if ref:
                            con.target = ref
                    else:
                        setattr(con, key, val)
        return {"object": obj.name, "constraint": con.name, "type": con.type}

    def remove_constraint(self, object_name, constraint_name):
        obj = bpy.data.objects.get(object_name)
        if not obj:
            raise ValueError(f"Object not found: {object_name}")
        con = obj.constraints.get(constraint_name)
        if not con:
            raise ValueError(f"Constraint not found: {constraint_name}")
        obj.constraints.remove(con)
        return {"object": obj.name, "removed": constraint_name}

    def set_constraint_params(self, object_name, constraint_name, params):
        obj = bpy.data.objects.get(object_name)
        if not obj:
            raise ValueError(f"Object not found: {object_name}")
        con = obj.constraints.get(constraint_name)
        if not con:
            raise ValueError(f"Constraint not found: {constraint_name}")
        if isinstance(params, str):
            params = json.loads(params)
        for key, val in params.items():
            if hasattr(con, key):
                if key == 'target':
                    ref = bpy.data.objects.get(val)
                    if ref:
                        con.target = ref
                else:
                    setattr(con, key, val)
        return {"object": obj.name, "constraint": con.name, "updated": True}

    def uv_unwrap(self, object_name, method="SMART_PROJECT", angle_limit=66.0, island_margin=0.001, uv_layer_name=None):
        import math
        obj = bpy.data.objects.get(object_name)
        if not obj or obj.type != 'MESH':
            raise ValueError(f"Mesh object not found: {object_name}")

        if uv_layer_name:
            if uv_layer_name not in obj.data.uv_layers:
                obj.data.uv_layers.new(name=uv_layer_name)
            obj.data.uv_layers.active = obj.data.uv_layers[uv_layer_name]

        bpy.context.view_layer.objects.active = obj
        obj.select_set(True)
        bpy.ops.object.mode_set(mode='EDIT')
        bpy.ops.mesh.select_all(action='SELECT')

        if method == "SMART_PROJECT":
            bpy.ops.uv.smart_project(angle_limit=math.radians(angle_limit), island_margin=island_margin)
        elif method == "CUBE_PROJECT":
            bpy.ops.uv.cube_project()
        elif method == "CYLINDER_PROJECT":
            bpy.ops.uv.cylinder_project()
        elif method == "SPHERE_PROJECT":
            bpy.ops.uv.sphere_project()
        elif method == "UNWRAP":
            bpy.ops.uv.unwrap(method='ANGLE_BASED', margin=island_margin)
        else:
            bpy.ops.object.mode_set(mode='OBJECT')
            raise ValueError(f"Unknown unwrap method: {method}")

        bpy.ops.object.mode_set(mode='OBJECT')
        active_uv = obj.data.uv_layers.active.name if obj.data.uv_layers.active else None
        return {"object": obj.name, "method": method, "uv_layer": active_uv}

    def set_vertex_group(self, object_name, group_name, vertex_indices, weight=1.0, action="REPLACE"):
        obj = bpy.data.objects.get(object_name)
        if not obj or obj.type != 'MESH':
            raise ValueError(f"Mesh object not found: {object_name}")

        vg = obj.vertex_groups.get(group_name)
        if not vg:
            vg = obj.vertex_groups.new(name=group_name)

        if action == "REPLACE":
            vg.add(vertex_indices, weight, 'REPLACE')
        elif action == "ADD":
            vg.add(vertex_indices, weight, 'ADD')
        elif action == "SUBTRACT":
            vg.add(vertex_indices, weight, 'SUBTRACT')
        elif action == "REMOVE":
            vg.remove(vertex_indices)
        else:
            raise ValueError(f"Unknown action: {action}")

        return {"object": obj.name, "group": vg.name, "vertex_count": len(vertex_indices), "action": action}

    def set_render_settings(self, engine=None, resolution_x=None, resolution_y=None, samples=None, denoising=None, output_format=None, film_transparent=None):
        scene = bpy.context.scene
        render = scene.render

        if engine is not None:
            render.engine = engine
        if resolution_x is not None:
            render.resolution_x = resolution_x
        if resolution_y is not None:
            render.resolution_y = resolution_y
        if output_format is not None:
            render.image_settings.file_format = output_format
        if film_transparent is not None:
            render.film_transparent = film_transparent

        if samples is not None:
            if render.engine == 'CYCLES':
                scene.cycles.samples = samples
            elif render.engine in ('BLENDER_EEVEE', 'BLENDER_EEVEE_NEXT'):
                scene.eevee.taa_render_samples = samples

        if denoising is not None and render.engine == 'CYCLES':
            scene.cycles.use_denoising = denoising

        return {
            "engine": render.engine,
            "resolution": [render.resolution_x, render.resolution_y],
            "output_format": render.image_settings.file_format,
        }

    def render_image(self, animation=False, filepath=None):
        scene = bpy.context.scene

        if animation:
            if filepath:
                scene.render.filepath = filepath
            bpy.ops.render.render(animation=True)
            return {"rendered": True, "animation": True, "filepath": scene.render.filepath}
        else:
            tmp_path = filepath
            if not tmp_path:
                tmp_path = os.path.join(tempfile.gettempdir(), f"blender_render_{os.getpid()}.png")

            scene.render.filepath = tmp_path
            bpy.ops.render.render(write_still=True)

            if os.path.exists(tmp_path):
                return {
                    "rendered": True,
                    "animation": False,
                    "filepath": tmp_path,
                    "resolution": [scene.render.resolution_x, scene.render.resolution_y],
                }
            else:
                raise Exception("Render failed - output file not created")

    def frame_selected(self, camera_name=None, object_names=None):
        import math

        if camera_name:
            cam_obj = bpy.data.objects.get(camera_name)
            if not cam_obj or cam_obj.type != 'CAMERA':
                raise ValueError(f"Camera not found: {camera_name}")
        else:
            cam_obj = bpy.context.scene.camera
            if not cam_obj:
                raise ValueError("No active camera in scene")

        if not object_names:
            object_names = [obj.name for obj in bpy.context.selected_objects]
        if not object_names:
            raise ValueError("No objects specified or selected")

        all_min = mathutils.Vector((float('inf'), float('inf'), float('inf')))
        all_max = mathutils.Vector((float('-inf'), float('-inf'), float('-inf')))

        for name in object_names:
            obj = bpy.data.objects.get(name)
            if not obj:
                continue
            for corner in obj.bound_box:
                world_corner = obj.matrix_world @ mathutils.Vector(corner)
                all_min.x = min(all_min.x, world_corner.x)
                all_min.y = min(all_min.y, world_corner.y)
                all_min.z = min(all_min.z, world_corner.z)
                all_max.x = max(all_max.x, world_corner.x)
                all_max.y = max(all_max.y, world_corner.y)
                all_max.z = max(all_max.z, world_corner.z)

        center = (all_min + all_max) / 2
        dims = all_max - all_min
        max_dim = max(dims.x, dims.y, dims.z)

        if max_dim <= 0:
            raise ValueError("Objects have zero bounding box size")

        cam_data = cam_obj.data
        fov = cam_data.angle
        distance = (max_dim / (2 * math.tan(fov / 2))) * 1.5

        direction = mathutils.Vector((0, -1, 0.3)).normalized()
        cam_obj.location = center - direction * distance

        direction_to_target = center - cam_obj.location
        rot = direction_to_target.to_track_quat('-Z', 'Y')
        cam_obj.rotation_euler = rot.to_euler()

        bpy.context.scene.camera = cam_obj

        return {
            "camera": cam_obj.name,
            "target_center": list(center),
            "distance": distance,
            "location": list(cam_obj.location),
        }

    def get_viewport_screenshot(self, max_size=800, filepath=None, format="png"):
        """
        Capture a screenshot of the current 3D viewport and save it to the specified path.

        Parameters:
        - max_size: Maximum size in pixels for the largest dimension of the image
        - filepath: Path where to save the screenshot file
        - format: Image format (png, jpg, etc.)

        Returns success/error status
        """
        try:
            if not filepath:
                return {"error": "No filepath provided"}

            area = None
            for a in bpy.context.screen.areas:
                if a.type == 'VIEW_3D':
                    area = a
                    break

            if not area:
                return {"error": "No 3D viewport found"}

            with bpy.context.temp_override(area=area):
                bpy.ops.screen.screenshot_area(filepath=filepath)

            img = bpy.data.images.load(filepath)
            width, height = img.size

            if max(width, height) > max_size:
                scale = max_size / max(width, height)
                new_width = int(width * scale)
                new_height = int(height * scale)
                img.scale(new_width, new_height)

                img.file_format = format.upper()
                img.save()
                width, height = new_width, new_height

            bpy.data.images.remove(img)

            return {
                "success": True,
                "width": width,
                "height": height,
                "filepath": filepath
            }

        except Exception as e:
            return {"error": str(e)}

    def execute_code(self, code, timeout=30):
        """Execute arbitrary Blender Python code with timeout and output capture"""
        timeout = min(max(timeout, 1), MAX_EXEC_TIMEOUT)

        timeout_event = threading.Event()
        warning_msg = ""

        def _watchdog():
            timeout_event.set()

        timer = threading.Timer(timeout, _watchdog)
        timer.daemon = True

        try:
            namespace = {"bpy": bpy, "_timeout_check": timeout_event}

            stdout_buffer = io.StringIO()
            stderr_buffer = io.StringIO()

            timer.start()
            with redirect_stdout(stdout_buffer), redirect_stderr(stderr_buffer):
                exec(code, namespace)

            if timeout_event.is_set():
                warning_msg = f"Execution may have exceeded {timeout}s timeout"

            stdout_output = stdout_buffer.getvalue()
            stderr_output = stderr_buffer.getvalue()

            if len(stdout_output) > MAX_OUTPUT_SIZE:
                stdout_output = stdout_output[:MAX_OUTPUT_SIZE] + "\n[truncated]"
            if len(stderr_output) > MAX_OUTPUT_SIZE:
                stderr_output = stderr_output[:MAX_OUTPUT_SIZE] + "\n[truncated]"

            return {
                "executed": True,
                "result": stdout_output,
                "stderr": stderr_output,
                "warning": warning_msg,
            }
        except Exception as e:
            raise Exception(f"Code execution error: {str(e)}")
        finally:
            timer.cancel()

    def duplicate_object(self, object_name, new_name=None, linked=False, location=None):
        obj = bpy.data.objects.get(object_name)
        if not obj:
            raise ValueError(f"Object not found: {object_name}")
        if linked:
            new_obj = obj.copy()
        else:
            new_obj = obj.copy()
            if obj.data:
                new_obj.data = obj.data.copy()
        bpy.context.collection.objects.link(new_obj)
        if new_name:
            new_obj.name = new_name
        if location:
            new_obj.location = location
        return {
            "name": new_obj.name,
            "original": obj.name,
            "linked": linked,
            "location": list(new_obj.location),
        }

    def delete_object(self, object_name, delete_data=False):
        obj = bpy.data.objects.get(object_name)
        if not obj:
            raise ValueError(f"Object not found: {object_name}")
        data = obj.data
        for col in obj.users_collection:
            col.objects.unlink(obj)
        bpy.data.objects.remove(obj, do_unlink=True)
        if delete_data and data and data.users == 0:
            data_collections = {
                'MESH': bpy.data.meshes,
                'CURVE': bpy.data.curves,
                'SURFACE': bpy.data.curves,
                'FONT': bpy.data.curves,
                'META': bpy.data.metaballs,
                'ARMATURE': bpy.data.armatures,
                'LATTICE': bpy.data.lattices,
                'LIGHT': bpy.data.lights,
                'CAMERA': bpy.data.cameras,
                'SPEAKER': bpy.data.speakers,
                'GPENCIL': bpy.data.grease_pencils,
            }
            obj_type = type(data).__name__
            for dtype, collection in data_collections.items():
                if data.name in collection:
                    try:
                        collection.remove(data)
                    except:
                        pass
                    break
        return {"deleted": object_name, "data_deleted": delete_data}

    def set_object_transform(self, object_name, location=None, rotation=None, scale=None):
        obj = bpy.data.objects.get(object_name)
        if not obj:
            raise ValueError(f"Object not found: {object_name}")
        if location is not None:
            obj.location = location
        if rotation is not None:
            obj.rotation_euler = rotation
        if scale is not None:
            obj.scale = scale
        return {
            "name": obj.name,
            "location": list(obj.location),
            "rotation": list(obj.rotation_euler),
            "scale": list(obj.scale),
        }

    def set_parent(self, child_name, parent_name=None, keep_transform=True):
        child = bpy.data.objects.get(child_name)
        if not child:
            raise ValueError(f"Child object not found: {child_name}")
        if parent_name is None:
            if keep_transform:
                world_matrix = child.matrix_world.copy()
            child.parent = None
            if keep_transform:
                child.matrix_world = world_matrix
            return {"child": child.name, "parent": None, "keep_transform": keep_transform}
        parent = bpy.data.objects.get(parent_name)
        if not parent:
            raise ValueError(f"Parent object not found: {parent_name}")
        child.parent = parent
        if keep_transform:
            child.matrix_parent_inverse = parent.matrix_world.inverted()
        return {"child": child.name, "parent": parent.name, "keep_transform": keep_transform}

    def join_objects(self, object_names):
        if len(object_names) < 2:
            raise ValueError("At least 2 objects are required to join")
        objects = []
        for name in object_names:
            obj = bpy.data.objects.get(name)
            if not obj:
                raise ValueError(f"Object not found: {name}")
            if obj.type != 'MESH':
                raise ValueError(f"Object '{name}' is not a mesh (type: {obj.type})")
            objects.append(obj)
        bpy.ops.object.select_all(action='DESELECT')
        for obj in objects:
            obj.select_set(True)
        bpy.context.view_layer.objects.active = objects[0]
        bpy.ops.object.join()
        active = bpy.context.view_layer.objects.active
        return {
            "name": active.name,
            "joined_count": len(objects),
            "vertices": len(active.data.vertices),
            "polygons": len(active.data.polygons),
        }

    def separate_object(self, object_name, mode="SELECTED"):
        obj = bpy.data.objects.get(object_name)
        if not obj:
            raise ValueError(f"Object not found: {object_name}")
        if obj.type != 'MESH':
            raise ValueError(f"Object '{object_name}' is not a mesh (type: {obj.type})")
        if mode not in ('SELECTED', 'MATERIAL', 'LOOSE'):
            raise ValueError(f"Invalid mode: {mode}. Must be SELECTED, MATERIAL, or LOOSE")
        bpy.ops.object.select_all(action='DESELECT')
        obj.select_set(True)
        bpy.context.view_layer.objects.active = obj
        bpy.ops.object.mode_set(mode='EDIT')
        bpy.ops.mesh.select_all(action='SELECT')
        bpy.ops.mesh.separate(type=mode)
        bpy.ops.object.mode_set(mode='OBJECT')
        result_names = [o.name for o in bpy.context.selected_objects]
        return {"objects": result_names, "count": len(result_names)}

    def set_playback(self, action, frame=None):
        if action == "play":
            bpy.ops.screen.animation_play()
        elif action in ("stop", "pause"):
            bpy.ops.screen.animation_cancel(restore_frame=False)
        elif action == "jump":
            if frame is None:
                raise ValueError("Frame number is required for 'jump' action")
            bpy.context.scene.frame_set(frame)
        else:
            raise ValueError(f"Invalid action: {action}. Must be play, stop, pause, or jump")
        return {"action": action, "frame": bpy.context.scene.frame_current}

    def set_keyframe_interpolation(self, object_name, interpolation, property_path=None, frame=None):
        valid_types = ('BEZIER', 'LINEAR', 'CONSTANT')
        if interpolation not in valid_types:
            raise ValueError(f"Invalid interpolation: {interpolation}. Must be one of {valid_types}")
        obj = bpy.data.objects.get(object_name)
        if not obj:
            raise ValueError(f"Object not found: {object_name}")
        if not obj.animation_data or not obj.animation_data.action:
            raise ValueError(f"Object '{object_name}' has no animation data")
        action = obj.animation_data.action
        count = 0
        for fcurve in action.fcurves:
            if property_path and fcurve.data_path != property_path:
                continue
            for kp in fcurve.keyframe_points:
                if frame is not None and int(kp.co[0]) != frame:
                    continue
                kp.interpolation = interpolation
                count += 1
        return {"object": obj.name, "interpolation": interpolation, "modified_count": count}

    def create_shader_node(self, material_name, node_type, name=None, location=None, settings=None):
        mat = bpy.data.materials.get(material_name)
        if not mat:
            raise ValueError(f"Material not found: {material_name}")
        if not mat.use_nodes or not mat.node_tree:
            raise ValueError(f"Material '{material_name}' does not use nodes")
        node = mat.node_tree.nodes.new(type=node_type)
        if name:
            node.name = name
            node.label = name
        if location:
            node.location = location
        if settings:
            if isinstance(settings, str):
                import json as _json
                settings = _json.loads(settings)
            for key, value in settings.items():
                if key in node.inputs:
                    node.inputs[key].default_value = value
                elif hasattr(node, key):
                    setattr(node, key, value)
        return {
            "node": node.name,
            "type": node.type,
            "location": list(node.location),
            "inputs": [inp.name for inp in node.inputs],
            "outputs": [out.name for out in node.outputs],
        }

    def connect_shader_nodes(self, material_name, from_node, from_socket, to_node, to_socket):
        mat = bpy.data.materials.get(material_name)
        if not mat:
            raise ValueError(f"Material not found: {material_name}")
        if not mat.use_nodes or not mat.node_tree:
            raise ValueError(f"Material '{material_name}' does not use nodes")
        node_from = mat.node_tree.nodes.get(from_node)
        if not node_from:
            raise ValueError(f"Node not found: {from_node}")
        node_to = mat.node_tree.nodes.get(to_node)
        if not node_to:
            raise ValueError(f"Node not found: {to_node}")
        if isinstance(from_socket, int):
            output_socket = node_from.outputs[from_socket]
        else:
            output_socket = node_from.outputs.get(from_socket)
            if not output_socket:
                raise ValueError(f"Output socket '{from_socket}' not found on node '{from_node}'")
        if isinstance(to_socket, int):
            input_socket = node_to.inputs[to_socket]
        else:
            input_socket = node_to.inputs.get(to_socket)
            if not input_socket:
                raise ValueError(f"Input socket '{to_socket}' not found on node '{to_node}'")
        mat.node_tree.links.new(output_socket, input_socket)
        return {
            "material": mat.name,
            "link": {"from": f"{from_node}.{from_socket}", "to": f"{to_node}.{to_socket}"},
        }

    def blender_undo(self, steps=1):
        for _ in range(steps):
            bpy.ops.ed.undo()
        return {"action": "undo", "steps": steps}

    def blender_redo(self, steps=1):
        for _ in range(steps):
            bpy.ops.ed.redo()
        return {"action": "redo", "steps": steps}

    def select_objects(self, names=None, type=None, material=None, deselect_first=True, active=None):
        if deselect_first:
            bpy.ops.object.select_all(action='DESELECT')
        selected = []
        if names:
            for name in names:
                obj = bpy.data.objects.get(name)
                if obj:
                    obj.select_set(True)
                    selected.append(obj.name)
        if type:
            for obj in bpy.data.objects:
                if obj.type == type:
                    obj.select_set(True)
                    if obj.name not in selected:
                        selected.append(obj.name)
        if material:
            mat = bpy.data.materials.get(material)
            if mat:
                for obj in bpy.data.objects:
                    if hasattr(obj.data, 'materials') and mat.name in [m.name for m in obj.data.materials if m]:
                        obj.select_set(True)
                        if obj.name not in selected:
                            selected.append(obj.name)
        active_obj = None
        if active:
            active_obj = bpy.data.objects.get(active)
            if active_obj:
                bpy.context.view_layer.objects.active = active_obj
        return {
            "selected": selected,
            "active": active_obj.name if active_obj else None,
            "count": len(selected),
        }

    def export_scene(self, filepath, format="GLTF", selected_only=False):
        filepath = os.path.expanduser(filepath)
        parent_dir = os.path.dirname(filepath)
        if parent_dir:
            os.makedirs(parent_dir, exist_ok=True)
        if format in ("GLTF", "GLB"):
            bpy.ops.export_scene.gltf(filepath=filepath, export_format=format, use_selection=selected_only)
        elif format == "FBX":
            bpy.ops.export_scene.fbx(filepath=filepath, use_selection=selected_only)
        elif format == "OBJ":
            bpy.ops.wm.obj_export(filepath=filepath, export_selected_objects=selected_only)
        elif format == "STL":
            bpy.ops.wm.stl_export(filepath=filepath, export_selected_objects=selected_only)
        else:
            raise ValueError(f"Unsupported format: {format}. Must be GLTF, GLB, FBX, OBJ, or STL")
        return {"filepath": filepath, "format": format, "selected_only": selected_only}

    def save_scene(self, filepath=None):
        if filepath:
            filepath = os.path.expanduser(filepath)
            parent_dir = os.path.dirname(filepath)
            if parent_dir:
                os.makedirs(parent_dir, exist_ok=True)
            if not filepath.endswith(".blend"):
                filepath += ".blend"
            bpy.ops.wm.save_as_mainfile(filepath=filepath)
        else:
            if bpy.data.filepath:
                bpy.ops.wm.save_mainfile()
                filepath = bpy.data.filepath
            else:
                return {"error": "No filepath specified and file has never been saved. Provide a filepath."}
        return {"filepath": filepath, "saved": True}

    def set_auto_save(self, enabled, interval_seconds=300, filepath=None):
        if enabled:
            if filepath:
                filepath = os.path.expanduser(filepath)
                if not filepath.endswith(".blend"):
                    filepath += ".blend"

            def auto_save_timer():
                try:
                    save_path = filepath or bpy.data.filepath
                    if save_path:
                        bpy.ops.wm.save_as_mainfile(filepath=save_path)
                        logger.info(f"Auto-saved to {save_path}")
                    else:
                        logger.warning("Auto-save skipped: no filepath set")
                except Exception as e:
                    logger.error(f"Auto-save failed: {e}")
                return interval_seconds

            if hasattr(self, '_auto_save_timer') and self._auto_save_timer:
                try:
                    bpy.app.timers.unregister(self._auto_save_timer)
                except:
                    pass

            self._auto_save_timer = auto_save_timer
            bpy.app.timers.register(auto_save_timer, first_interval=interval_seconds)
            return {"enabled": True, "interval_seconds": interval_seconds, "filepath": filepath or bpy.data.filepath or "not set"}
        else:
            if hasattr(self, '_auto_save_timer') and self._auto_save_timer:
                try:
                    bpy.app.timers.unregister(self._auto_save_timer)
                except:
                    pass
                self._auto_save_timer = None
            return {"enabled": False}

    def get_polyhaven_categories(self, asset_type):
        """Get categories for a specific asset type from Polyhaven"""
        try:
            if asset_type not in ["hdris", "textures", "models", "all"]:
                return {"error": f"Invalid asset type: {asset_type}. Must be one of: hdris, textures, models, all"}

            response = requests.get(f"https://api.polyhaven.com/categories/{asset_type}", headers=REQ_HEADERS)
            if response.status_code == 200:
                return {"categories": response.json()}
            else:
                return {"error": f"API request failed with status code {response.status_code}"}
        except Exception as e:
            return {"error": str(e)}

    def search_polyhaven_assets(self, asset_type=None, categories=None, offset=0, limit=20):
        """Search for assets from Polyhaven with optional filtering and pagination"""
        try:
            url = "https://api.polyhaven.com/assets"
            params = {}

            if asset_type and asset_type != "all":
                if asset_type not in ["hdris", "textures", "models"]:
                    return {"error": f"Invalid asset type: {asset_type}. Must be one of: hdris, textures, models, all"}
                params["type"] = asset_type

            if categories:
                params["categories"] = categories

            response = requests.get(url, params=params, headers=REQ_HEADERS)
            if response.status_code == 200:
                assets = response.json()
                total_count = len(assets)
                # Use offset/limit pagination
                paginated_items = list(assets.items())[offset:offset + limit]
                limited_assets = dict(paginated_items)
                has_more = (offset + limit) < total_count

                return {
                    "assets": limited_assets,
                    "total_count": total_count,
                    "returned_count": len(limited_assets),
                    "offset": offset,
                    "limit": limit,
                    "has_more": has_more,
                }
            else:
                return {"error": f"API request failed with status code {response.status_code}"}
        except Exception as e:
            return {"error": str(e)}

    def download_polyhaven_asset(self, asset_id, asset_type, resolution="1k", file_format=None):
        self._set_operation_status(f"Downloading PolyHaven asset: {asset_id}...")
        try:
            files_response = requests.get(f"https://api.polyhaven.com/files/{asset_id}", headers=REQ_HEADERS)
            if files_response.status_code != 200:
                return {"error": f"Failed to get asset files: {files_response.status_code}"}

            files_data = files_response.json()

            if asset_type == "hdris":
                if not file_format:
                    file_format = "hdr"  # Default format for HDRIs

                if "hdri" in files_data and resolution in files_data["hdri"] and file_format in files_data["hdri"][resolution]:
                    file_info = files_data["hdri"][resolution][file_format]
                    file_url = file_info["url"]

                    # For HDRIs, we need to save to a temporary file first
                    # since Blender can't properly load HDR data directly from memory
                    with tempfile.NamedTemporaryFile(suffix=f".{file_format}", delete=False) as tmp_file:
                        response = requests.get(file_url, headers=REQ_HEADERS)
                        if response.status_code != 200:
                            return {"error": f"Failed to download HDRI: {response.status_code}"}

                        tmp_file.write(response.content)
                        tmp_path = tmp_file.name

                    try:
                        if not bpy.data.worlds:
                            bpy.data.worlds.new("World")

                        world = bpy.data.worlds[0]
                        world.use_nodes = True
                        node_tree = world.node_tree

                        for node in node_tree.nodes:
                            node_tree.nodes.remove(node)

                        tex_coord = node_tree.nodes.new(type='ShaderNodeTexCoord')
                        tex_coord.location = (-800, 0)

                        mapping = node_tree.nodes.new(type='ShaderNodeMapping')
                        mapping.location = (-600, 0)

                        env_tex = node_tree.nodes.new(type='ShaderNodeTexEnvironment')
                        env_tex.location = (-400, 0)
                        env_tex.image = bpy.data.images.load(tmp_path)

                        # Use a color space that exists in all Blender versions
                        if file_format.lower() == 'exr':
                            # Try to use Linear color space for EXR files
                            try:
                                env_tex.image.colorspace_settings.name = 'Linear'
                            except:
                                # Fallback to Non-Color if Linear isn't available
                                env_tex.image.colorspace_settings.name = 'Non-Color'
                        else:  # hdr
                            # For HDR files, try these options in order
                            for color_space in ['Linear', 'Linear Rec.709', 'Non-Color']:
                                try:
                                    env_tex.image.colorspace_settings.name = color_space
                                    break  # Stop if we successfully set a color space
                                except:
                                    continue

                        background = node_tree.nodes.new(type='ShaderNodeBackground')
                        background.location = (-200, 0)

                        output = node_tree.nodes.new(type='ShaderNodeOutputWorld')
                        output.location = (0, 0)

                        node_tree.links.new(tex_coord.outputs['Generated'], mapping.inputs['Vector'])
                        node_tree.links.new(mapping.outputs['Vector'], env_tex.inputs['Vector'])
                        node_tree.links.new(env_tex.outputs['Color'], background.inputs['Color'])
                        node_tree.links.new(background.outputs['Background'], output.inputs['Surface'])

                        bpy.context.scene.world = world

                        return {
                            "success": True,
                            "message": f"HDRI {asset_id} imported successfully",
                            "image_name": env_tex.image.name
                        }
                    except Exception as e:
                        return {"error": f"Failed to set up HDRI in Blender: {str(e)}"}
                    finally:
                        try:
                            os.unlink(tmp_path)
                        except Exception:
                            pass
                else:
                    return {"error": f"Requested resolution or format not available for this HDRI"}

            elif asset_type == "textures":
                if not file_format:
                    file_format = "jpg"  # Default format for textures

                downloaded_maps = {}

                try:
                    for map_type in files_data:
                        if map_type not in ["blend", "gltf"]:  # Skip non-texture files
                            if resolution in files_data[map_type] and file_format in files_data[map_type][resolution]:
                                file_info = files_data[map_type][resolution][file_format]
                                file_url = file_info["url"]

                                with tempfile.NamedTemporaryFile(suffix=f".{file_format}", delete=False) as tmp_file:
                                    response = requests.get(file_url, headers=REQ_HEADERS)
                                    if response.status_code == 200:
                                        tmp_file.write(response.content)
                                        tmp_path = tmp_file.name

                                        image = bpy.data.images.load(tmp_path)
                                        image.name = f"{asset_id}_{map_type}.{file_format}"

                                        image.pack()

                                        if map_type in ['color', 'diffuse', 'albedo']:
                                            try:
                                                image.colorspace_settings.name = 'sRGB'
                                            except:
                                                pass
                                        else:
                                            try:
                                                image.colorspace_settings.name = 'Non-Color'
                                            except:
                                                pass

                                        downloaded_maps[map_type] = image

                                        try:
                                            os.unlink(tmp_path)
                                        except:
                                            pass

                    if not downloaded_maps:
                        return {"error": f"No texture maps found for the requested resolution and format"}

                    mat = bpy.data.materials.new(name=asset_id)
                    mat.use_nodes = True
                    nodes = mat.node_tree.nodes
                    links = mat.node_tree.links

                    for node in nodes:
                        nodes.remove(node)

                    output = nodes.new(type='ShaderNodeOutputMaterial')
                    output.location = (300, 0)

                    principled = nodes.new(type='ShaderNodeBsdfPrincipled')
                    principled.location = (0, 0)
                    links.new(principled.outputs[0], output.inputs[0])

                    tex_coord = nodes.new(type='ShaderNodeTexCoord')
                    tex_coord.location = (-800, 0)

                    mapping = nodes.new(type='ShaderNodeMapping')
                    mapping.location = (-600, 0)
                    mapping.vector_type = 'TEXTURE'  # Changed from default 'POINT' to 'TEXTURE'
                    links.new(tex_coord.outputs['UV'], mapping.inputs['Vector'])

                    x_pos = -400
                    y_pos = 300

                    for map_type, image in downloaded_maps.items():
                        tex_node = nodes.new(type='ShaderNodeTexImage')
                        tex_node.location = (x_pos, y_pos)
                        tex_node.image = image

                        if map_type.lower() in ['color', 'diffuse', 'albedo']:
                            try:
                                tex_node.image.colorspace_settings.name = 'sRGB'
                            except:
                                pass
                        else:
                            try:
                                tex_node.image.colorspace_settings.name = 'Non-Color'
                            except:
                                pass

                        links.new(mapping.outputs['Vector'], tex_node.inputs['Vector'])

                        if map_type.lower() in ['color', 'diffuse', 'albedo']:
                            links.new(tex_node.outputs['Color'], principled.inputs['Base Color'])
                        elif map_type.lower() in ['roughness', 'rough']:
                            links.new(tex_node.outputs['Color'], principled.inputs['Roughness'])
                        elif map_type.lower() in ['metallic', 'metalness', 'metal']:
                            links.new(tex_node.outputs['Color'], principled.inputs['Metallic'])
                        elif map_type.lower() in ['normal', 'nor']:
                            normal_map = nodes.new(type='ShaderNodeNormalMap')
                            normal_map.location = (x_pos + 200, y_pos)
                            links.new(tex_node.outputs['Color'], normal_map.inputs['Color'])
                            links.new(normal_map.outputs['Normal'], principled.inputs['Normal'])
                        elif map_type in ['displacement', 'disp', 'height']:
                            disp_node = nodes.new(type='ShaderNodeDisplacement')
                            disp_node.location = (x_pos + 200, y_pos - 200)
                            links.new(tex_node.outputs['Color'], disp_node.inputs['Height'])
                            links.new(disp_node.outputs['Displacement'], output.inputs['Displacement'])

                        y_pos -= 250

                    return {
                        "success": True,
                        "message": f"Texture {asset_id} imported as material",
                        "material": mat.name,
                        "maps": list(downloaded_maps.keys())
                    }

                except Exception as e:
                    return {"error": f"Failed to process textures: {str(e)}"}

            elif asset_type == "models":
                if not file_format:
                    file_format = "gltf"

                if file_format in files_data and resolution in files_data[file_format]:
                    file_info = files_data[file_format][resolution][file_format]
                    file_url = file_info["url"]

                    temp_dir = tempfile.mkdtemp()
                    main_file_path = ""

                    try:
                        main_file_name = file_url.split("/")[-1]
                        main_file_path = os.path.join(temp_dir, main_file_name)

                        response = requests.get(file_url, headers=REQ_HEADERS)
                        if response.status_code != 200:
                            return {"error": f"Failed to download model: {response.status_code}"}

                        with open(main_file_path, "wb") as f:
                            f.write(response.content)

                        if "include" in file_info and file_info["include"]:
                            for include_path, include_info in file_info["include"].items():
                                include_url = include_info["url"]

                                include_file_path = os.path.join(temp_dir, include_path)
                                os.makedirs(os.path.dirname(include_file_path), exist_ok=True)

                                include_response = requests.get(include_url, headers=REQ_HEADERS)
                                if include_response.status_code == 200:
                                    with open(include_file_path, "wb") as f:
                                        f.write(include_response.content)
                                else:
                                    print(f"Failed to download included file: {include_path}")

                        if file_format == "gltf" or file_format == "glb":
                            bpy.ops.import_scene.gltf(filepath=main_file_path)
                        elif file_format == "fbx":
                            bpy.ops.import_scene.fbx(filepath=main_file_path)
                        elif file_format == "obj":
                            bpy.ops.import_scene.obj(filepath=main_file_path)
                        elif file_format == "blend":
                            with bpy.data.libraries.load(main_file_path, link=False) as (data_from, data_to):
                                data_to.objects = data_from.objects

                            for obj in data_to.objects:
                                if obj is not None:
                                    bpy.context.collection.objects.link(obj)
                        else:
                            return {"error": f"Unsupported model format: {file_format}"}

                        imported_objects = [obj.name for obj in bpy.context.selected_objects]

                        return {
                            "success": True,
                            "message": f"Model {asset_id} imported successfully",
                            "imported_objects": imported_objects
                        }
                    except Exception as e:
                        return {"error": f"Failed to import model: {str(e)}"}
                    finally:
                        with suppress(Exception):
                            shutil.rmtree(temp_dir)
                else:
                    return {"error": f"Requested format or resolution not available for this model"}

            else:
                return {"error": f"Unsupported asset type: {asset_type}"}

        except Exception as e:
            return {"error": f"Failed to download asset: {str(e)}"}
        finally:
            self._set_operation_status("")

    def set_texture(self, object_name, texture_id):
        """Apply a previously downloaded Polyhaven texture to an object by creating a new material"""
        try:
            obj = bpy.data.objects.get(object_name)
            if not obj:
                return {"error": f"Object not found: {object_name}"}

            if not hasattr(obj, 'data') or not hasattr(obj.data, 'materials'):
                return {"error": f"Object {object_name} cannot accept materials"}

            texture_images = {}
            for img in bpy.data.images:
                if img.name.startswith(texture_id + "_"):
                    map_type = img.name.split('_')[-1].split('.')[0]

                    img.reload()

                    if map_type.lower() in ['color', 'diffuse', 'albedo']:
                        try:
                            img.colorspace_settings.name = 'sRGB'
                        except:
                            pass
                    else:
                        try:
                            img.colorspace_settings.name = 'Non-Color'
                        except:
                            pass

                    if not img.packed_file:
                        img.pack()

                    texture_images[map_type] = img

            if not texture_images:
                return {"error": f"No texture images found for: {texture_id}. Please download the texture first."}

            new_mat_name = f"{texture_id}_material_{object_name}"

            existing_mat = bpy.data.materials.get(new_mat_name)
            if existing_mat:
                bpy.data.materials.remove(existing_mat)

            new_mat = bpy.data.materials.new(name=new_mat_name)
            new_mat.use_nodes = True

            nodes = new_mat.node_tree.nodes
            links = new_mat.node_tree.links

            nodes.clear()

            output = nodes.new(type='ShaderNodeOutputMaterial')
            output.location = (600, 0)

            principled = nodes.new(type='ShaderNodeBsdfPrincipled')
            principled.location = (300, 0)
            links.new(principled.outputs[0], output.inputs[0])

            tex_coord = nodes.new(type='ShaderNodeTexCoord')
            tex_coord.location = (-800, 0)

            mapping = nodes.new(type='ShaderNodeMapping')
            mapping.location = (-600, 0)
            mapping.vector_type = 'TEXTURE'  # Changed from default 'POINT' to 'TEXTURE'
            links.new(tex_coord.outputs['UV'], mapping.inputs['Vector'])

            x_pos = -400
            y_pos = 300

            for map_type, image in texture_images.items():
                tex_node = nodes.new(type='ShaderNodeTexImage')
                tex_node.location = (x_pos, y_pos)
                tex_node.image = image

                if map_type.lower() in ['color', 'diffuse', 'albedo']:
                    try:
                        tex_node.image.colorspace_settings.name = 'sRGB'
                    except:
                        pass
                else:
                    try:
                        tex_node.image.colorspace_settings.name = 'Non-Color'
                    except:
                        pass

                links.new(mapping.outputs['Vector'], tex_node.inputs['Vector'])

                if map_type.lower() in ['color', 'diffuse', 'albedo']:
                    links.new(tex_node.outputs['Color'], principled.inputs['Base Color'])
                elif map_type.lower() in ['roughness', 'rough']:
                    links.new(tex_node.outputs['Color'], principled.inputs['Roughness'])
                elif map_type.lower() in ['metallic', 'metalness', 'metal']:
                    links.new(tex_node.outputs['Color'], principled.inputs['Metallic'])
                elif map_type.lower() in ['normal', 'nor', 'dx', 'gl']:
                    normal_map = nodes.new(type='ShaderNodeNormalMap')
                    normal_map.location = (x_pos + 200, y_pos)
                    links.new(tex_node.outputs['Color'], normal_map.inputs['Color'])
                    links.new(normal_map.outputs['Normal'], principled.inputs['Normal'])
                elif map_type.lower() in ['displacement', 'disp', 'height']:
                    disp_node = nodes.new(type='ShaderNodeDisplacement')
                    disp_node.location = (x_pos + 200, y_pos - 200)
                    disp_node.inputs['Scale'].default_value = 0.1  # Reduce displacement strength
                    links.new(tex_node.outputs['Color'], disp_node.inputs['Height'])
                    links.new(disp_node.outputs['Displacement'], output.inputs['Displacement'])

                y_pos -= 250

            texture_nodes = {}

            for node in nodes:
                if node.type == 'TEX_IMAGE' and node.image:
                    for map_type, image in texture_images.items():
                        if node.image == image:
                            texture_nodes[map_type] = node
                            break

            for map_name in ['color', 'diffuse', 'albedo']:
                if map_name in texture_nodes:
                    links.new(texture_nodes[map_name].outputs['Color'], principled.inputs['Base Color'])
                    break

            for map_name in ['roughness', 'rough']:
                if map_name in texture_nodes:
                    links.new(texture_nodes[map_name].outputs['Color'], principled.inputs['Roughness'])
                    break

            for map_name in ['metallic', 'metalness', 'metal']:
                if map_name in texture_nodes:
                    links.new(texture_nodes[map_name].outputs['Color'], principled.inputs['Metallic'])
                    break

            for map_name in ['gl', 'dx', 'nor']:
                if map_name in texture_nodes:
                    normal_map_node = nodes.new(type='ShaderNodeNormalMap')
                    normal_map_node.location = (100, 100)
                    links.new(texture_nodes[map_name].outputs['Color'], normal_map_node.inputs['Color'])
                    links.new(normal_map_node.outputs['Normal'], principled.inputs['Normal'])
                    break

            for map_name in ['displacement', 'disp', 'height']:
                if map_name in texture_nodes:
                    disp_node = nodes.new(type='ShaderNodeDisplacement')
                    disp_node.location = (300, -200)
                    disp_node.inputs['Scale'].default_value = 0.1  # Reduce displacement strength
                    links.new(texture_nodes[map_name].outputs['Color'], disp_node.inputs['Height'])
                    links.new(disp_node.outputs['Displacement'], output.inputs['Displacement'])
                    break

            # Handle ARM texture (Ambient Occlusion, Roughness, Metallic)
            if 'arm' in texture_nodes:
                separate_rgb = nodes.new(type='ShaderNodeSeparateRGB')
                separate_rgb.location = (-200, -100)
                links.new(texture_nodes['arm'].outputs['Color'], separate_rgb.inputs['Image'])

                # Connect Roughness (G) if no dedicated roughness map
                if not any(map_name in texture_nodes for map_name in ['roughness', 'rough']):
                    links.new(separate_rgb.outputs['G'], principled.inputs['Roughness'])

                # Connect Metallic (B) if no dedicated metallic map
                if not any(map_name in texture_nodes for map_name in ['metallic', 'metalness', 'metal']):
                    links.new(separate_rgb.outputs['B'], principled.inputs['Metallic'])

                # For AO (R channel), multiply with base color if we have one
                base_color_node = None
                for map_name in ['color', 'diffuse', 'albedo']:
                    if map_name in texture_nodes:
                        base_color_node = texture_nodes[map_name]
                        break

                if base_color_node:
                    mix_node = nodes.new(type='ShaderNodeMixRGB')
                    mix_node.location = (100, 200)
                    mix_node.blend_type = 'MULTIPLY'
                    mix_node.inputs['Fac'].default_value = 0.8  # 80% influence

                    for link in base_color_node.outputs['Color'].links:
                        if link.to_socket == principled.inputs['Base Color']:
                            links.remove(link)

                    links.new(base_color_node.outputs['Color'], mix_node.inputs[1])
                    links.new(separate_rgb.outputs['R'], mix_node.inputs[2])
                    links.new(mix_node.outputs['Color'], principled.inputs['Base Color'])

            if 'ao' in texture_nodes:
                base_color_node = None
                for map_name in ['color', 'diffuse', 'albedo']:
                    if map_name in texture_nodes:
                        base_color_node = texture_nodes[map_name]
                        break

                if base_color_node:
                    mix_node = nodes.new(type='ShaderNodeMixRGB')
                    mix_node.location = (100, 200)
                    mix_node.blend_type = 'MULTIPLY'
                    mix_node.inputs['Fac'].default_value = 0.8  # 80% influence

                    for link in base_color_node.outputs['Color'].links:
                        if link.to_socket == principled.inputs['Base Color']:
                            links.remove(link)

                    links.new(base_color_node.outputs['Color'], mix_node.inputs[1])
                    links.new(texture_nodes['ao'].outputs['Color'], mix_node.inputs[2])
                    links.new(mix_node.outputs['Color'], principled.inputs['Base Color'])

            while len(obj.data.materials) > 0:
                obj.data.materials.pop(index=0)

            obj.data.materials.append(new_mat)

            bpy.context.view_layer.objects.active = obj
            obj.select_set(True)

            bpy.context.view_layer.update()

            texture_maps = list(texture_images.keys())

            material_info = {
                "name": new_mat.name,
                "has_nodes": new_mat.use_nodes,
                "node_count": len(new_mat.node_tree.nodes),
                "texture_nodes": []
            }

            for node in new_mat.node_tree.nodes:
                if node.type == 'TEX_IMAGE' and node.image:
                    connections = []
                    for output in node.outputs:
                        for link in output.links:
                            connections.append(f"{output.name} → {link.to_node.name}.{link.to_socket.name}")

                    material_info["texture_nodes"].append({
                        "name": node.name,
                        "image": node.image.name,
                        "colorspace": node.image.colorspace_settings.name,
                        "connections": connections
                    })

            return {
                "success": True,
                "message": f"Created new material and applied texture {texture_id} to {object_name}",
                "material": new_mat.name,
                "maps": texture_maps,
                "material_info": material_info
            }

        except Exception as e:
            print(f"Error in set_texture: {str(e)}")
            traceback.print_exc()
            return {"error": f"Failed to apply texture: {str(e)}"}

    def get_polyhaven_status(self):
        """Get the current status of PolyHaven integration"""
        enabled = bpy.context.scene.blendermcp_use_polyhaven
        if enabled:
            return {"enabled": True, "message": "PolyHaven integration is enabled and ready to use."}
        else:
            return {
                "enabled": False,
                "message": """PolyHaven integration is currently disabled. To enable it:
                            1. In the 3D Viewport, find the BlenderMCP panel in the sidebar (press N if hidden)
                            2. Check the 'Use assets from Poly Haven' checkbox
                            3. Restart the connection to Claude"""
        }

    #region Hyper3D
    def get_hyper3d_status(self):
        """Get the current status of Hyper3D Rodin integration"""
        enabled = bpy.context.scene.blendermcp_use_hyper3d
        if enabled:
            if not bpy.context.scene.blendermcp_hyper3d_api_key:
                return {
                    "enabled": False,
                    "message": """Hyper3D Rodin integration is currently enabled, but API key is not given. To enable it:
                                1. In the 3D Viewport, find the BlenderMCP panel in the sidebar (press N if hidden)
                                2. Keep the 'Use Hyper3D Rodin 3D model generation' checkbox checked
                                3. Choose the right plaform and fill in the API Key
                                4. Restart the connection to Claude"""
                }
            mode = bpy.context.scene.blendermcp_hyper3d_mode
            message = f"Hyper3D Rodin integration is enabled and ready to use. Mode: {mode}. " + \
                f"Key type: {'private' if bpy.context.scene.blendermcp_hyper3d_api_key != RODIN_FREE_TRIAL_KEY else 'free_trial'}"
            return {
                "enabled": True,
                "message": message
            }
        else:
            return {
                "enabled": False,
                "message": """Hyper3D Rodin integration is currently disabled. To enable it:
                            1. In the 3D Viewport, find the BlenderMCP panel in the sidebar (press N if hidden)
                            2. Check the 'Use Hyper3D Rodin 3D model generation' checkbox
                            3. Restart the connection to Claude"""
            }

    def create_rodin_job(self, *args, **kwargs):
        self._set_operation_status("Creating Hyper3D Rodin job...")
        try:
            match bpy.context.scene.blendermcp_hyper3d_mode:
                case "MAIN_SITE":
                    return self.create_rodin_job_main_site(*args, **kwargs)
                case "FAL_AI":
                    return self.create_rodin_job_fal_ai(*args, **kwargs)
                case _:
                    return f"Error: Unknown Hyper3D Rodin mode!"
        finally:
            self._set_operation_status("")

    def create_rodin_job_main_site(
            self,
            text_prompt: str=None,
            images: list[tuple[str, str]]=None,
            bbox_condition=None
        ):
        try:
            if images is None:
                images = []
            """Call Rodin API, get the job uuid and subscription key"""
            files = [
                *[("images", (f"{i:04d}{img_suffix}", img)) for i, (img_suffix, img) in enumerate(images)],
                ("tier", (None, "Sketch")),
                ("mesh_mode", (None, "Raw")),
            ]
            if text_prompt:
                files.append(("prompt", (None, text_prompt)))
            if bbox_condition:
                files.append(("bbox_condition", (None, json.dumps(bbox_condition))))
            response = requests.post(
                "https://hyperhuman.deemos.com/api/v2/rodin",
                headers={
                    "Authorization": f"Bearer {bpy.context.scene.blendermcp_hyper3d_api_key}",
                },
                files=files
            )
            data = response.json()
            return data
        except Exception as e:
            return {"error": str(e)}

    def create_rodin_job_fal_ai(
            self,
            text_prompt: str=None,
            images: list[tuple[str, str]]=None,
            bbox_condition=None
        ):
        try:
            req_data = {
                "tier": "Sketch",
            }
            if images:
                req_data["input_image_urls"] = images
            if text_prompt:
                req_data["prompt"] = text_prompt
            if bbox_condition:
                req_data["bbox_condition"] = bbox_condition
            response = requests.post(
                "https://queue.fal.run/fal-ai/hyper3d/rodin",
                headers={
                    "Authorization": f"Key {bpy.context.scene.blendermcp_hyper3d_api_key}",
                    "Content-Type": "application/json",
                },
                json=req_data
            )
            data = response.json()
            return data
        except Exception as e:
            return {"error": str(e)}

    def poll_rodin_job_status(self, *args, **kwargs):
        match bpy.context.scene.blendermcp_hyper3d_mode:
            case "MAIN_SITE":
                return self.poll_rodin_job_status_main_site(*args, **kwargs)
            case "FAL_AI":
                return self.poll_rodin_job_status_fal_ai(*args, **kwargs)
            case _:
                return f"Error: Unknown Hyper3D Rodin mode!"

    def poll_rodin_job_status_main_site(self, subscription_key: str):
        """Call the job status API to get the job status"""
        response = requests.post(
            "https://hyperhuman.deemos.com/api/v2/status",
            headers={
                "Authorization": f"Bearer {bpy.context.scene.blendermcp_hyper3d_api_key}",
            },
            json={
                "subscription_key": subscription_key,
            },
        )
        data = response.json()
        return {
            "status_list": [i["status"] for i in data["jobs"]]
        }

    def poll_rodin_job_status_fal_ai(self, request_id: str):
        """Call the job status API to get the job status"""
        response = requests.get(
            f"https://queue.fal.run/fal-ai/hyper3d/requests/{request_id}/status",
            headers={
                "Authorization": f"KEY {bpy.context.scene.blendermcp_hyper3d_api_key}",
            },
        )
        data = response.json()
        return data

    @staticmethod
    def _clean_imported_glb(filepath, mesh_name=None):
        existing_objects = set(bpy.data.objects)

        bpy.ops.import_scene.gltf(filepath=filepath)

        bpy.context.view_layer.update()

        imported_objects = list(set(bpy.data.objects) - existing_objects)

        if not imported_objects:
            print("Error: No objects were imported.")
            return

        mesh_obj = None

        if len(imported_objects) == 1 and imported_objects[0].type == 'MESH':
            mesh_obj = imported_objects[0]
        else:
            if len(imported_objects) == 2:
                empty_objs = [i for i in imported_objects if i.type == "EMPTY"]
                if len(empty_objs) != 1:
                    print("Error: Expected an empty node with one mesh child or a single mesh object.")
                    return
                parent_obj = empty_objs.pop()
                if len(parent_obj.children) == 1:
                    potential_mesh = parent_obj.children[0]
                    if potential_mesh.type == 'MESH':
                        potential_mesh.parent = None
                        bpy.data.objects.remove(parent_obj)

                        mesh_obj = potential_mesh
                    else:
                        print("Error: Child is not a mesh object.")
                        return
                else:
                    print("Error: Expected an empty node with one mesh child or a single mesh object.")
                    return
            else:
                print("Error: Expected an empty node with one mesh child or a single mesh object.")
                return

        try:
            if mesh_obj and mesh_obj.name is not None and mesh_name:
                mesh_obj.name = mesh_name
                if mesh_obj.data.name is not None:
                    mesh_obj.data.name = mesh_name
        except Exception:
            pass  # Renaming is best-effort

        return mesh_obj

    def import_generated_asset(self, *args, **kwargs):
        match bpy.context.scene.blendermcp_hyper3d_mode:
            case "MAIN_SITE":
                return self.import_generated_asset_main_site(*args, **kwargs)
            case "FAL_AI":
                return self.import_generated_asset_fal_ai(*args, **kwargs)
            case _:
                return f"Error: Unknown Hyper3D Rodin mode!"

    def import_generated_asset_main_site(self, task_uuid: str, name: str):
        """Fetch the generated asset, import into blender"""
        response = requests.post(
            "https://hyperhuman.deemos.com/api/v2/download",
            headers={
                "Authorization": f"Bearer {bpy.context.scene.blendermcp_hyper3d_api_key}",
            },
            json={
                'task_uuid': task_uuid
            }
        )
        data_ = response.json()
        temp_file = None
        for i in data_["list"]:
            if i["name"].endswith(".glb"):
                temp_file = tempfile.NamedTemporaryFile(
                    delete=False,
                    prefix=task_uuid,
                    suffix=".glb",
                )

                try:
                    response = requests.get(i["url"], stream=True)
                    response.raise_for_status()

                    for chunk in response.iter_content(chunk_size=8192):
                        temp_file.write(chunk)

                    temp_file.close()

                except Exception as e:
                    temp_file.close()
                    os.unlink(temp_file.name)
                    return {"succeed": False, "error": str(e)}

                break
        else:
            return {"succeed": False, "error": "Generation failed. Please first make sure that all jobs of the task are done and then try again later."}

        try:
            obj = self._clean_imported_glb(
                filepath=temp_file.name,
                mesh_name=name
            )
            result = {
                "name": obj.name,
                "type": obj.type,
                "location": [obj.location.x, obj.location.y, obj.location.z],
                "rotation": [obj.rotation_euler.x, obj.rotation_euler.y, obj.rotation_euler.z],
                "scale": [obj.scale.x, obj.scale.y, obj.scale.z],
            }

            if obj.type == "MESH":
                bounding_box = self._get_aabb(obj)
                result["world_bounding_box"] = bounding_box

            return {
                "succeed": True, **result
            }
        except Exception as e:
            return {"succeed": False, "error": str(e)}
        finally:
            try:
                os.unlink(temp_file.name)
            except Exception:
                pass

    def import_generated_asset_fal_ai(self, request_id: str, name: str):
        """Fetch the generated asset, import into blender"""
        response = requests.get(
            f"https://queue.fal.run/fal-ai/hyper3d/requests/{request_id}",
            headers={
                "Authorization": f"Key {bpy.context.scene.blendermcp_hyper3d_api_key}",
            }
        )
        data_ = response.json()
        temp_file = None

        temp_file = tempfile.NamedTemporaryFile(
            delete=False,
            prefix=request_id,
            suffix=".glb",
        )

        try:
            response = requests.get(data_["model_mesh"]["url"], stream=True)
            response.raise_for_status()

            for chunk in response.iter_content(chunk_size=8192):
                temp_file.write(chunk)

            temp_file.close()

        except Exception as e:
            temp_file.close()
            os.unlink(temp_file.name)
            return {"succeed": False, "error": str(e)}

        try:
            obj = self._clean_imported_glb(
                filepath=temp_file.name,
                mesh_name=name
            )
            result = {
                "name": obj.name,
                "type": obj.type,
                "location": [obj.location.x, obj.location.y, obj.location.z],
                "rotation": [obj.rotation_euler.x, obj.rotation_euler.y, obj.rotation_euler.z],
                "scale": [obj.scale.x, obj.scale.y, obj.scale.z],
            }

            if obj.type == "MESH":
                bounding_box = self._get_aabb(obj)
                result["world_bounding_box"] = bounding_box

            return {
                "succeed": True, **result
            }
        except Exception as e:
            return {"succeed": False, "error": str(e)}
        finally:
            try:
                os.unlink(temp_file.name)
            except Exception:
                pass
    #endregion

    #region Sketchfab API
    def get_sketchfab_status(self):
        """Get the current status of Sketchfab integration"""
        enabled = bpy.context.scene.blendermcp_use_sketchfab
        api_key = bpy.context.scene.blendermcp_sketchfab_api_key

        if api_key:
            try:
                headers = {
                    "Authorization": f"Token {api_key}"
                }

                response = requests.get(
                    "https://api.sketchfab.com/v3/me",
                    headers=headers,
                    timeout=30  # Add timeout of 30 seconds
                )

                if response.status_code == 200:
                    user_data = response.json()
                    username = user_data.get("username", "Unknown user")
                    return {
                        "enabled": True,
                        "message": f"Sketchfab integration is enabled and ready to use. Logged in as: {username}"
                    }
                else:
                    return {
                        "enabled": False,
                        "message": f"Sketchfab API key seems invalid. Status code: {response.status_code}"
                    }
            except requests.exceptions.Timeout:
                return {
                    "enabled": False,
                    "message": "Timeout connecting to Sketchfab API. Check your internet connection."
                }
            except Exception as e:
                return {
                    "enabled": False,
                    "message": f"Error testing Sketchfab API key: {str(e)}"
                }

        if enabled and api_key:
            return {"enabled": True, "message": "Sketchfab integration is enabled and ready to use."}
        elif enabled and not api_key:
            return {
                "enabled": False,
                "message": """Sketchfab integration is currently enabled, but API key is not given. To enable it:
                            1. In the 3D Viewport, find the BlenderMCP panel in the sidebar (press N if hidden)
                            2. Keep the 'Use Sketchfab' checkbox checked
                            3. Enter your Sketchfab API Key
                            4. Restart the connection to Claude"""
            }
        else:
            return {
                "enabled": False,
                "message": """Sketchfab integration is currently disabled. To enable it:
                            1. In the 3D Viewport, find the BlenderMCP panel in the sidebar (press N if hidden)
                            2. Check the 'Use assets from Sketchfab' checkbox
                            3. Enter your Sketchfab API Key
                            4. Restart the connection to Claude"""
            }

    def search_sketchfab_models(self, query, categories=None, count=20, downloadable=True):
        """Search for models on Sketchfab based on query and optional filters"""
        try:
            api_key = bpy.context.scene.blendermcp_sketchfab_api_key
            if not api_key:
                return {"error": "Sketchfab API key is not configured"}

            params = {
                "type": "models",
                "q": query,
                "count": count,
                "downloadable": downloadable,
                "archives_flavours": False
            }

            if categories:
                params["categories"] = categories

            headers = {
                "Authorization": f"Token {api_key}"
            }


            response = requests.get(
                "https://api.sketchfab.com/v3/search",
                headers=headers,
                params=params,
                timeout=30  # Add timeout of 30 seconds
            )

            if response.status_code == 401:
                return {"error": "Authentication failed (401). Check your API key."}

            if response.status_code != 200:
                return {"error": f"API request failed with status code {response.status_code}"}

            response_data = response.json()

            if response_data is None:
                return {"error": "Received empty response from Sketchfab API"}

            results = response_data.get("results", [])
            if not isinstance(results, list):
                return {"error": f"Unexpected response format from Sketchfab API: {response_data}"}

            response_data["has_next"] = bool(response_data.get("next"))
            return response_data

        except requests.exceptions.Timeout:
            return {"error": "Request timed out. Check your internet connection."}
        except json.JSONDecodeError as e:
            return {"error": f"Invalid JSON response from Sketchfab API: {str(e)}"}
        except Exception as e:
            import traceback
            traceback.print_exc()
            return {"error": str(e)}

    def get_sketchfab_model_preview(self, uid):
        """Get thumbnail preview image of a Sketchfab model by its UID"""
        try:
            import base64
            
            api_key = bpy.context.scene.blendermcp_sketchfab_api_key
            if not api_key:
                return {"error": "Sketchfab API key is not configured"}

            headers = {"Authorization": f"Token {api_key}"}
            
            response = requests.get(
                f"https://api.sketchfab.com/v3/models/{uid}",
                headers=headers,
                timeout=30
            )
            
            if response.status_code == 401:
                return {"error": "Authentication failed (401). Check your API key."}
            
            if response.status_code == 404:
                return {"error": f"Model not found: {uid}"}
            
            if response.status_code != 200:
                return {"error": f"Failed to get model info: {response.status_code}"}
            
            data = response.json()
            thumbnails = data.get("thumbnails", {}).get("images", [])
            
            if not thumbnails:
                return {"error": "No thumbnail available for this model"}
            
            # Prefer medium size ~640px thumbnail
            selected_thumbnail = None
            for thumb in thumbnails:
                width = thumb.get("width", 0)
                if 400 <= width <= 800:
                    selected_thumbnail = thumb
                    break
            
            if not selected_thumbnail:
                selected_thumbnail = thumbnails[0]
            
            thumbnail_url = selected_thumbnail.get("url")
            if not thumbnail_url:
                return {"error": "Thumbnail URL not found"}
            
            img_response = requests.get(thumbnail_url, timeout=30)
            if img_response.status_code != 200:
                return {"error": f"Failed to download thumbnail: {img_response.status_code}"}
            
            image_data = base64.b64encode(img_response.content).decode('ascii')

            content_type = img_response.headers.get("Content-Type", "")
            if "png" in content_type or thumbnail_url.endswith(".png"):
                img_format = "png"
            else:
                img_format = "jpeg"
            
            model_name = data.get("name", "Unknown")
            author = data.get("user", {}).get("username", "Unknown")
            
            return {
                "success": True,
                "image_data": image_data,
                "format": img_format,
                "model_name": model_name,
                "author": author,
                "uid": uid,
                "thumbnail_width": selected_thumbnail.get("width"),
                "thumbnail_height": selected_thumbnail.get("height")
            }
            
        except requests.exceptions.Timeout:
            return {"error": "Request timed out. Check your internet connection."}
        except Exception as e:
            import traceback
            traceback.print_exc()
            return {"error": f"Failed to get model preview: {str(e)}"}

    def download_sketchfab_model(self, uid, normalize_size=False, target_size=1.0):
        """Download a model from Sketchfab by its UID

        Parameters:
        - uid: The unique identifier of the Sketchfab model
        - normalize_size: If True, scale the model so its largest dimension equals target_size
        - target_size: The target size in Blender units (meters) for the largest dimension
        """
        self._set_operation_status(f"Downloading Sketchfab model: {uid}...")
        try:
            api_key = bpy.context.scene.blendermcp_sketchfab_api_key
            if not api_key:
                return {"error": "Sketchfab API key is not configured"}

            headers = {
                "Authorization": f"Token {api_key}"
            }

            download_endpoint = f"https://api.sketchfab.com/v3/models/{uid}/download"

            response = requests.get(
                download_endpoint,
                headers=headers,
                timeout=30  # Add timeout of 30 seconds
            )

            if response.status_code == 401:
                return {"error": "Authentication failed (401). Check your API key."}

            if response.status_code != 200:
                return {"error": f"Download request failed with status code {response.status_code}"}

            data = response.json()

            if data is None:
                return {"error": "Received empty response from Sketchfab API for download request"}

            gltf_data = data.get("gltf")
            if not gltf_data:
                return {"error": "No gltf download URL available for this model. Response: " + str(data)}

            download_url = gltf_data.get("url")
            if not download_url:
                return {"error": "No download URL available for this model. Make sure the model is downloadable and you have access."}

            model_response = requests.get(download_url, timeout=60)

            if model_response.status_code != 200:
                return {"error": f"Model download failed with status code {model_response.status_code}"}

            temp_dir = tempfile.mkdtemp()
            zip_file_path = os.path.join(temp_dir, f"{uid}.zip")

            with open(zip_file_path, "wb") as f:
                f.write(model_response.content)

            # Zip slip prevention: verify all entries stay within temp_dir before extracting
            with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
                for file_info in zip_ref.infolist():
                    file_path = file_info.filename

                    # Handles both / and \ in zip entries
                    target_path = os.path.join(temp_dir, os.path.normpath(file_path))

                    abs_temp_dir = os.path.abspath(temp_dir)
                    abs_target_path = os.path.abspath(target_path)

                    if not abs_target_path.startswith(abs_temp_dir):
                        with suppress(Exception):
                            shutil.rmtree(temp_dir)
                        return {"error": "Security issue: Zip contains files with path traversal attempt"}

                    if ".." in file_path:
                        with suppress(Exception):
                            shutil.rmtree(temp_dir)
                        return {"error": "Security issue: Zip contains files with directory traversal sequence"}

                zip_ref.extractall(temp_dir)

            gltf_files = [f for f in os.listdir(temp_dir) if f.endswith('.gltf') or f.endswith('.glb')]

            if not gltf_files:
                with suppress(Exception):
                    shutil.rmtree(temp_dir)
                return {"error": "No glTF file found in the downloaded model"}

            main_file = os.path.join(temp_dir, gltf_files[0])

            bpy.ops.import_scene.gltf(filepath=main_file)

            imported_objects = list(bpy.context.selected_objects)
            imported_object_names = [obj.name for obj in imported_objects]

            with suppress(Exception):
                shutil.rmtree(temp_dir)

            root_objects = [obj for obj in imported_objects if obj.parent is None]

            def get_all_mesh_children(obj):
                """Recursively collect all mesh objects in the hierarchy"""
                meshes = []
                if obj.type == 'MESH':
                    meshes.append(obj)
                for child in obj.children:
                    meshes.extend(get_all_mesh_children(child))
                return meshes

            all_meshes = []
            for obj in root_objects:
                all_meshes.extend(get_all_mesh_children(obj))
            
            if all_meshes:
                all_min = mathutils.Vector((float('inf'), float('inf'), float('inf')))
                all_max = mathutils.Vector((float('-inf'), float('-inf'), float('-inf')))
                
                for mesh_obj in all_meshes:
                    for corner in mesh_obj.bound_box:
                        world_corner = mesh_obj.matrix_world @ mathutils.Vector(corner)
                        all_min.x = min(all_min.x, world_corner.x)
                        all_min.y = min(all_min.y, world_corner.y)
                        all_min.z = min(all_min.z, world_corner.z)
                        all_max.x = max(all_max.x, world_corner.x)
                        all_max.y = max(all_max.y, world_corner.y)
                        all_max.z = max(all_max.z, world_corner.z)

                dimensions = [
                    all_max.x - all_min.x,
                    all_max.y - all_min.y,
                    all_max.z - all_min.z
                ]
                max_dimension = max(dimensions)
                
                scale_applied = 1.0
                if normalize_size and max_dimension > 0:
                    scale_factor = target_size / max_dimension
                    scale_applied = scale_factor
                    
                    # ✅ Only apply scale to ROOT objects (not children!)
                    # Child objects inherit parent's scale through matrix_world
                    for root in root_objects:
                        root.scale = (
                            root.scale.x * scale_factor,
                            root.scale.y * scale_factor,
                            root.scale.z * scale_factor
                        )
                    
                    bpy.context.view_layer.update()

                    # Recalculate bounding box after scaling
                    all_min = mathutils.Vector((float('inf'), float('inf'), float('inf')))
                    all_max = mathutils.Vector((float('-inf'), float('-inf'), float('-inf')))
                    
                    for mesh_obj in all_meshes:
                        for corner in mesh_obj.bound_box:
                            world_corner = mesh_obj.matrix_world @ mathutils.Vector(corner)
                            all_min.x = min(all_min.x, world_corner.x)
                            all_min.y = min(all_min.y, world_corner.y)
                            all_min.z = min(all_min.z, world_corner.z)
                            all_max.x = max(all_max.x, world_corner.x)
                            all_max.y = max(all_max.y, world_corner.y)
                            all_max.z = max(all_max.z, world_corner.z)
                    
                    dimensions = [
                        all_max.x - all_min.x,
                        all_max.y - all_min.y,
                        all_max.z - all_min.z
                    ]
                
                world_bounding_box = [[all_min.x, all_min.y, all_min.z], [all_max.x, all_max.y, all_max.z]]
            else:
                world_bounding_box = None
                dimensions = None
                scale_applied = 1.0

            result = {
                "success": True,
                "message": "Model imported successfully",
                "imported_objects": imported_object_names
            }
            
            if world_bounding_box:
                result["world_bounding_box"] = world_bounding_box
            if dimensions:
                result["dimensions"] = [round(d, 4) for d in dimensions]
            if normalize_size:
                result["scale_applied"] = round(scale_applied, 6)
                result["normalized"] = True
            
            return result

        except requests.exceptions.Timeout:
            return {"error": "Request timed out. Check your internet connection and try again with a simpler model."}
        except json.JSONDecodeError as e:
            return {"error": f"Invalid JSON response from Sketchfab API: {str(e)}"}
        except Exception as e:
            import traceback
            traceback.print_exc()
            return {"error": f"Failed to download model: {str(e)}"}
        finally:
            self._set_operation_status("")
    #endregion

    #region Hunyuan3D
    def get_hunyuan3d_status(self):
        """Get the current status of Hunyuan3D integration"""
        enabled = bpy.context.scene.blendermcp_use_hunyuan3d
        hunyuan3d_mode = bpy.context.scene.blendermcp_hunyuan3d_mode
        if enabled:
            match hunyuan3d_mode:
                case "OFFICIAL_API":
                    if not bpy.context.scene.blendermcp_hunyuan3d_secret_id or not bpy.context.scene.blendermcp_hunyuan3d_secret_key:
                        return {
                            "enabled": False, 
                            "mode": hunyuan3d_mode, 
                            "message": """Hunyuan3D integration is currently enabled, but SecretId or SecretKey is not given. To enable it:
                                1. In the 3D Viewport, find the BlenderMCP panel in the sidebar (press N if hidden)
                                2. Keep the 'Use Tencent Hunyuan 3D model generation' checkbox checked
                                3. Choose the right platform and fill in the SecretId and SecretKey
                                4. Restart the connection to Claude"""
                        }
                case "LOCAL_API":
                    if not bpy.context.scene.blendermcp_hunyuan3d_api_url:
                        return {
                            "enabled": False, 
                            "mode": hunyuan3d_mode, 
                            "message": """Hunyuan3D integration is currently enabled, but API URL  is not given. To enable it:
                                1. In the 3D Viewport, find the BlenderMCP panel in the sidebar (press N if hidden)
                                2. Keep the 'Use Tencent Hunyuan 3D model generation' checkbox checked
                                3. Choose the right platform and fill in the API URL
                                4. Restart the connection to Claude"""
                        }
                case _:
                    return {
                        "enabled": False, 
                        "message": "Hunyuan3D integration is enabled and mode is not supported."
                    }
            return {
                "enabled": True, 
                "mode": hunyuan3d_mode,
                "message": "Hunyuan3D integration is enabled and ready to use."
            }
        return {
            "enabled": False, 
            "message": """Hunyuan3D integration is currently disabled. To enable it:
                        1. In the 3D Viewport, find the BlenderMCP panel in the sidebar (press N if hidden)
                        2. Check the 'Use Tencent Hunyuan 3D model generation' checkbox
                        3. Restart the connection to Claude"""
        }
    
    @staticmethod
    def get_tencent_cloud_sign_headers(
        method: str,
        path: str,
        headParams: dict,
        data: dict,
        service: str,
        region: str,
        secret_id: str,
        secret_key: str,
        host: str = None
    ):
        """Generate the signature header required for Tencent Cloud API requests headers"""
        timestamp = int(time.time())
        date = datetime.utcfromtimestamp(timestamp).strftime("%Y-%m-%d")

        if not host:
            host = f"{service}.tencentcloudapi.com"

        endpoint = f"https://{host}"

        payload_str = json.dumps(data)
        
        # ************* Step 1: Concatenate the canonical request string *************
        canonical_uri = path
        canonical_querystring = ""
        ct = "application/json; charset=utf-8"
        canonical_headers = f"content-type:{ct}\nhost:{host}\nx-tc-action:{headParams.get('Action', '').lower()}\n"
        signed_headers = "content-type;host;x-tc-action"
        hashed_request_payload = hashlib.sha256(payload_str.encode("utf-8")).hexdigest()
        
        canonical_request = (method + "\n" +
                            canonical_uri + "\n" +
                            canonical_querystring + "\n" +
                            canonical_headers + "\n" +
                            signed_headers + "\n" +
                            hashed_request_payload)

        # ************* Step 2: Construct the reception signature string *************
        credential_scope = f"{date}/{service}/tc3_request"
        hashed_canonical_request = hashlib.sha256(canonical_request.encode("utf-8")).hexdigest()
        string_to_sign = ("TC3-HMAC-SHA256" + "\n" +
                        str(timestamp) + "\n" +
                        credential_scope + "\n" +
                        hashed_canonical_request)

        # ************* Step 3: Calculate the signature *************
        def sign(key, msg):
            return hmac.new(key, msg.encode("utf-8"), hashlib.sha256).digest()

        secret_date = sign(("TC3" + secret_key).encode("utf-8"), date)
        secret_service = sign(secret_date, service)
        secret_signing = sign(secret_service, "tc3_request")
        signature = hmac.new(
            secret_signing, 
            string_to_sign.encode("utf-8"), 
            hashlib.sha256
        ).hexdigest()

        # ************* Step 4: Connect Authorization *************
        authorization = ("TC3-HMAC-SHA256" + " " +
                        "Credential=" + secret_id + "/" + credential_scope + ", " +
                        "SignedHeaders=" + signed_headers + ", " +
                        "Signature=" + signature)

        headers = {
            "Authorization": authorization,
            "Content-Type": "application/json; charset=utf-8",
            "Host": host,
            "X-TC-Action": headParams.get("Action", ""),
            "X-TC-Timestamp": str(timestamp),
            "X-TC-Version": headParams.get("Version", ""),
            "X-TC-Region": region
        }

        return headers, endpoint

    def create_hunyuan_job(self, *args, **kwargs):
        self._set_operation_status("Creating Hunyuan3D job...")
        try:
            match bpy.context.scene.blendermcp_hunyuan3d_mode:
                case "OFFICIAL_API":
                    return self.create_hunyuan_job_main_site(*args, **kwargs)
                case "LOCAL_API":
                    return self.create_hunyuan_job_local_site(*args, **kwargs)
                case _:
                    return f"Error: Unknown Hunyuan3D mode!"
        finally:
            self._set_operation_status("")

    def create_hunyuan_job_main_site(
        self,
        text_prompt: str = None,
        image: str = None
    ):
        try:
            secret_id = bpy.context.scene.blendermcp_hunyuan3d_secret_id
            secret_key = bpy.context.scene.blendermcp_hunyuan3d_secret_key

            if not secret_id or not secret_key:
                return {"error": "SecretId or SecretKey is not given"}

            if not text_prompt and not image:
                return {"error": "Prompt or Image is required"}
            if text_prompt and image:
                return {"error": "Prompt and Image cannot be provided simultaneously"}

            service = "hunyuan"
            action = "SubmitHunyuanTo3DJob"
            version = "2023-09-01"
            region = "ap-guangzhou"

            headParams={
                "Action": action,
                "Version": version,
                "Region": region,
            }

            data = {
                "Num": 1  # The current API limit is only 1
            }

            if text_prompt:
                if len(text_prompt) > 200:
                    return {"error": "Prompt exceeds 200 characters limit"}
                data["Prompt"] = text_prompt

            if image:
                if re.match(r'^https?://', image, re.IGNORECASE) is not None:
                    data["ImageUrl"] = image
                else:
                    try:
                        with open(image, "rb") as f:
                            image_base64 = base64.b64encode(f.read()).decode("ascii")
                        data["ImageBase64"] = image_base64
                    except Exception as e:
                        return {"error": f"Image encoding failed: {str(e)}"}

            headers, endpoint = self.get_tencent_cloud_sign_headers("POST", "/", headParams, data, service, region, secret_id, secret_key)

            response = requests.post(
                endpoint,
                headers = headers,
                data = json.dumps(data)
            )

            if response.status_code == 200:
                return response.json()
            return {
                "error": f"API request failed with status {response.status_code}: {response}"
            }
        except Exception as e:
            return {"error": str(e)}

    def create_hunyuan_job_local_site(
        self,
        text_prompt: str = None,
        image: str = None):
        try:
            base_url = bpy.context.scene.blendermcp_hunyuan3d_api_url.rstrip('/')
            octree_resolution = bpy.context.scene.blendermcp_hunyuan3d_octree_resolution
            num_inference_steps = bpy.context.scene.blendermcp_hunyuan3d_num_inference_steps
            guidance_scale = bpy.context.scene.blendermcp_hunyuan3d_guidance_scale
            texture = bpy.context.scene.blendermcp_hunyuan3d_texture

            if not base_url:
                return {"error": "API URL is not given"}
            if not text_prompt and not image:
                return {"error": "Prompt or Image is required"}

            data = {
                "octree_resolution": octree_resolution,
                "num_inference_steps": num_inference_steps,
                "guidance_scale": guidance_scale,
                "texture": texture,
            }

            if text_prompt:
                data["text"] = text_prompt

            if image:
                if re.match(r'^https?://', image, re.IGNORECASE) is not None:
                    try:
                        resImg = requests.get(image)
                        resImg.raise_for_status()
                        image_base64 = base64.b64encode(resImg.content).decode("ascii")
                        data["image"] = image_base64
                    except Exception as e:
                        return {"error": f"Failed to download or encode image: {str(e)}"} 
                else:
                    try:
                        with open(image, "rb") as f:
                            image_base64 = base64.b64encode(f.read()).decode("ascii")
                        data["image"] = image_base64
                    except Exception as e:
                        return {"error": f"Image encoding failed: {str(e)}"}

            response = requests.post(
                f"{base_url}/generate",
                json = data,
            )

            if response.status_code != 200:
                return {
                    "error": f"Generation failed: {response.text}"
                }
        
            with tempfile.NamedTemporaryFile(delete=False, suffix=".glb") as temp_file:
                temp_file.write(response.content)
                temp_file_name = temp_file.name

            def import_handler():
                bpy.ops.import_scene.gltf(filepath=temp_file_name)
                os.unlink(temp_file.name)
                return None
            
            bpy.app.timers.register(import_handler)

            return {
                "status": "DONE",
                "message": "Generation and Import glb succeeded"
            }
        except Exception as e:
            print(f"An error occurred: {e}")
            return {"error": str(e)}
        
    
    def poll_hunyuan_job_status(self, *args, **kwargs):
        return self.poll_hunyuan_job_status_ai(*args, **kwargs)
    
    def poll_hunyuan_job_status_ai(self, job_id: str):
        """Call the job status API to get the job status"""
        print(job_id)
        try:
            secret_id = bpy.context.scene.blendermcp_hunyuan3d_secret_id
            secret_key = bpy.context.scene.blendermcp_hunyuan3d_secret_key

            if not secret_id or not secret_key:
                return {"error": "SecretId or SecretKey is not given"}
            if not job_id:
                return {"error": "JobId is required"}
            
            service = "hunyuan"
            action = "QueryHunyuanTo3DJob"
            version = "2023-09-01"
            region = "ap-guangzhou"

            headParams={
                "Action": action,
                "Version": version,
                "Region": region,
            }

            clean_job_id = job_id.removeprefix("job_")
            data = {
                "JobId": clean_job_id
            }

            headers, endpoint = self.get_tencent_cloud_sign_headers("POST", "/", headParams, data, service, region, secret_id, secret_key)

            response = requests.post(
                endpoint,
                headers=headers,
                data=json.dumps(data)
            )

            if response.status_code == 200:
                return response.json()
            return {
                "error": f"API request failed with status {response.status_code}: {response}"
            }
        except Exception as e:
            return {"error": str(e)}

    def import_generated_asset_hunyuan(self, *args, **kwargs):
        return self.import_generated_asset_hunyuan_ai(*args, **kwargs)
            
    def import_generated_asset_hunyuan_ai(self, name: str , zip_file_url: str):
        if not zip_file_url:
            return {"error": "Zip file not found"}
        
        if not re.match(r'^https?://', zip_file_url, re.IGNORECASE):
            return {"error": "Invalid URL format. Must start with http:// or https://"}
        
        temp_dir = tempfile.mkdtemp(prefix="tencent_obj_")
        zip_file_path = osp.join(temp_dir, "model.zip")
        obj_file_path = osp.join(temp_dir, "model.obj")
        mtl_file_path = osp.join(temp_dir, "model.mtl")

        try:
            zip_response = requests.get(zip_file_url, stream=True)
            zip_response.raise_for_status()
            with open(zip_file_path, "wb") as f:
                for chunk in zip_response.iter_content(chunk_size=8192):
                    f.write(chunk)

            with zipfile.ZipFile(zip_file_path, "r") as zip_ref:
                zip_ref.extractall(temp_dir)

            for file in os.listdir(temp_dir):
                if file.endswith(".obj"):
                    obj_file_path = osp.join(temp_dir, file)

            if not osp.exists(obj_file_path):
                return {"succeed": False, "error": "OBJ file not found after extraction"}

            if bpy.app.version>=(4, 0, 0):
                bpy.ops.wm.obj_import(filepath=obj_file_path)
            else:
                bpy.ops.import_scene.obj(filepath=obj_file_path)

            imported_objs = [obj for obj in bpy.context.selected_objects if obj.type == 'MESH']
            if not imported_objs:
                return {"succeed": False, "error": "No mesh objects imported"}

            obj = imported_objs[0]
            if name:
                obj.name = name

            result = {
                "name": obj.name,
                "type": obj.type,
                "location": [obj.location.x, obj.location.y, obj.location.z],
                "rotation": [obj.rotation_euler.x, obj.rotation_euler.y, obj.rotation_euler.z],
                "scale": [obj.scale.x, obj.scale.y, obj.scale.z],
            }

            if obj.type == "MESH":
                bounding_box = self._get_aabb(obj)
                result["world_bounding_box"] = bounding_box

            return {"succeed": True, **result}
        except Exception as e:
            return {"succeed": False, "error": str(e)}
        finally:
            try:
                if os.path.exists(zip_file_path):
                    os.remove(zip_file_path) 
                if os.path.exists(obj_file_path):
                    os.remove(obj_file_path)
            except Exception as e:
                print(f"Failed to clean up temporary directory {temp_dir}: {e}")
    #endregion

    #region Meshy
    def get_meshy_status(self):
        """Get the current status of Meshy integration"""
        enabled = bpy.context.scene.blendermcp_use_meshy
        if enabled:
            api_key = bpy.context.scene.blendermcp_meshy_api_key
            if not api_key:
                return {
                    "enabled": False,
                    "message": """Meshy integration is currently enabled, but the API key is not set. To enable it:
                        1. In the 3D Viewport, find the BlenderMCP panel in the sidebar (press N if hidden)
                        2. Keep the 'Use Meshy AI 3D model generation' checkbox checked
                        3. Enter your Meshy API key
                        4. Restart the connection to Claude"""
                }
            return {
                "enabled": True,
                "message": "Meshy integration is enabled and ready to use."
            }
        return {
            "enabled": False,
            "message": """Meshy integration is currently disabled. To enable it:
                        1. In the 3D Viewport, find the BlenderMCP panel in the sidebar (press N if hidden)
                        2. Check the 'Use Meshy AI 3D model generation' checkbox
                        3. Enter your Meshy API key
                        4. Restart the connection to Claude"""
        }

    def create_meshy_job(self, prompt=None, image_url=None, mode="preview",
                         preview_task_id=None, art_style="realistic",
                         topology="triangle", target_polycount=30000):
        """Create a Meshy text-to-3D or image-to-3D generation job"""
        api_key = bpy.context.scene.blendermcp_meshy_api_key
        if not api_key:
            return {"error": "Meshy API key is not configured"}

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }

        try:
            if image_url:
                url = "https://api.meshy.ai/openapi/v1/image-to-3d"
                payload = {"image_url": image_url}
                self._set_operation_status("Creating Meshy image-to-3D job...")
            elif mode == "refine" and preview_task_id:
                url = "https://api.meshy.ai/openapi/v2/text-to-3d"
                payload = {
                    "mode": "refine",
                    "preview_task_id": preview_task_id,
                }
                self._set_operation_status("Creating Meshy refine job...")
            else:
                url = "https://api.meshy.ai/openapi/v2/text-to-3d"
                payload = {
                    "mode": "preview",
                    "prompt": prompt or "",
                    "ai_model": "meshy-6",
                    "art_style": art_style,
                    "topology": topology,
                    "target_polycount": target_polycount,
                }
                self._set_operation_status("Creating Meshy text-to-3D job...")

            response = requests.post(url, headers=headers, json=payload, timeout=30)

            if response.status_code not in (200, 201, 202):
                return {"error": f"Meshy API returned status {response.status_code}: {response.text}"}

            data = response.json()
            task_id = data.get("result") or data.get("task_id") or data.get("id")
            if not task_id:
                return {"error": f"No task ID in response: {data}"}

            task_type = "image-to-3d" if image_url else "text-to-3d"
            return {"task_id": task_id, "mode": mode, "task_type": task_type}

        except requests.exceptions.Timeout:
            return {"error": "Request timed out connecting to Meshy API"}
        except Exception as e:
            return {"error": str(e)}
        finally:
            self._set_operation_status("")

    def poll_meshy_job_status(self, task_id, task_type="text-to-3d"):
        """Poll the status of a Meshy generation job"""
        api_key = bpy.context.scene.blendermcp_meshy_api_key
        if not api_key:
            return {"error": "Meshy API key is not configured"}

        headers = {
            "Authorization": f"Bearer {api_key}",
        }

        try:
            if task_type == "image-to-3d":
                url = f"https://api.meshy.ai/openapi/v1/image-to-3d/{task_id}"
            else:
                url = f"https://api.meshy.ai/openapi/v2/text-to-3d/{task_id}"

            response = requests.get(url, headers=headers, timeout=30)

            if response.status_code != 200:
                return {"error": f"Meshy API returned status {response.status_code}: {response.text}"}

            return response.json()

        except requests.exceptions.Timeout:
            return {"error": "Request timed out connecting to Meshy API"}
        except Exception as e:
            return {"error": str(e)}

    def import_meshy_asset(self, task_id, task_type="text-to-3d", name=None):
        """Import a completed Meshy asset into Blender"""
        self._set_operation_status(f"Importing Meshy asset: {task_id}...")
        try:
            status_result = self.poll_meshy_job_status(task_id, task_type)
            if "error" in status_result:
                return status_result

            status = status_result.get("status", "")
            if status != "SUCCEEDED":
                return {"error": f"Task is not complete yet. Current status: {status}"}

            model_urls = status_result.get("model_urls", {})
            glb_url = model_urls.get("glb")
            if not glb_url:
                return {"error": "No GLB model URL available in task result"}

            response = requests.get(glb_url, timeout=120)
            if response.status_code != 200:
                return {"error": f"Failed to download GLB model: status {response.status_code}"}

            temp_file = tempfile.NamedTemporaryFile(
                delete=False,
                prefix=f"meshy_{task_id}_",
                suffix=".glb",
            )
            temp_file.write(response.content)
            temp_file.close()

            mesh_obj = BlenderMCPServer._clean_imported_glb(temp_file.name, mesh_name=name)

            try:
                os.unlink(temp_file.name)
            except Exception:
                pass

            if mesh_obj is None:
                return {"error": "Failed to import GLB model"}

            result = {
                "name": mesh_obj.name,
                "type": mesh_obj.type,
                "location": [mesh_obj.location.x, mesh_obj.location.y, mesh_obj.location.z],
                "rotation": [mesh_obj.rotation_euler.x, mesh_obj.rotation_euler.y, mesh_obj.rotation_euler.z],
                "scale": [mesh_obj.scale.x, mesh_obj.scale.y, mesh_obj.scale.z],
            }

            if mesh_obj.type == "MESH":
                bounding_box = self._get_aabb(mesh_obj)
                result["world_bounding_box"] = bounding_box

            return result

        except Exception as e:
            return {"error": str(e)}
        finally:
            self._set_operation_status("")
    #endregion

class BLENDERMCP_AddonPreferences(bpy.types.AddonPreferences):
    bl_idname = __name__

    log_to_file: BoolProperty(
        name="Log to File",
        description="Write logs to blendermcp.log in Blender config directory",
        default=False,
        update=lambda self, ctx: self._toggle_file_logging()
    )

    def _toggle_file_logging(self):
        from logging.handlers import RotatingFileHandler
        handler_name = "blendermcp_file"
        for h in logger.handlers[:]:
            if getattr(h, 'name', '') == handler_name:
                logger.removeHandler(h)
                h.close()
        if self.log_to_file:
            log_dir = bpy.utils.user_resource('CONFIG')
            log_path = os.path.join(log_dir, "blendermcp.log")
            fh = RotatingFileHandler(log_path, maxBytes=5*1024*1024, backupCount=3)
            fh.name = handler_name
            fh.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
            logger.addHandler(fh)

    def draw(self, context):
        layout = self.layout

        layout.label(text="Logging:", icon='TEXT')
        box = layout.box()
        box.prop(self, "log_to_file", text="Log to File")

class BLENDERMCP_PT_Panel(bpy.types.Panel):
    bl_label = "Blender MCP"
    bl_idname = "BLENDERMCP_PT_Panel"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = 'BlenderMCP'

    def draw(self, context):
        layout = self.layout
        scene = context.scene

        layout.prop(scene, "blendermcp_port")
        layout.prop(scene, "blendermcp_use_polyhaven", text="Use assets from Poly Haven")

        layout.prop(scene, "blendermcp_use_hyper3d", text="Use Hyper3D Rodin 3D model generation")
        if scene.blendermcp_use_hyper3d:
            layout.prop(scene, "blendermcp_hyper3d_mode", text="Rodin Mode")
            layout.prop(scene, "blendermcp_hyper3d_api_key", text="API Key")
            layout.operator("blendermcp.set_hyper3d_free_trial_api_key", text="Set Free Trial API Key")

        layout.prop(scene, "blendermcp_use_sketchfab", text="Use assets from Sketchfab")
        if scene.blendermcp_use_sketchfab:
            layout.prop(scene, "blendermcp_sketchfab_api_key", text="API Key")

        layout.prop(scene, "blendermcp_use_hunyuan3d", text="Use Tencent Hunyuan 3D model generation")
        if scene.blendermcp_use_hunyuan3d:
            layout.prop(scene, "blendermcp_hunyuan3d_mode", text="Hunyuan3D Mode")
            if scene.blendermcp_hunyuan3d_mode == 'OFFICIAL_API':
                layout.prop(scene, "blendermcp_hunyuan3d_secret_id", text="SecretId")
                layout.prop(scene, "blendermcp_hunyuan3d_secret_key", text="SecretKey")
            if scene.blendermcp_hunyuan3d_mode == 'LOCAL_API':
                layout.prop(scene, "blendermcp_hunyuan3d_api_url", text="API URL")
                layout.prop(scene, "blendermcp_hunyuan3d_octree_resolution", text="Octree Resolution")
                layout.prop(scene, "blendermcp_hunyuan3d_num_inference_steps", text="Number of Inference Steps")
                layout.prop(scene, "blendermcp_hunyuan3d_guidance_scale", text="Guidance Scale")
                layout.prop(scene, "blendermcp_hunyuan3d_texture", text="Generate Texture")

        layout.prop(scene, "blendermcp_use_meshy", text="Use Meshy AI 3D model generation")
        if scene.blendermcp_use_meshy:
            layout.prop(scene, "blendermcp_meshy_api_key", text="API Key")

        if not scene.blendermcp_server_running:
            layout.prop(scene, "blendermcp_auth_token", text="Auth Token")
            layout.operator("blendermcp.start_server", text="Connect to MCP server")
        else:
            layout.operator("blendermcp.stop_server", text="Disconnect from MCP server")
            layout.label(text=f"Running on port {scene.blendermcp_port}")
            if scene.blendermcp_current_operation:
                layout.label(text=scene.blendermcp_current_operation, icon='SORTTIME')

class BLENDERMCP_OT_SetFreeTrialHyper3DAPIKey(bpy.types.Operator):
    bl_idname = "blendermcp.set_hyper3d_free_trial_api_key"
    bl_label = "Set Free Trial API Key"

    def execute(self, context):
        context.scene.blendermcp_hyper3d_api_key = RODIN_FREE_TRIAL_KEY
        context.scene.blendermcp_hyper3d_mode = 'MAIN_SITE'
        self.report({'INFO'}, "API Key set successfully!")
        return {'FINISHED'}

class BLENDERMCP_OT_StartServer(bpy.types.Operator):
    bl_idname = "blendermcp.start_server"
    bl_label = "Connect to Claude"
    bl_description = "Start the BlenderMCP server to connect with Claude"

    def execute(self, context):
        scene = context.scene

        if not hasattr(bpy.types, "blendermcp_server") or not bpy.types.blendermcp_server:
            bpy.types.blendermcp_server = BlenderMCPServer(port=scene.blendermcp_port)

        bpy.types.blendermcp_server.start()
        scene.blendermcp_server_running = True

        return {'FINISHED'}

class BLENDERMCP_OT_StopServer(bpy.types.Operator):
    bl_idname = "blendermcp.stop_server"
    bl_label = "Stop the connection to Claude"
    bl_description = "Stop the connection to Claude"

    def execute(self, context):
        scene = context.scene

        if hasattr(bpy.types, "blendermcp_server") and bpy.types.blendermcp_server:
            bpy.types.blendermcp_server.stop()
            del bpy.types.blendermcp_server

        scene.blendermcp_server_running = False

        return {'FINISHED'}

def register():
    bpy.types.Scene.blendermcp_port = IntProperty(
        name="Port",
        description="Port for the BlenderMCP server",
        default=9876,
        min=1024,
        max=65535
    )

    bpy.types.Scene.blendermcp_server_running = bpy.props.BoolProperty(
        name="Server Running",
        default=False
    )

    bpy.types.Scene.blendermcp_use_polyhaven = bpy.props.BoolProperty(
        name="Use Poly Haven",
        description="Enable Poly Haven asset integration",
        default=False
    )

    bpy.types.Scene.blendermcp_use_hyper3d = bpy.props.BoolProperty(
        name="Use Hyper3D Rodin",
        description="Enable Hyper3D Rodin generatino integration",
        default=False
    )

    bpy.types.Scene.blendermcp_hyper3d_mode = bpy.props.EnumProperty(
        name="Rodin Mode",
        description="Choose the platform used to call Rodin APIs",
        items=[
            ("MAIN_SITE", "hyper3d.ai", "hyper3d.ai"),
            ("FAL_AI", "fal.ai", "fal.ai"),
        ],
        default="MAIN_SITE"
    )

    bpy.types.Scene.blendermcp_hyper3d_api_key = bpy.props.StringProperty(
        name="Hyper3D API Key",
        subtype="PASSWORD",
        description="API Key provided by Hyper3D",
        default=""
    )

    bpy.types.Scene.blendermcp_use_hunyuan3d = bpy.props.BoolProperty(
        name="Use Hunyuan 3D",
        description="Enable Hunyuan asset integration",
        default=False
    )

    bpy.types.Scene.blendermcp_hunyuan3d_mode = bpy.props.EnumProperty(
        name="Hunyuan3D Mode",
        description="Choose a local or official APIs",
        items=[
            ("LOCAL_API", "local api", "local api"),
            ("OFFICIAL_API", "official api", "official api"),
        ],
        default="LOCAL_API"
    )

    bpy.types.Scene.blendermcp_hunyuan3d_secret_id = bpy.props.StringProperty(
        name="Hunyuan 3D SecretId",
        description="SecretId provided by Hunyuan 3D",
        default=""
    )

    bpy.types.Scene.blendermcp_hunyuan3d_secret_key = bpy.props.StringProperty(
        name="Hunyuan 3D SecretKey",
        subtype="PASSWORD",
        description="SecretKey provided by Hunyuan 3D",
        default=""
    )

    bpy.types.Scene.blendermcp_hunyuan3d_api_url = bpy.props.StringProperty(
        name="API URL",
        description="URL of the Hunyuan 3D API service",
        default="http://localhost:8081"
    )

    bpy.types.Scene.blendermcp_hunyuan3d_octree_resolution = bpy.props.IntProperty(
        name="Octree Resolution",
        description="Octree resolution for the 3D generation",
        default=256,
        min=128,
        max=512,
    )

    bpy.types.Scene.blendermcp_hunyuan3d_num_inference_steps = bpy.props.IntProperty(
        name="Number of Inference Steps",
        description="Number of inference steps for the 3D generation",
        default=20,
        min=20,
        max=50,
    )

    bpy.types.Scene.blendermcp_hunyuan3d_guidance_scale = bpy.props.FloatProperty(
        name="Guidance Scale",
        description="Guidance scale for the 3D generation",
        default=5.5,
        min=1.0,
        max=10.0,
    )

    bpy.types.Scene.blendermcp_hunyuan3d_texture = bpy.props.BoolProperty(
        name="Generate Texture",
        description="Whether to generate texture for the 3D model",
        default=False,
    )
    
    bpy.types.Scene.blendermcp_use_sketchfab = bpy.props.BoolProperty(
        name="Use Sketchfab",
        description="Enable Sketchfab asset integration",
        default=False
    )

    bpy.types.Scene.blendermcp_sketchfab_api_key = bpy.props.StringProperty(
        name="Sketchfab API Key",
        subtype="PASSWORD",
        description="API Key provided by Sketchfab",
        default=""
    )

    bpy.types.Scene.blendermcp_use_meshy = bpy.props.BoolProperty(
        name="Use Meshy",
        description="Enable Meshy AI 3D model generation integration",
        default=False
    )

    bpy.types.Scene.blendermcp_meshy_api_key = bpy.props.StringProperty(
        name="Meshy API Key",
        subtype="PASSWORD",
        description="API Key provided by Meshy",
        default=""
    )

    bpy.types.Scene.blendermcp_current_operation = bpy.props.StringProperty(
        name="Current Operation",
        default=""
    )

    bpy.types.Scene.blendermcp_auth_token = bpy.props.StringProperty(
        name="Auth Token",
        subtype="PASSWORD",
        description="Optional authentication token for socket connections",
        default=""
    )

    bpy.utils.register_class(BLENDERMCP_AddonPreferences)

    bpy.utils.register_class(BLENDERMCP_PT_Panel)
    bpy.utils.register_class(BLENDERMCP_OT_SetFreeTrialHyper3DAPIKey)
    bpy.utils.register_class(BLENDERMCP_OT_StartServer)
    bpy.utils.register_class(BLENDERMCP_OT_StopServer)

    logger.info("BlenderMCP addon registered")

def unregister():
    if hasattr(bpy.types, "blendermcp_server") and bpy.types.blendermcp_server:
        bpy.types.blendermcp_server.stop()
        del bpy.types.blendermcp_server

    for h in logger.handlers[:]:
        if getattr(h, 'name', '') == 'blendermcp_file':
            logger.removeHandler(h)
            h.close()

    bpy.utils.unregister_class(BLENDERMCP_PT_Panel)
    bpy.utils.unregister_class(BLENDERMCP_OT_SetFreeTrialHyper3DAPIKey)
    bpy.utils.unregister_class(BLENDERMCP_OT_StartServer)
    bpy.utils.unregister_class(BLENDERMCP_OT_StopServer)
    bpy.utils.unregister_class(BLENDERMCP_AddonPreferences)

    del bpy.types.Scene.blendermcp_port
    del bpy.types.Scene.blendermcp_server_running
    del bpy.types.Scene.blendermcp_use_polyhaven
    del bpy.types.Scene.blendermcp_use_hyper3d
    del bpy.types.Scene.blendermcp_hyper3d_mode
    del bpy.types.Scene.blendermcp_hyper3d_api_key
    del bpy.types.Scene.blendermcp_use_sketchfab
    del bpy.types.Scene.blendermcp_sketchfab_api_key
    del bpy.types.Scene.blendermcp_use_meshy
    del bpy.types.Scene.blendermcp_meshy_api_key
    del bpy.types.Scene.blendermcp_use_hunyuan3d
    del bpy.types.Scene.blendermcp_hunyuan3d_mode
    del bpy.types.Scene.blendermcp_hunyuan3d_secret_id
    del bpy.types.Scene.blendermcp_hunyuan3d_secret_key
    del bpy.types.Scene.blendermcp_hunyuan3d_api_url
    del bpy.types.Scene.blendermcp_hunyuan3d_octree_resolution
    del bpy.types.Scene.blendermcp_hunyuan3d_num_inference_steps
    del bpy.types.Scene.blendermcp_hunyuan3d_guidance_scale
    del bpy.types.Scene.blendermcp_hunyuan3d_texture
    del bpy.types.Scene.blendermcp_current_operation
    del bpy.types.Scene.blendermcp_auth_token

    logger.info("BlenderMCP addon unregistered")

if __name__ == "__main__":
    register()
