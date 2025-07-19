import asyncio
import threading
import weakref
import struct
import re
from enum import Enum
import os
import time

import numpy as np

from libs.websockets import client
from libs.websockets.exceptions import ConnectionClosed, InvalidURI, WebSocketException

try:
    from PySide6 import QtCore
except ImportError:
    from PySide2 import QtCore


# Maximum message size (bytes) for the WebSocket connection
max_size = 2 ** 32 - 1

# Define message type identifiers matching the Plasticity protocol
class MessageType(Enum):
    TRANSACTION_1    = 0   # Full scene transaction (initial or live updates)
    ADD_1            = 1   # New object added
    UPDATE_1         = 2   # Object updated
    DELETE_1         = 3   # Object deleted
    MOVE_1           = 4   # Object moved (if applicable)
    ATTRIBUTE_1      = 5   # Attribute change (if applicable)
    NEW_VERSION_1    = 10  # New version of an open Plasticity file
    NEW_FILE_1       = 11  # New Plasticity file opened
    LIST_ALL_1       = 20  # Request for listing all objects
    LIST_SOME_1      = 21  # Request for listing some objects (by ID)
    LIST_VISIBLE_1   = 22  # Request for listing visible objects
    SUBSCRIBE_ALL_1  = 23  # Subscribe to all updates (live link on)
    SUBSCRIBE_SOME_1 = 24  # Subscribe to specific objects
    UNSUBSCRIBE_ALL_1= 25  # Unsubscribe from all updates (live link off)
    REFACET_SOME_1   = 26  # Request to refacet (remesh) specific objects

# Define object type identifiers (for decoding geometry data)
class ObjectType(Enum):
    SOLID = 0   # Solid body (mesh data will follow)
    SHEET = 1   # Sheet body (mesh data will follow)
    WIRE  = 2   # Wire/curve (if applicable, may not include mesh data)
    GROUP = 5   # Group or assembly
    EMPTY = 6   # Empty container or null object

# Define facet shape types for refaceting (used when sending refacet requests)
class FacetShapeType(Enum):
    ANY    = 20500
    CUT    = 20501
    CONVEX = 20502

class UISignalEmitter(QtCore.QObject):
    """
    QObject to emit signals that can be connected to UI slots
    This must be created on the main thread
    """
    connected = QtCore.Signal()
    disconnected = QtCore.Signal()
    error_occurred = QtCore.Signal(str)
    status_update = QtCore.Signal(str, str)  # level, message

    def __init__(self):
        super().__init__()
        self.moveToThread(QtCore.QCoreApplication.instance().thread())

class PlasticityClient:
    def __init__(self, handler):
        self.server = None
        self.websocket = None
        self.loop = asyncio.new_event_loop()
        self.connected = False
        self.handler = handler
        self.debug_log = True
        self.subscribed = False
        self.filename = None
        self.message_id = 0
        self.thread = None

        self.debug_log_path = r"C:\3dsmax_dev\PlasticityLiveLink_Max\plasticity_debug.log"
        self._ensure_debug_file()
        
        # Create signal emitter
        self.signals = UISignalEmitter()
        self.signals.connected.connect(self._on_connected_signal)
        self.signals.disconnected.connect(self._on_disconnected_signal)



    def _ensure_debug_file(self):
        """Ensure debug log directory and file exist"""
        try:
            os.makedirs(os.path.dirname(self.debug_log_path), exist_ok=True)
            if not os.path.exists(self.debug_log_path):
                with open(self.debug_log_path, 'w'): pass
        except Exception as e:
            print(f"Could not initialize debug log: {e}")

    def log_debug(self, message):
        """Thread-safe debug logging"""
        if not self.debug_log:
            return
        
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        log_entry = f"[{timestamp}] {message}\n"
        
        try:
            with open(self.debug_log_path, "a", encoding="utf-8") as f:
                f.write(log_entry)
        except Exception as e:
            print(f"Debug log failed: {e}")

    def connect(self, server):
        """Main method to initiate connection - called from UI thread"""
        self.server = server
        self.log_debug("connect###############")
        
        def start_loop():
            self.log_debug("On LOOP - Starting event loop")
            asyncio.set_event_loop(self.loop)
            self.log_debug("Creating connect_async task")
            self.loop.create_task(self.connect_async(server))
            self.log_debug("Running event loop")
            self.loop.run_forever()
            self.log_debug("Event loop exited")

        self.thread = threading.Thread(target=start_loop, daemon=True)
        self.thread.start()
        self.log_debug(f"CCCCCCCCONNECT - Thread started (alive: {self.thread.is_alive()})")

    async def connect_async(self, server):
        """Async connection handler - runs in background thread"""
        self.log_debug("connect_async started")
        try:
            self.log_debug(f"Attempting to connect to ws://{server}")
            self._emit_ui_signal('status_update', 'INFO', f"Connecting to ws://{server}")
            
            ws = await client.connect(f"ws://{server}", max_size=max_size)
            self.log_debug("WebSocket connection established")
            
            self.websocket = weakref.proxy(ws)
            self.connected = True
            self.log_debug("Connection successful")
            
            self._emit_ui_signal('connected')
            self.log_debug("Connected signal emitted")
            
            # Ping test
            try:
                await ws.ping()
                self.log_debug("Ping successful")
            except Exception as e:
                self.log_debug(f"Ping failed: {str(e)}")
            
            self.log_debug("Starting message processing loop")

            # Message processing loop
            while True:
                try:
                    message = await ws.recv()
                    if message is None:
                        raise ConnectionClosed(code=1000, reason="Connection closed")
                    self.log_debug(f"Received message of length {len(message)} bytes")
                    await self.on_message(ws, message)
                    
                except ConnectionClosed as e:
                    self.log_debug(f"Disconnected: {e}")
                    break
                except Exception as e:
                    self.log_debug(f"Message error: {e}")
                    continue
                    
        except Exception as e:
            self.log_debug(f"Connection error: {e}")
            self._emit_ui_signal('error_occurred', str(e))
        finally:
            self.log_debug("Cleaning up connection")
            self.cleanup_connection()

    def disconnect(self):
        """Thread-safe disconnection method"""
        if not self.connected:
            return

        # Schedule the disconnect on the event loop
        future = asyncio.run_coroutine_threadsafe(self._async_disconnect(), self.loop)
        
        try:
            # Wait for disconnect to complete with timeout
            future.result(timeout=5)
        except Exception as e:
            print(f"Disconnect timed out or failed: {e}")
        finally:
            self._safe_stop_loop()
            print("Disconnected")

    async def _async_disconnect(self):
        """Actual disconnect coroutine"""
        try:
            if self.websocket:
                await self.websocket.close()
        except Exception as e:
            print(f"WebSocket close error: {e}")
        finally:
            self.cleanup_connection()

    def _safe_stop_loop(self):
        """Safely stop the event loop"""
        if self.loop.is_running():
            self.loop.call_soon_threadsafe(self.loop.stop)
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=1)

    def cleanup_connection(self):
        """Clean up connection resources"""
        self.connected = False
        self.websocket = None
        self.filename = None
        self.subscribed = False
        self._emit_ui_signal('disconnected')

    def _emit_ui_signal(self, signal, *args):
        """Thread-safe signal emission"""
        if not hasattr(self.signals, signal):
            return
            
        if QtCore.QThread.currentThread() != QtCore.QCoreApplication.instance().thread():
            # Emit from background thread
            getattr(self.signals, signal).emit(*args)
        else:
            # Direct call if already on main thread
            getattr(self.signals, signal).emit(*args)

    def _on_connected_signal(self):
        """Handle connection success on main thread"""
        print("_on_connected_singnal Connected")
        if hasattr(self.handler, 'on_connect'):
            self.handler.on_connect()

    def _on_disconnected_signal(self):
        """Handle disconnection on main thread"""
        if hasattr(self.handler, 'on_disconnect'):
            self.handler.on_disconnect()

    # --- Outgoing Message Methods (requests to server) ---
    def list_all(self):
        """
        Request all items from server (assumes UI prevents calls when disconnected)
        """
        future = asyncio.run_coroutine_threadsafe(self.list_all_async(), self.loop)
        print("listall")
        try:
            future.result(timeout=2)  # Brief wait to catch obvious failures
        except Exception:
            pass  # Let the async method handle errors via message loop

    async def list_all_async(self):
        """
        Core implementation - runs in event loop thread
        """
        self.message_id += 1
        msg = struct.pack("<II", MessageType.LIST_ALL_1.value, self.message_id)
        await self.websocket.send(msg)

    def list_visible(self):
        """
        Request visible items from server (assumes UI prevents calls when disconnected)
        """
        future = asyncio.run_coroutine_threadsafe(self.list_visible_async(), self.loop)
        print("listviz")
        try:
            future.result(timeout=2)
        except Exception:
            pass

    async def list_visible_async(self):
        """
        Send a message to the server to list visible items.
        """
        self.message_id += 1  # Increment message_id
        get_objects_message = struct.pack("<I", MessageType.LIST_VISIBLE_1.value)
        get_objects_message += struct.pack("<I", self.message_id)
        await self.websocket.send(get_objects_message)


    async def on_message(self, ws, message):
        self.log_debug("on message async entered")
        self.log_debug(f"DEBUG: Received message of length {len(message)} bytes")  # Add this
        view = memoryview(message)
        offset = 0
        message_type = MessageType(
            int.from_bytes(view[offset:offset + 4], 'little'))
        offset += 4

        if message_type == MessageType.TRANSACTION_1:
            self.__on_transaction(view, offset, update_only=True)

        elif message_type == MessageType.LIST_ALL_1 or message_type == MessageType.LIST_SOME_1 or message_type == MessageType.LIST_VISIBLE_1:
            message_id = int.from_bytes(view[offset:offset + 4], 'little')
            offset += 4

            code = int.from_bytes(view[offset:offset + 4], 'little')
            offset += 4

            if code != 200:
                self.log_debug({'ERROR'}, f"List all failed with code: {code}")
                return

            # NOTE: ListAll only has an Add message inside it so it is a bit unlike a regular transaction
            self.__on_transaction(view, offset, update_only=False)

        elif message_type == MessageType.NEW_VERSION_1:
            
            filename_length = int.from_bytes(view[offset:offset + 4], 'little')
            offset += 4

            filename = view[offset:offset +
                            filename_length].tobytes().decode('utf-8')
            offset += filename_length

            self.filename = filename

            # Add string padding for byte alignment
            padding = (4 - (filename_length % 4)) % 4
            offset += padding

            version = int.from_bytes(view[offset:offset + 4], 'little')
            offset += 4

            # bpy.app.timers.register(
            #     lambda: self.handler.on_new_version(filename, version), first_interval=0.001)
            # maya.utils.executeDeferred(lambda: self.handler.on_new_version(filename, version))
            QtCore.QTimer.singleShot(0, lambda: self.handler.on_new_version(filename, version))


        elif message_type == MessageType.NEW_FILE_1:
            filename_length = int.from_bytes(view[offset:offset + 4], 'little')
            offset += 4

            filename = view[offset:offset +
                            filename_length].tobytes().decode('utf-8')
            offset += filename_length

            self.filename = filename

            # bpy.app.timers.register(
            #     lambda: self.handler.on_new_file(filename), first_interval=0.001)
            # maya.utils.executeDeferred(lambda: self.handler.on_new_file(filename))
            QtCore.QTimer.singleShot(0, lambda: self.handler.on_new_file(filename))

        elif message_type == MessageType.REFACET_SOME_1:
            self.__on_refacet(view, offset)


    def on_message_item(self, view, transaction):
        self.log_debug("on_message_item entered")
        offset = 0
        message_type = MessageType(
            int.from_bytes(view[offset:offset + 4], 'little'))
        offset += 4

        if message_type == MessageType.DELETE_1:
            num_objects = int.from_bytes(view[:4], 'little')
            offset += 4
            transaction["delete"].extend(
                np.frombuffer(view[offset:offset + num_objects * 4], dtype=np.int32))
        elif message_type == MessageType.ADD_1:
            self.log_debug(" on message item ADD")
            transaction["add"].extend(decode_objects(view[4:], logger=self.log_debug))
        elif message_type == MessageType.UPDATE_1:
            self.log_debug(" on message item UPDATE")
            transaction["update"].extend(decode_objects(view[4:], logger=self.log_debug))


    def subscribe_all(self):
        """Subscribe to updates for all objects (enable live link)."""
        self.log_debug("SUBSCRIBE ALL")
        if self.connected:
            future = asyncio.run_coroutine_threadsafe(self.subscribe_all_async(), self.loop)
            future.result()
            # Mark that we have an active subscription
            self.subscribed = True

    async def subscribe_all_async(self):
        """Async coroutine to send a SUBSCRIBE_ALL request."""
        self.log_debug("ASYNC SUBSCRIBE ALL")
        self.message_id += 1
        msg = struct.pack("<I", MessageType.SUBSCRIBE_ALL_1.value)
        msg += struct.pack("<I", self.message_id)
        await self.websocket.send(msg)

    def unsubscribe_all(self):
        """Unsubscribe from all updates (disable live link)."""
        self.log_debug("UNSUBSCRIBE")
        if self.connected:
            future = asyncio.run_coroutine_threadsafe(self.unsubscribe_all_async(), self.loop)
            future.result()
            self.subscribed = False

    async def unsubscribe_all_async(self):
        """Async coroutine to send an UNSUBSCRIBE_ALL request."""
        self.log_debug("ASYNC UNSUBSCRIBE")
        self.message_id += 1
        msg = struct.pack("<I", MessageType.UNSUBSCRIBE_ALL_1.value)
        msg += struct.pack("<I", self.message_id)
        await self.websocket.send(msg)

    def subscribe_some(self, filename: str, plasticity_ids: list[int]):
        """
        Subscribe to updates for specific objects by their Plasticity IDs.
        `filename` is the Plasticity file name, and `plasticity_ids` is a list of object IDs.
        """
        self.log_debug("SUBSCRIBE SOME")
        if self.connected:
            future = asyncio.run_coroutine_threadsafe(
                self.subscribe_some_async(filename, plasticity_ids), self.loop
            )
            future.result()

    async def subscribe_some_async(self, filename: str, plasticity_ids: list[int]):
        """Async coroutine to send a SUBSCRIBE_SOME request for specific objects."""
        if not plasticity_ids:
            return  # nothing to subscribe if list is empty
        self.message_id += 1
        # Pack message type and ID
        msg = struct.pack("<I", MessageType.SUBSCRIBE_SOME_1.value)
        msg += struct.pack("<I", self.message_id)
        # Pack filename length and filename bytes
        msg += struct.pack("<I", len(filename))
        msg += struct.pack(f"<{len(filename)}s", filename.encode('utf-8'))
        # Pad filename to 4-byte boundary
        padding = (4 - (len(filename) % 4)) % 4
        if padding:
            msg += struct.pack(f"<{padding}x")
        # Pack number of IDs and each ID
        msg += struct.pack("<I", len(plasticity_ids))
        for pid in plasticity_ids:
            msg += struct.pack("<I", pid)
        # Send the composed message
        await self.websocket.send(msg)
        self.log_debug("SUBSCRIBE SOME ASYNC SENT****")

    def refacet_some(self, filename: str, plasticity_ids: list[int],
                     relative_to_bbox: bool = True,
                     curve_chord_tolerance: float = 0.01,
                     curve_chord_angle: float = 0.35,
                     surface_plane_tolerance: float = 0.01,
                     surface_plane_angle: float = 0.35,
                     match_topology: bool = True,
                     max_sides: int = 3,
                     plane_angle: float = 0.0,
                     min_width: float = 0.0,
                     max_width: float = 0.0,
                     curve_chord_max: float = 0.0,
                     shape: FacetShapeType = FacetShapeType.CUT):
        """
        Request a refacet (remeshing) operation on specified objects.
        This will send the current mesh data to Plasticity for refaceting.
        """
        if self.connected:
            # Log the operation (to console or UI for feedback)
            self.handler.report("INFO", "Refaceting selected objects...")
            future = asyncio.run_coroutine_threadsafe(
                self.refacet_some_async(filename, plasticity_ids, relative_to_bbox,
                                         curve_chord_tolerance, curve_chord_angle,
                                         surface_plane_tolerance, surface_plane_angle,
                                         match_topology, max_sides, plane_angle,
                                         min_width, max_width, curve_chord_max, shape),
                self.loop
            )
            future.result()

    async def refacet_some_async(self, filename: str, plasticity_ids: list[int],
                                 relative_to_bbox: bool = True,
                                 curve_chord_tolerance: float = 0.01,
                                 curve_chord_angle: float = 0.35,
                                 surface_plane_tolerance: float = 0.01,
                                 surface_plane_angle: float = 0.35,
                                 match_topology: bool = True,
                                 max_sides: int = 3,
                                 plane_angle: float = 0.0,
                                 min_width: float = 0.0,
                                 max_width: float = 0.0,
                                 curve_chord_max: float = 0.0,
                                 shape: FacetShapeType = FacetShapeType.CUT):
        """Async coroutine to send a REFACET_SOME request with specified parameters."""
        if not plasticity_ids:
            return  # nothing to refacet if list is empty
        self.message_id += 1
        # Pack message header (type and unique message ID)
        msg = struct.pack("<I", MessageType.REFACET_SOME_1.value)
        msg += struct.pack("<I", self.message_id)
        # Pack filename length and filename
        msg += struct.pack("<I", len(filename))
        msg += struct.pack(f"<{len(filename)}s", filename.encode('utf-8'))
        # Pad filename to 4-byte boundary
        padding = (4 - (len(filename) % 4)) % 4
        if padding:
            msg += struct.pack(f"<{padding}x")
        # Pack number of objects and their IDs
        msg += struct.pack("<I", len(plasticity_ids))
        for pid in plasticity_ids:
            msg += struct.pack("<I", pid)
        # Pack boolean flags and numeric parameters for refacet operation
        msg += struct.pack("<I", 1 if relative_to_bbox else 0)
        msg += struct.pack("<f", curve_chord_tolerance)
        msg += struct.pack("<f", curve_chord_angle)
        msg += struct.pack("<f", surface_plane_tolerance)
        msg += struct.pack("<f", surface_plane_angle)
        msg += struct.pack("<I", 1 if match_topology else 0)
        msg += struct.pack("<I", max_sides)
        msg += struct.pack("<f", plane_angle)
        msg += struct.pack("<f", min_width)
        msg += struct.pack("<f", max_width)
        msg += struct.pack("<f", curve_chord_max)
        msg += struct.pack("<I", shape.value)
        # Send the composed refacet request message
        await self.websocket.send(msg)


    def __on_transaction(self, view: memoryview, offset: int, update_only: bool):
        """
        Internal helper to decode a TRANSACTION message.
        If update_only is True, it's a live update (only changes);
        if False, it's a full list response (treat adds as new data).
        """
        # Read Plasticity file name
        self.log_debug("Client side on transaction entered")
        filename_length = int.from_bytes(view[offset:offset + 4], 'little')
        offset += 4
        filename = view[offset:offset + filename_length].tobytes().decode('utf-8')
        offset += filename_length
        self.filename = filename  # current active file name
        # Skip padding for 4-byte alignment
        padding = (4 - (filename_length % 4)) % 4
        offset += padding
        # Read file version (an integer)
        version = int.from_bytes(view[offset:offset + 4], 'little')
        offset += 4
        # Read number of messages (objects) in this transaction
        num_messages = int.from_bytes(view[offset:offset + 4], 'little')
        offset += 4

        # Prepare a transaction dict to collect changes
        transaction = {
            "filename": filename,
            "version": version,
            "delete": [],
            "add": [],
            "update": []
        }
        # Loop through each message (entity) in the transaction
        for _ in range(num_messages):
            item_length = int.from_bytes(view[offset:offset + 4], 'little')
            offset += 4
            # Each item is a message (Add, Update, or Delete), decode it
            self.on_message_item(view[offset:offset + item_length], transaction)
            offset += item_length

        # Dispatch the completed transaction to the handler on the main thread
        if update_only:
            # For live updates, treat as a transaction update
            self.log_debug("ARRIVED update only")
            self.handler.on_transaction(transaction)
        else:
            self.log_debug("ARRIVED on list")
            # For full list responses, use on_list (which may handle as a fresh add list)
            self.handler.on_list(transaction)
            self.log_debug("Exit on list")

    def __on_refacet(self, view: memoryview, offset: int):
        """
        Internal helper to decode a REFACET_SOME response message.
        Currently, this just notifies the handler that refacet is complete.
        (The actual mesh data returned can be processed here if needed.)
        """
        # Read and skip message_id and status code
        message_id = int.from_bytes(view[offset:offset + 4], 'little')
        offset += 4
        code = int.from_bytes(view[offset:offset + 4], 'little')
        offset += 4
        if code != 200:
            # Refacet request failed on the server
            self.handler.report("ERROR", f"Refacet operation failed with code: {code}")
            return
        # Read Plasticity filename (even if not used currently)
        filename_length = int.from_bytes(view[offset:offset + 4], 'little')
        offset += 4
        filename = view[offset:offset + filename_length].tobytes().decode('utf-8')
        offset += filename_length
        self.filename = filename
        # Align to 4-byte boundary after filename
        padding = (4 - (filename_length % 4)) % 4
        offset += padding
        # Skip file version and any returned data for now (not used in this implementation)
        # (In a fuller implementation, you would parse updated mesh data here.)
        # Notify handler that refacet operation is complete (on main thread)
        QtCore.QTimer.singleShot(0, lambda: self.handler.on_refacet())



def decode_objects(buffer, logger=None):
    if logger:
        logger("decode objects entered")
    view = memoryview(buffer)
    num_objects = int.from_bytes(view[:4], 'little')
    offset = 4
    objects = []

    for _ in range(num_objects):
        object_type, object_id, version_id, parent_id, material_id, flags, name, vertices, faces, normals, offset, groups, face_ids = decode_object_data(
            view, offset, logger)
        objects.append({"type": object_type, "id": object_id, "version": version_id, "parent_id": parent_id, "material_id": material_id,
                       "flags": flags, "name": name, "vertices": vertices, "faces": faces, "normals": normals, "groups": groups, "face_ids": face_ids})
    return objects  


def decode_object_data(view, offset, logger=None):
    if logger:
        logger("decode_object_data entered")
    object_type = int.from_bytes(view[offset:offset + 4], 'little')
    offset += 4

    object_id = int.from_bytes(view[offset:offset + 4], 'little')
    offset += 4

    version_id = int.from_bytes(view[offset:offset + 4], 'little')
    offset += 4

    parent_id = int.from_bytes(view[offset:offset + 4], 'little', signed=True)
    offset += 4

    material_id = int.from_bytes(
        view[offset:offset + 4], 'little', signed=True)
    offset += 4

    flags = int.from_bytes(view[offset:offset + 4], 'little')
    offset += 4

    name_length = int.from_bytes(view[offset:offset + 4], 'little')
    offset += 4

    name = view[offset:offset + name_length].tobytes().decode('utf-8')
    offset += name_length

    # Add object ID suffix to the name
    name_with_id = f"{name}_{object_id}"

    # Add string padding for byte alignment
    padding = (4 - (name_length % 4)) % 4
    offset += padding

    vertices = None
    faces = None
    normals = None
    groups = None
    face_ids = None

    if object_type == ObjectType.SOLID.value or object_type == ObjectType.SHEET.value:
        num_vertices = int.from_bytes(view[offset:offset + 4], 'little')
        offset += 4

        vertices = np.frombuffer(
            view[offset:offset + num_vertices * 12], dtype=np.float32)
        offset += num_vertices * 12

        num_faces = int.from_bytes(view[offset:offset + 4], 'little')
        offset += 4

        faces = np.frombuffer(
            view[offset:offset + num_faces * 12], dtype=np.int32)
        offset += num_faces * 12

        num_normals = int.from_bytes(view[offset:offset + 4], 'little')
        offset += 4

        normals = np.frombuffer(
            view[offset:offset + num_normals * 12], dtype=np.float32)
        offset += num_normals * 12

        num_groups = int.from_bytes(view[offset:offset + 4], 'little')
        offset += 4

        groups = np.frombuffer(
            view[offset:offset + num_groups * 4], dtype=np.int32)
        offset += num_groups * 4
        # NOTE: As of blender 4.2, the concrete type of user attributes cannot be numpy arrays.
        groups = groups.tolist()

        num_face_ids = int.from_bytes(view[offset:offset + 4], 'little')
        offset += 4

        face_ids = np.frombuffer(
            view[offset:offset + num_face_ids * 4], dtype=np.int32)
        offset += num_face_ids * 4
        # NOTE: As of blender 4.2, the concrete type of user attributes cannot be numpy arrays.
        face_ids = face_ids.tolist()

    elif object_type == ObjectType.GROUP.value:
        pass



    if logger:
        decoded_data = {
            'object_type': object_type,
            'object_id': object_id,
            'version_id': version_id,
            'parent_id': parent_id,
            'material_id': material_id,
            'flags': flags,
            'name': name,
            'vertices': vertices.tolist() if isinstance(vertices, np.ndarray) else vertices,
            'faces': faces.tolist() if isinstance(faces, np.ndarray) else faces,
            'normals': normals.tolist() if isinstance(normals, np.ndarray) else normals,
            'groups': groups.tolist() if isinstance(groups, np.ndarray) else groups,
            'face_ids': face_ids.tolist() if isinstance(face_ids, np.ndarray) else face_ids
        }
        # logger("CLIENT: decode_object_data: Returning from function with: " + ", ".join(f"{key}={value}" for key, value in decoded_data.items()))
    
    final_name = name_with_id 
    if final_name and final_name[0].isdigit():
        final_name = f"Null_{final_name}"
    return object_type, object_id, version_id, parent_id, material_id, flags, final_name, vertices, faces, normals, offset, groups, face_ids

def sanitize_name(name):
    """Sanitizes a string to remove invalid characters and replace them with underscores."""
    return re.sub(r'[^a-zA-Z0-9_]', '_', name).strip() if isinstance(name, str) else ""

