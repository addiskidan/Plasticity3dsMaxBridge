# import maya.cmds as cmds
# import maya.utils
from client import decode_object_data
from enum import Enum
import numpy as np
# import maya.OpenMaya as om
import re
import os
import time
import queue
# import plasticity_ui
from contextlib import contextmanager
from mainthread import queue_main_thread
from types import SimpleNamespace
try:
    from PySide6 import QtCore
    from PySide6.QtCore import QCoreApplication
except ImportError:
    from PySide2 import QtCore
    from PySide2.QtCore import QCoreApplication


class PlasticityIdUniquenessScope(Enum):
    ITEM = 0
    GROUP = 1
    EMPTY = 2

class ObjectType(Enum):
    SOLID = 0
    SHEET = 1
    WIRE = 2
    GROUP = 5
    EMPTY = 6

class SceneHandler:
    def __init__(self, plasticity_ui):
        self.plasticity_ui = plasticity_ui
        self.connected = False
        self.files = {}
        self.debug_log = True
        self.debug_log_path = r"C:\3dsmax_dev\PlasticityLiveLink_Max\handler_debug.log"
        self._ensure_debug_file()

    def on_connect(self):
        self.log_debug("[HANDLER] on_connect called")
        QtCore.QTimer.singleShot(0, lambda: print("[HANDLER] Triggering UI update"))
        QtCore.QTimer.singleShot(0, self.plasticity_ui.update_ui_connected)


    def on_disconnect(self):
        """
        Called when the client disconnects from the server.
        """
        self.log_debug("[HANDLER] on_disconnect called")
        self.connected = False
        QtCore.QTimer.singleShot(0, self.plasticity_ui.update_ui_disconnected)


    def report(self, level, message):
        print(f"[{level}] {message}", flush=True)
        QtCore.QTimer.singleShot(0, lambda: self.plasticity_ui.console_output.append(f"[{level}] {message}"))


    def sanitize_name(self, name):
        """
        Replaces invalid characters in Maya node names with underscores.
        """
        return re.sub(r'[^a-zA-Z0-9_]', '_', name)

    def on_refacet(self, *args, **kwargs):
        self.plasticity_ui.console_output.append("Refacet completed")

    def on_new_version(self, filename, version):
        self.plasticity_ui.console_output.append(f"New version for {filename} v{version}")

    def on_new_file(self, filename):
        self.plasticity_ui.console_output.append(f"New file: {filename}")

    def report(self, level, message):
        self.plasticity_ui.console_output.append(f"[{level}] {message}")

    # def __create_mesh(self, name, verts, indices, normals, groups, face_ids):
    #     self.log_debug("__create_mesh called using pymxs")

    #     try:
    #         import pymxs
    #         rt = pymxs.runtime

    #         scale_factor = self.plasticity_ui.scale_spin.value()

    #         # Convert numpy arrays to Python lists if necessary
    #         verts = verts.tolist() if isinstance(verts, np.ndarray) else verts
    #         indices = indices.tolist() if isinstance(indices, np.ndarray) else indices
    #         normals = normals.tolist() if isinstance(normals, np.ndarray) else normals

    #         num_verts = len(verts) // 3
    #         num_faces = len(indices) // 3

    #         # Create a new mesh
    #         mesh = rt.mesh()
    #         mesh.numverts = num_verts
    #         mesh.numfaces = num_faces

    #         # Set vertices
    #         for i in range(num_verts):
    #             x = verts[i * 3] * scale_factor
    #             y = verts[i * 3 + 1] * scale_factor
    #             z = verts[i * 3 + 2] * scale_factor
    #             rt.SetVert(mesh, i + 1, rt.Point3(x, y, z))

    #         # Set faces
    #         for i in range(num_faces):
    #             a = indices[i * 3] + 1
    #             b = indices[i * 3 + 1] + 1
    #             c = indices[i * 3 + 2] + 1
    #             rt.setFace(mesh, i + 1, a, b, c)
    #             rt.setEdgeVis(mesh, i + 1, 1, True)
    #             rt.setEdgeVis(mesh, i + 1, 2, True)
    #             rt.setEdgeVis(mesh, i + 1, 3, True)

    #         # Finalize the mesh
    #         rt.update(mesh)
    #         rt.redrawViews()
    #         node = rt.objects[-1]  # Assumes the newly created mesh is the last selected object
    #         node.name = name

    #         # Store metadata
    #         if groups:
    #             rt.setUserProp(node, "groups", ",".join(map(str, groups)))
    #         if face_ids:
    #             rt.setUserProp(node, "face_ids", ",".join(map(str, face_ids)))

            
    #         return node

    #     except Exception as e:
    #         self.log_debug(f"__create_mesh failed: {e}")
    #         import traceback
    #         self.log_debug(traceback.format_exc())
    #         return None


    def __create_mesh(self, name, verts, indices, normals, groups, face_ids):
        self.log_debug("__create_mesh called using pymxs")

        try:
            import pymxs
            rt = pymxs.runtime

            # Delete any existing objects with the same name
            existing_obj = rt.getNodeByName(name)
            if existing_obj:
                self.log_debug(f"Found existing object with name {name}, deleting it.")
                rt.delete(existing_obj)  # Delete the object from the scene

            scale_factor = self.plasticity_ui.scale_spin.value()

            # Convert numpy arrays to Python lists if necessary
            verts = verts.tolist() if isinstance(verts, np.ndarray) else verts
            indices = indices.tolist() if isinstance(indices, np.ndarray) else indices
            normals = normals.tolist() if isinstance(normals, np.ndarray) else normals

            num_verts = len(verts) // 3
            num_faces = len(indices) // 3

            # Create a new mesh
            mesh = rt.mesh()
            mesh.numverts = num_verts
            mesh.numfaces = num_faces

            # Set vertices
            for i in range(num_verts):
                x = verts[i * 3] * scale_factor
                y = verts[i * 3 + 1] * scale_factor
                z = verts[i * 3 + 2] * scale_factor
                rt.SetVert(mesh, i + 1, rt.Point3(x, y, z))

            # Set faces
            for i in range(num_faces):
                a = indices[i * 3] + 1
                b = indices[i * 3 + 1] + 1
                c = indices[i * 3 + 2] + 1
                rt.setFace(mesh, i + 1, a, b, c)
                rt.setEdgeVis(mesh, i + 1, 1, True)
                rt.setEdgeVis(mesh, i + 1, 2, True)
                rt.setEdgeVis(mesh, i + 1, 3, True)

            # Finalize the mesh
            rt.update(mesh)
            rt.redrawViews()
            node = rt.objects[-1]  # Assumes the newly created mesh is the last selected object
            node.name = name

            # Store metadata
            if groups:
                rt.setUserProp(node, "groups", ",".join(map(str, groups)))
            if face_ids:
                rt.setUserProp(node, "face_ids", ",".join(map(str, face_ids)))

            return node

        except Exception as e:
            self.log_debug(f"__create_mesh failed: {e}")
            import traceback
            self.log_debug(traceback.format_exc())
            return None



    def __update_object_and_mesh(self, obj, object_type, version, name, verts, indices, normals, groups, face_ids, plasticity_id):
        def update():
            import pymxs
            rt = pymxs.runtime

            try:
                # Define sanitized_name at the beginning of the function
                sanitized_name = self.sanitize_name(name)

                self.log_debug(f"Updating mesh for: {obj}")

                scale_factor = self.plasticity_ui.scale_spin.value()

                # Convert numpy arrays to lists if necessary
                verts_list = verts.tolist() if isinstance(verts, np.ndarray) else verts
                indices_list = indices.tolist() if isinstance(indices, np.ndarray) else indices
                normals_list = normals.tolist() if isinstance(normals, np.ndarray) else normals

                num_verts = len(verts_list) // 3
                num_faces = len(indices_list) // 3

                # Check if obj is a pymxs node; if not, retrieve by name
                node = obj
                self.log_debug(node)
                self.log_debug("node")

                if not node:
                    self.log_debug(f"Node not found: {obj}")
                    return None

                # Save existing transform and material
                transform_matrix = node.transform
                material = node.material if rt.isProperty(node, "material") else None

                # Delete the old object from the files dictionary
                if sanitized_name in self.files:
                    item_dict = self.files[sanitized_name].get(PlasticityIdUniquenessScope.ITEM, {})
                    if plasticity_id in item_dict:
                        del item_dict[plasticity_id]  # Delete the old mesh
                        self.log_debug(f"Deleted object with ID {plasticity_id} from files")
                rt.delete(node)

                # Create new mesh
                mesh = rt.mesh()
                mesh.numverts = num_verts
                mesh.numfaces = num_faces

                for i in range(num_verts):
                    x = verts_list[i * 3] * scale_factor
                    y = verts_list[i * 3 + 1] * scale_factor
                    z = verts_list[i * 3 + 2] * scale_factor
                    rt.setVert(mesh, i + 1, rt.Point3(x, y, z))

                for i in range(num_faces):
                    a = indices_list[i * 3] + 1
                    b = indices_list[i * 3 + 1] + 1
                    c = indices_list[i * 3 + 2] + 1
                    rt.setFace(mesh, i + 1, a, b, c)
                    rt.setEdgeVis(mesh, i + 1, 1, True)
                    rt.setEdgeVis(mesh, i + 1, 2, True)
                    rt.setEdgeVis(mesh, i + 1, 3, True)

                rt.update(mesh)
                rt.redrawViews()
                node = mesh

                new_node = rt.objects[-1]
                new_node.name = name
                new_node.transform = transform_matrix

                if material:
                    new_node.material = material

                # Store user data
                if groups:
                    rt.setUserProp(new_node, "groups", ",".join(map(str, groups)))
                if face_ids:
                    rt.setUserProp(new_node, "face_ids", ",".join(map(str, face_ids)))

                rt.setUserProp(new_node, "plasticity_id", plasticity_id)
                rt.setUserProp(new_node, "plasticity_filename", self.sanitize_name(name))

                # Update internal reference if tracking
                self.files[sanitized_name][PlasticityIdUniquenessScope.ITEM][parent_id] = new_node

                self.log_debug(f"Updated mesh node: {new_node.name}")

            except Exception as e:
                import traceback
                self.log_debug(f"Error in __update_object_and_mesh: {e}")
                self.log_debug(traceback.format_exc())

        # Ensure the update function runs on the main thread
        queue_main_thread(update)

    def __update_mesh_ngons(self, obj, version, faces, verts, indices, normals, groups, face_ids):
        try:
            from MaxPlus import Factory, ClassIds, Point3, TriObject, INode
            import MaxPlus

            # Get the node to update
            node = MaxPlus.INode.GetINodeByName(obj)
            if not node:
                raise ValueError(f"No valid object found: {obj}")

            # Store current material assignment
            material = node.GetMaterial()

            # Store current transform properties
            transform = node.GetWorldTM()
            pivot_pos = node.GetLocalPivot()

            # Get scale factor from UI
            scale_factor = self.plasticity_ui.scale_spin.value() / 100.0

            # Ensure input format
            if isinstance(verts, np.ndarray):
                verts = verts.reshape(-1, 3)
            if isinstance(indices, np.ndarray):
                indices = indices.tolist()
            if isinstance(normals, np.ndarray):
                normals = normals.tolist()

            # Unique verts and index remap
            unique_verts, inverse_indices = np.unique(verts, axis=0, return_inverse=True)
            new_indices = inverse_indices[indices]

            # Create new TriObject
            tri_object = Factory.CreateTriObject()
            mesh = tri_object.GetMesh()

            # Set vertices (convert from Y-up to Z-up)
            num_verts = len(unique_verts)
            mesh.SetNumVerts(num_verts)
            for i in range(num_verts):
                x = float(unique_verts[i][0]) * scale_factor
                y = float(unique_verts[i][2]) * scale_factor
                z = -float(unique_verts[i][1]) * scale_factor
                mesh.SetVert(i, Point3(x, y, z))

            # Build face connectivity
            if len(faces) == 0:
                # Triangulated case
                num_faces = len(new_indices) // 3
                mesh.SetNumFaces(num_faces)
                for i in range(num_faces):
                    a = new_indices[i*3]
                    b = new_indices[i*3+1]
                    c = new_indices[i*3+2]
                    mesh.Faces[i].SetVerts(a, b, c)
                    mesh.Faces[i].SetEdgeVisFlags(1, 1, 1)  # Make all edges visible
            else:
                # Ngon case using face grouping array
                diffs = np.where(np.diff(faces))[0] + 1
                loop_start = np.insert(diffs, 0, 0)
                loop_total = np.append(np.diff(loop_start), [len(faces) - loop_start[-1]])
                
                num_faces = len(loop_total)
                mesh.SetNumFaces(num_faces)
                
                vert_index = 0
                for i in range(num_faces):
                    face_verts = []
                    for j in range(loop_total[i]):
                        face_verts.append(new_indices[loop_start[i] + j])
                    
                    # Triangulate ngon (3ds Max requires triangles)
                    for j in range(1, len(face_verts)-1):
                        mesh.Faces[vert_index].SetVerts(face_verts[0], face_verts[j], face_verts[j+1])
                        mesh.Faces[vert_index].SetEdgeVisFlags(1, 1, 1)
                        vert_index += 1

            # Build normals if provided
            if normals and len(normals) == len(new_indices):
                mesh.SpecifyNormals()
                normals = mesh.GetSpecifiedNormals()
                normals.SetNumNormals(num_verts)
                
                # First pass: set normals for vertices
                for i in range(num_verts):
                    nx, ny, nz = normals[i]
                    normals.SetNormal(i, Point3(float(nx), float(nz), -float(ny)))
                
                # Second pass: assign normals to faces
                for i in range(num_faces):
                    face = mesh.Faces[i]
                    for j in range(3):  # Always triangles in 3ds Max
                        normals.SetNormal(face.GetVert(j), face.GetVert(j))

            # Replace the existing mesh with the new one
            node.EvalWorldState(0, True)  # Force evaluation
            obj_node = node.GetObject()
            if obj_node.CanConvertTo(ClassIds.TriObject):
                tri_obj = obj_node.ConvertToType(0, ClassIds.TriObject)
                if tri_obj:
                    tri_obj.GetMesh().DeepCopy(mesh, 0)
                    tri_obj.MeshChanged()

            # Restore material assignment
            if material:
                node.SetMaterial(material, 0)

            # Restore transform properties
            node.SetWorldTM(transform)
            node.SetLocalPivot(pivot_pos)

            # Apply smoothing
            node.GetObject().SetUseEdgeDist(True)  # Enable smoothing
            node.GetObject().SetEdgeDist(30.0)     # 30 degree angle threshold

            # Update the mesh display
            mesh.InvalidateGeomCache()
            mesh.InvalidateTopologyCache()
            mesh.BuildNormals()

            # Store custom attributes
            if self.plasticity_ui.is_triangles_mode:
                self.store_groups_and_face_ids(node, groups, face_ids, trigs=True, faces=faces)
            else:
                self.store_groups_and_face_ids(node, groups, face_ids, trigs=False, faces=faces)

            # Update pivot if needed
            self.update_pivot(node.GetName())

            return node.GetName()

        except Exception as e:
            print(f"HANDLER: __update_mesh_ngons: Failed to update ngon mesh in 3ds Max: {e}")
            return None

    def __replace_objects(self, filename, inbox_collection, version, objects):
        """
        Replaces/updates objects in 3ds Max scene using pymxs
        Handles both mesh objects and groups with proper hierarchy
        """
        self.log_debug("REPLACE*********************")
        try:
            import pymxs
            from pymxs import runtime as rt
            self.log_debug(f"BEGIN __replace_objects for {filename} (v{version}) with {len(objects)} objects")

            collections_to_unlink = set()
            sanitized_name = self.sanitize_name(filename)

            for idx, item in enumerate(objects, 1):
                try:
                    object_type = item['type']
                    name = self.sanitize_name(item['name'])
                    plasticity_id = item['id']
                    parent_id = item['parent_id']
                    flags = item['flags']

                    self.log_debug(f"[{idx}/{len(objects)}] Processing {name} (ID:{plasticity_id}, Type:{object_type})")

                    if object_type in (ObjectType.SOLID.value, ObjectType.SHEET.value):
                        # Handle mesh objects
                        if plasticity_id:
                            self.log_debug(f"Creating new mesh: {name}")
                            mesh_result = SimpleNamespace(value=None, done=False)

                            def create_mesh_safe():
                                mesh_result.value = self.__create_mesh(
                                    name,
                                    item['vertices'],
                                    item['faces'],
                                    item['normals'],
                                    item['groups'],
                                    item['face_ids']
                                )
                                mesh_result.done = True

                            queue_main_thread(create_mesh_safe)

                            timeout = 1
                            start_time = time.time()
                            while not mesh_result.done and (time.time() - start_time) < timeout:
                                QCoreApplication.processEvents()
                                time.sleep(0.01)

                            mesh = mesh_result.value

                            self.__add_object(filename, object_type, plasticity_id, name, mesh)
                        else:
                            obj = self.files[sanitized_name][PlasticityIdUniquenessScope.ITEM].get(plasticity_id)
                            if obj:
                                self.log_debug(f"Updating existing mesh: {name}")
                                update_result = SimpleNamespace(done=False)

                                def safe_update():
                                    self.__update_object_and_mesh(
                                        obj, object_type, version, name,
                                        item['vertices'], item['faces'],
                                        item['normals'], item['groups'],
                                        item['face_ids'], plasticity_id
                                    )
                                    update_result.done = True

                                queue_main_thread(safe_update)

                                # Wait for it to complete (max 10 seconds)
                                timeout = 1
                                start_time = time.time()
                                while not update_result.done and (time.time() - start_time) < timeout:
                                    QCoreApplication.processEvents()
                                    time.sleep(0.01)


                    elif object_type == ObjectType.GROUP.value and plasticity_id > 0:
                        # Handle groups
                        if plasticity_id not in self.files[sanitized_name][PlasticityIdUniquenessScope.GROUP]:
                            self.log_debug(f"Creating new group: {name}")
                            group_collection = rt.group(name=name)
                            rt.addNewUserProp(group_collection, "plasticity_id", plasticity_id)
                            rt.addNewUserProp(group_collection, "plasticity_filename", sanitized_name)
                            self.files[sanitized_name][PlasticityIdUniquenessScope.GROUP][plasticity_id] = group_collection
                        else:
                            group_collection = self.files[sanitized_name][PlasticityIdUniquenessScope.GROUP].get(plasticity_id)
                            rt.setUserProp(group_collection, "name", name)
                            collections_to_unlink.add(group_collection)

                except Exception as e:
                    self.log_debug(f"Failed to process item {idx}: {str(e)}")
                    continue

            # Handle hierarchy
            self.log_debug("Setting up object hierarchy")
            for item in objects:
                plasticity_id = item['id']
                object_type = item['type']
                self.log_debug(plasticity_id)
                self.log_debug(object_type)
                self.log_debug("PID OBJ TYP")
                if plasticity_id == 0:  # Skip root
                    continue

                uniqueness_scope = (PlasticityIdUniquenessScope.ITEM 
                                  if object_type != ObjectType.GROUP.value 
                                  else PlasticityIdUniquenessScope.GROUP)
                
                obj = self.files[sanitized_name][uniqueness_scope].get(plasticity_id)
                self.log_debug(obj)
                self.log_debug("OOOOOOOOOOOO&&&&&&&&&&&&&&")
                if not obj:
                    self.log_debug(f"Skipping hierarchy for missing object ID {plasticity_id}")
                    continue

                parent_id = item['parent_id']
                parent = (inbox_collection if parent_id == 0 
                         else self.files[sanitized_name][PlasticityIdUniquenessScope.GROUP].get(parent_id))
                
                if not parent:
                    self.log_debug(f"No parent found for {obj.name} (parent ID:{parent_id})")
                    continue

                try:
                    if parent is not None:
                        def assign_parent(obj=obj, parent=parent):
                            try:
                                obj.parent = parent
                                self.log_debug(f"Set parent of {obj.name} to {parent.name}")
                            except Exception as e:
                                self.log_debug(f"Parenting failed for {obj.name}: {e}")

                        queue_main_thread(assign_parent)



                    
                    def assign_flags():
                        try:
                            rt.setProperty(obj, "isHidden", bool(item['flags'] & 1))
                            rt.setProperty(obj, "renderable", not bool(item['flags'] & 2))
                            rt.setProperty(obj, "isFrozen", not bool(item['flags'] & 4))
                        except Exception as e:
                            self.log_debug(f"Flag assignment failed for {obj.name}: {e}")

                    queue_main_thread(assign_flags)


                except Exception as e:
                    self.log_debug(f"Hierarchy failed for {obj.name}: {str(e)}")

            self.log_debug(f"COMPLETED __replace_objects for {filename}")
            return True

        except Exception as e:
            self.log_debug(f"CRITICAL ERROR in __replace_objects: {str(e)}")
            import traceback
            self.log_debug(f"Traceback:\n{traceback.format_exc()}")
            return False

    def __add_object(self, filename, object_type, plasticity_id, name, mesh_node):
        """
        Adds a created mesh to 3ds Max's scene and stores it in the tracking dictionary.
        """
        self.log_debug("ADDING OBJECTS")
        try:
            import pymxs
            rt = pymxs.runtime

            sanitized_name = self.sanitize_name(filename)

            if not mesh_node:
                self.log_debug(f"Warning: Mesh creation failed for {name}. Skipping add_object.")
                return None

            # Store the node for future lookup (e.g. for parenting)
            self.files[sanitized_name][PlasticityIdUniquenessScope.ITEM][plasticity_id] = mesh_node

            # Schedule tagging on main thread
            def tag_node():
                try:
                    mesh_node.name = name
                    rt.setUserProp(mesh_node, "plasticity_id", plasticity_id)
                    rt.setUserProp(mesh_node, "plasticity_filename", sanitized_name)
                    self.log_debug(f"Tagged object {name} with PID {plasticity_id}")
                except Exception as e:
                    self.log_debug(f"Failed to tag object {name}: {e}")

            queue_main_thread(tag_node)
            return mesh_node

        except Exception as e:
            self.log_debug(f"HANDLER: __add_object: Failed to add object in 3ds Max: {e}")
            import traceback
            self.log_debug(traceback.format_exc())
            return None

    def __inbox_for_filename(self, filename):
        result = SimpleNamespace(value=None, done=False)

        def create_collection():
            try:
                import pymxs
                rt = pymxs.runtime
                sanitized_name = self.sanitize_name(filename)

                self.log_debug(f"Creating collections for: {sanitized_name}")

                def create_temp_point_helper():
                    point_helper = rt.point()
                    point_helper.name = "__temp__"
                    point_helper.isHidden = True
                    point_helper.renderable = False
                    point_helper.isFrozen = True
                    return point_helper

                root = rt.getNodeByName("Plasticity")
                if not root:
                    temp_point_helper = create_temp_point_helper()
                    rt.select(temp_point_helper)
                    root = rt.group(rt.selection)
                    root.name = "Plasticity"
                    rt.setUserProp(root, "plasticity_root", True)
                    self.log_debug("Created group: Plasticity")
                else:
                    self.log_debug("Found existing group: Plasticity")

                file_group = rt.getNodeByName(sanitized_name)
                if not file_group:
                    temp_point_helper = create_temp_point_helper()
                    rt.select(temp_point_helper)
                    file_group = rt.group(rt.selection)
                    file_group.name = sanitized_name
                    file_group.parent = root
                    rt.setUserProp(file_group, "plasticity_file", True)
                    self.log_debug(f"Created group: {sanitized_name}")
                else:
                    self.log_debug(f"Found existing group: {sanitized_name}")

                inbox = None
                for child in file_group.Children:
                    if child.name == "Inbox":
                        inbox = child
                        break

                if not inbox:
                    temp_point_helper = create_temp_point_helper()
                    rt.select(temp_point_helper)
                    inbox = rt.group(rt.selection)
                    inbox.name = "Inbox"
                    inbox.parent = file_group
                    rt.setUserProp(inbox, "plasticity_inbox", True)
                    self.log_debug("Created group: Inbox")
                else:
                    self.log_debug("Found existing group: Inbox")

                result.value = inbox
                result.done = True
                self.log_debug(">>> result assigned successfully")

            except Exception as e:
                import traceback
                self.log_debug(f"Exception in create_collection: {e}\n{traceback.format_exc()}")
                result.done = True


        # Queue main-thread pymxs call
        queue_main_thread(create_collection)

        # Explicitly wait until done is set to True
        timeout = 1.0  # allow up to 5 seconds
        start_time = time.time()
        while not result.done and (time.time() - start_time) < timeout:
            QCoreApplication.processEvents()
            time.sleep(0.01)

        if not result.done or result.value is None:
            raise RuntimeError("Timed out waiting for collection creation")

        return result.value


    def __prepare(self, filename):
        result = SimpleNamespace(value=None, done=False)

        def prepare_and_create_collection():
            try:
                import pymxs
                rt = pymxs.runtime
                sanitized_name = self.sanitize_name(filename)

                self.log_debug(f"Preparing scene for {sanitized_name}")

                # Create Root Group
                root = rt.getNodeByName("Plasticity")
                if not root:
                    temp_point_helper = rt.point()  # Create a Point helper instead of a dummy
                    temp_point_helper.name = "__temp__"
                    temp_point_helper.isHidden = True
                    temp_point_helper.renderable = False
                    rt.select(temp_point_helper)
                    root = rt.group(rt.selection)
                    root.name = "Plasticity"
                    rt.setUserProp(root, "plasticity_root", True)
                    self.log_debug("Created group: Plasticity")
                else:
                    self.log_debug("Found existing group: Plasticity")

                # Create File Group
                file_group = rt.getNodeByName(sanitized_name)
                if not file_group:
                    temp_point_helper = rt.point()  # Create a Point helper instead of a dummy
                    temp_point_helper.name = "__temp__"
                    temp_point_helper.isHidden = True
                    temp_point_helper.renderable = False
                    rt.select(temp_point_helper)
                    file_group = rt.group(rt.selection)
                    file_group.name = sanitized_name
                    file_group.parent = root
                    rt.setUserProp(file_group, "plasticity_file", True)
                    self.log_debug(f"Created group: {sanitized_name}")
                else:
                    self.log_debug(f"Found existing group: {sanitized_name}")

                # Create Inbox Group
                inbox = None
                for child in file_group.Children:
                    if child.name == "Inbox":
                        inbox = child
                        break

                if not inbox:
                    temp_point_helper = rt.point()  # Create a Point helper instead of a dummy
                    temp_point_helper.name = "__temp__"
                    temp_point_helper.isHidden = True
                    temp_point_helper.renderable = False
                    rt.select(temp_point_helper)
                    inbox = rt.group(rt.selection)
                    inbox.name = "Inbox"
                    inbox.parent = file_group
                    rt.setUserProp(inbox, "plasticity_inbox", True)
                    self.log_debug("Created group: Inbox")
                else:
                    self.log_debug("Found existing group: Inbox")

                # Gather existing objects
                objects, collections = [], []
                children = list(inbox.Children)
                for child in children:
                    if rt.classOf(child) in (rt.Editable_Mesh, rt.PolyMeshObject):
                        objects.append(child)
                    elif rt.isGroupHead(child):
                        collections.append(child)

                existing_objects = {
                    PlasticityIdUniquenessScope.ITEM: {},
                    PlasticityIdUniquenessScope.GROUP: {}
                }

                for obj in objects:
                    pid = rt.getUserProp(obj, "plasticity_id")
                    if pid is not None:
                        existing_objects[PlasticityIdUniquenessScope.ITEM][pid] = obj

                for group in collections:
                    pid = rt.getUserProp(group, "plasticity_id")
                    if pid is not None:
                        existing_objects[PlasticityIdUniquenessScope.GROUP][pid] = group

                self.files[sanitized_name] = existing_objects
                result.value = inbox
                result.done = True
                self.log_debug(f"Found {len(objects)} objects and {len(collections)} groups")

            except Exception as e:
                import traceback
                self.log_debug(f"Exception in prepare_and_create_collection: {e}\n{traceback.format_exc()}")
                result.done = True


        # Queue this single combined function once
        queue_main_thread(prepare_and_create_collection)

        # Wait explicitly for completion
        timeout = 10.0
        start_time = time.time()
        while not result.done and (time.time() - start_time) < timeout:
            QCoreApplication.processEvents()
            time.sleep(0.05)

        if not result.done or result.value is None:
            raise RuntimeError("Timed out waiting for scene preparation")

        return result.value

    def on_transaction(self, transaction):
        """
        Processes a transaction by updating the scene with new, updated, or deleted objects.
        """
        self.log_debug("HANDLER ON TRANSACTION")
        try:
            # Using pymxs for scene updates in 3ds Max
            import pymxs
            from pymxs import runtime as rt

            filename = transaction["filename"]
            version = transaction["version"]

            # Report transaction update
            print(f"Updating {filename} to version {version}")

            # Prepare scene for updates (equivalent to the original MaxPlus scene preparation)
            inbox_collection = self.__prepare(filename)

            # Handle object deletion
            if "delete" in transaction:
                self.log_debug("DDDDEEEELLLLLLLLLEEEEEEEEEEEETTTTTTTTTTTEEEEEEEEEEE&&&&&&&")
                for plasticity_id in transaction["delete"]:
                    self.__delete_object(filename, version, plasticity_id)

            # Handle object addition
            if "add" in transaction:
                self.__replace_objects(filename, inbox_collection, version, transaction["add"])

            # Handle object updates
            if "update" in transaction:
                self.__replace_objects(filename, inbox_collection, version, transaction["update"])

            # Ensure the scene is updated properly
            self.log_debug(f"=== Completed update for {filename} ===")
            self.plasticity_ui.console_output.append(f"Updated {filename} to v{version}")

        except Exception as e:
            error_msg = f"HANDLER: on_transaction: Failed to process transaction in 3ds Max: {e}"
            self.plasticity_ui.console_output.append(error_msg)
            self.log_debug(error_msg)
            import traceback
            self.log_debug(f"Traceback:\n{traceback.format_exc()}")

    def on_list(self, message):
        """
        Processes full scene list updates in 3ds Max without undo blocks
        Includes comprehensive debug logging and error handling
        """
        try:
            import pymxs
            from pymxs import runtime as rt
            from mainthread import queue_main_thread

            # Extract and validate transaction data
            filename = message.get("filename", "unknown_file")
            version = message.get("version", 0)
            self.log_debug(filename)
            sanitized_name = self.sanitize_name(filename)
            self.log_debug(sanitized_name)
            self.log_debug(f"=== Starting scene update for {filename} v{version} ===")
            self.log_debug(f"Transaction keys: {list(message.keys())}")

            # Prepare scene infrastructure on main thread
            inbox_collection = self.__prepare(sanitized_name)
            if not inbox_collection:
                raise ValueError(f"Failed to initialize collection for {sanitized_name}")


            # Process object additions/updates
            current_items = set()
            current_groups = set()
            
            if "add" in message:
                add_ops = message["add"]
                self.log_debug(f"Processing {len(add_ops)} additions/updates")
                
                for idx, item in enumerate(add_ops, 1):
                    item_id = item.get("id", "N/A")
                    if item["type"] == ObjectType.GROUP.value:
                        current_groups.add(item_id)
                        self.log_debug(f"[{idx}/{len(add_ops)}] Group {item_id}")
                    else:
                        current_items.add(item_id)
                        self.log_debug(f"[{idx}/{len(add_ops)}] Object {item_id}")
                
                self.__replace_objects(filename, inbox_collection, version, add_ops)
                self.log_debug("PAST REPLACE")

            # Process deletions with existence checks
            if sanitized_name in self.files:
                file_data = self.files[sanitized_name]
                self.log_debug("33333333333333333333333333 probably never runs this??")
                # Object deletions
                obsolete_items = [
                    pid for pid in file_data[PlasticityIdUniquenessScope.ITEM]
                    if pid not in current_items
                ]
                self.log_debug(f"Found {len(obsolete_items)} obsolete objects")
                
                for pid in obsolete_items:
                    self.log_debug(f"Deleting object {pid}")
                    self.__delete_object(sanitized_name, version, pid)

                # Group deletions
                obsolete_groups = [
                    pid for pid in file_data[PlasticityIdUniquenessScope.GROUP]
                    if pid not in current_groups
                ]
                self.log_debug(f"Found {len(obsolete_groups)} obsolete groups")
                
                for pid in obsolete_groups:
                    self.log_debug(f"Deleting group {pid}")
                    self.__delete_group(sanitized_name, version, pid)

            # Final cleanup
            # rt.clearSelection()
            self.log_debug(f"=== Completed update for {filename} ===")
            self.plasticity_ui.console_output.append(f"Updated {filename} to v{version}")

        except Exception as e:
            error_msg = f"Scene update failed: {str(e)}"
            self.plasticity_ui.console_output.append(error_msg)
            self.log_debug(error_msg)
            import traceback
            self.log_debug(f"Traceback:\n{traceback.format_exc()}")


    def log_handler_debug(self, message):
        """Logs handler-specific debug messages to the same log file as log_debug."""
        try:
            # Use the same debug log path as log_debug
            log_path = self.debug_log_path
            
            # Open the log file and append the new message
            with open(log_path, "a", encoding="utf-8") as f:
                timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
                log_entry = f"[{timestamp}] {message}\n"
                f.write(log_entry)
        except Exception as e:
            print(f"Failed to write to handler debug log: {e}")

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

    def __delete_object(self, filename, version, plasticity_id):
        """
        Deletes an object from the 3ds Max scene by its plasticity_id.
        Ensures that the delete operation runs on the main thread.
        """
        sanitized_name = self.sanitize_name(filename)
        
        # Retrieve the object using the plasticity_id from the files dictionary
        obj = self.files[sanitized_name][PlasticityIdUniquenessScope.ITEM].pop(plasticity_id, None)
        
        if obj:
            self.log_debug(f"Attempting to delete object: {obj.name}")

            def delete_object_on_main_thread():
                try:
                    import pymxs
                    rt = pymxs.runtime

                    # Check if the object exists in the scene and delete it
                    if rt.isValidNode(obj):
                        rt.delete(obj)  # Delete the object from the scene
                        self.log_debug(f"Deleted object {obj.name}")
                    else:
                        self.log_debug(f"Object {obj.name} no longer exists in the scene.")
                except Exception as e:
                    self.log_debug(f"Error deleting object {obj.name}: {e}")

            # Schedule the deletion to run on the main thread
            queue_main_thread(delete_object_on_main_thread)


    def __delete_group(self, filename, version, plasticity_id):
        """
        Deletes a group from the 3ds Max scene by its plasticity_id.
        Ensures that the delete operation runs on the main thread.
        """
        sanitized_name = self.sanitize_name(filename)
        
        # Retrieve the group using the plasticity_id from the files dictionary
        group = self.files[sanitized_name][PlasticityIdUniquenessScope.GROUP].pop(plasticity_id, None)
        
        if group:
            self.log_debug(f"Attempting to delete group: {group.name}")

            def delete_group_on_main_thread():
                try:
                    import pymxs
                    rt = pymxs.runtime

                    # Check if the group exists in the scene and delete it
                    if rt.isValidNode(group):
                        rt.delete(group)  # Delete the group from the scene
                        self.log_debug(f"Deleted group {group.name}")
                    else:
                        self.log_debug(f"Group {group.name} no longer exists in the scene.")
                except Exception as e:
                    self.log_debug(f"Error deleting group {group.name}: {e}")

            # Schedule the deletion to run on the main thread
            queue_main_thread(delete_group_on_main_thread)
