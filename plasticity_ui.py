# plasticity_ui_max.py
# exec(open(r"C:\3dsmax_dev\PlasticityLiveLink_Max\dev_reload.py").read())
try:
    from PySide6 import QtWidgets, QtGui, QtCore
except ImportError:
    from PySide2 import QtWidgets, QtGui, QtCore

from client import PlasticityClient
from handler import SceneHandler
from mainthread import main_thread_queue


class PlasticityUI(QtWidgets.QWidget):
    def __init__(self, parent=None):
        # Get 3ds Max main window if no parent provided
        if parent is None:
            parent = self.get_max_main_window()
            
        super().__init__(parent)
        
        # Setup the core UI
        self.setup_ui()
        
        # Create dock widget if we have a proper parent
        if parent and isinstance(parent, QtWidgets.QMainWindow):
            self.dock = QtWidgets.QDockWidget("Plasticity", parent)
            self.dock.setObjectName("PlasticityDock")
            self.dock.setWidget(self)
            
            # Configure docking features
            self.dock.setFeatures(
                QtWidgets.QDockWidget.DockWidgetFloatable |
                QtWidgets.QDockWidget.DockWidgetMovable |
                QtWidgets.QDockWidget.DockWidgetClosable
            )
            
            # Add to main window's docking system
            parent.addDockWidget(QtCore.Qt.RightDockWidgetArea, self.dock)
            self.dock.show()
        else:
            # Fallback to floating window
            self.setWindowFlags(QtCore.Qt.Window)
            self.show()

        # Start queue timer
        self.queue_timer = QtCore.QTimer(self)
        self.queue_timer.timeout.connect(self.process_main_thread_queue)
        self.queue_timer.start(50)

    def get_max_main_window(self):
        """Find the 3ds Max main window reliably"""
        try:
            # Method 1: Get the main window through MaxPlus (for older 3ds Max versions)
            try:
                import MaxPlus
                return MaxPlus.GetQMaxMainWindow()
            except:
                pass
            
            # Method 2: Try pymxs GetQMaxMainWindow (newer versions)
            try:
                from pymxs import runtime as rt
                if hasattr(rt, 'GetQMaxMainWindow'):
                    return rt.GetQMaxMainWindow()
            except:
                pass
            
            # Method 3: Search through existing widgets
            app = QtWidgets.QApplication.instance()
            for widget in app.topLevelWidgets():
                if widget.objectName() == 'QMaxMainWindow':
                    return widget
                if '3ds Max' in widget.windowTitle():
                    return widget
            
            return None
            
        except Exception as e:
            print(f"Could not find 3ds Max main window: {e}")
            return None

    def setup_ui(self):
        """Setup all the UI components (moved from __init__)"""
        self.setObjectName("PlasticityUI")
        self.setWindowTitle("Plasticity")
        self.setMinimumWidth(270)
        self.setMinimumHeight(1000)

        self.handler = SceneHandler(self)
        self.client = PlasticityClient(self.handler)

        self.status_label = QtWidgets.QLabel("Status: Disconnected")

        main_layout = QtWidgets.QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)

        scroll_area = QtWidgets.QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setFrameShape(QtWidgets.QFrame.NoFrame)
        main_layout.addWidget(scroll_area)

        content_widget = QtWidgets.QWidget()
        content_widget.setMinimumWidth(270)
        scroll_area.setWidget(content_widget)

        layout = QtWidgets.QVBoxLayout(content_widget)
        layout.setContentsMargins(5, 5, 5, 5)

        # Connection Group
        self.connection_box = QtWidgets.QGroupBox()
        connection_layout = QtWidgets.QVBoxLayout(self.connection_box)
        connection_layout.setContentsMargins(12, 26, 12, 12)
        connection_layout.setSpacing(4)

        self.connect_btn = QtWidgets.QPushButton("Connect")
        self.connect_btn.setCheckable(True)
        self.connect_btn.setFixedHeight(32)

        self.server_field = QtWidgets.QLineEdit("localhost:8980")
        self.server_field.setFixedHeight(22)

        shared_width_container = QtWidgets.QWidget()
        shared_layout = QtWidgets.QVBoxLayout(shared_width_container)
        shared_layout.setContentsMargins(0, 0, 0, 0)
        shared_layout.setSpacing(4)
        shared_layout.setAlignment(QtCore.Qt.AlignHCenter)

        shared_layout.addWidget(self.connect_btn)
        shared_layout.addWidget(self.server_field)

        uniform_width = 180
        self.connect_btn.setFixedWidth(uniform_width)
        self.server_field.setFixedWidth(uniform_width)

        connection_layout.addWidget(shared_width_container, alignment=QtCore.Qt.AlignHCenter)
        layout.addWidget(self.connection_box)

        # Refresh + Live Link Group
        self.refresh_live_box = QtWidgets.QGroupBox()
        merged_layout = QtWidgets.QVBoxLayout(self.refresh_live_box)
        merged_layout.setContentsMargins(0, 0, 0, 0)
        merged_layout.setSpacing(8)

        top_row = QtWidgets.QHBoxLayout()
        top_row.setSpacing(0)

        self.refresh_btn = QtWidgets.QPushButton("Refresh")
        self.refresh_btn.setFixedSize(120, 40)

        self.live_btn = QtWidgets.QPushButton("Live Link")
        self.live_btn.setCheckable(True)
        self.live_btn.setFixedSize(120, 40)

        top_row.addWidget(self.refresh_btn)
        top_row.addWidget(self.live_btn)

        bottom_settings_frame = QtWidgets.QFrame()
        bottom_layout = QtWidgets.QHBoxLayout(bottom_settings_frame)
        bottom_layout.setContentsMargins(8, 8, 8, 8)
        bottom_layout.setSpacing(12)

        left_col = QtWidgets.QVBoxLayout()
        self.pid_suffix_check = QtWidgets.QCheckBox("PID Suffix")
        self.pid_suffix_check.setChecked(True)
        self.visible_check = QtWidgets.QCheckBox("Only visible")
        self.visible_check.setChecked(True)
        left_col.addWidget(self.pid_suffix_check)
        left_col.addWidget(self.visible_check)
        left_col.addStretch()

        right_col = QtWidgets.QVBoxLayout()
        scale_label = QtWidgets.QLabel("Scale")
        self.scale_spin = QtWidgets.QSpinBox()
        self.scale_spin.setRange(1, 10000)
        self.scale_spin.setValue(100)
        self.scale_spin.setFixedWidth(100)
        self.scale_spin.setFixedHeight(20)
        right_col.addWidget(scale_label)
        right_col.addWidget(self.scale_spin)
        right_col.addStretch()

        bottom_layout.addLayout(left_col)
        bottom_layout.addLayout(right_col)

        merged_layout.addLayout(top_row)
        merged_layout.addWidget(bottom_settings_frame)
        layout.addWidget(self.refresh_live_box)

        # Refacet Group
        self.refacet_box = QtWidgets.QGroupBox()
        refacet_layout = QtWidgets.QVBoxLayout(self.refacet_box)
        refacet_layout.setContentsMargins(8, 8, 8, 8)
        refacet_layout.setSpacing(8)

        refacet_row = QtWidgets.QHBoxLayout()
        self.refacet_btn = QtWidgets.QPushButton("Refacet")
        self.refacet_btn.setFixedSize(120, 40)
        refacet_row.addWidget(self.refacet_btn)

        facet_radio_col = QtWidgets.QVBoxLayout()
        self.ngon_radio = QtWidgets.QRadioButton("Ngons")
        self.ngon_radio.setChecked(True)
        self.tri_radio = QtWidgets.QRadioButton("Triangles")
        facet_radio_col.addWidget(self.ngon_radio)
        facet_radio_col.addWidget(self.tri_radio)
        refacet_row.addLayout(facet_radio_col)
        refacet_row.addStretch()

        refacet_layout.addLayout(refacet_row)

        self.refacet_tabs = QtWidgets.QTabWidget()

        self.basic_tab = QtWidgets.QWidget()
        basic_form = QtWidgets.QFormLayout(self.basic_tab)
        self.tolerance_spin = QtWidgets.QDoubleSpinBox()
        self.tolerance_spin.setRange(0, 1)
        self.tolerance_spin.setDecimals(5)
        self.tolerance_spin.setValue(0.01)
        basic_form.addRow("Tolerance", self.tolerance_spin)
        self.angle_spin = QtWidgets.QDoubleSpinBox()
        self.angle_spin.setRange(0, 1)
        self.angle_spin.setDecimals(5)
        self.angle_spin.setValue(0.2)
        basic_form.addRow("Angle", self.angle_spin)

        self.advanced_tab = QtWidgets.QWidget()
        adv_form = QtWidgets.QFormLayout(self.advanced_tab)

        self.min_width_spin = QtWidgets.QDoubleSpinBox()
        self.min_width_spin.setRange(0, 1)
        self.min_width_spin.setValue(0.0)
        adv_form.addRow("Min Width", self.min_width_spin)

        self.max_width_spin = QtWidgets.QDoubleSpinBox()
        self.max_width_spin.setRange(0, 10)
        self.max_width_spin.setValue(0.0)
        adv_form.addRow("Max Width", self.max_width_spin)

        self.edge_tol_spin = QtWidgets.QDoubleSpinBox()
        self.edge_tol_spin.setRange(0, 1)
        self.edge_tol_spin.setDecimals(4)
        self.edge_tol_spin.setValue(0.01)
        adv_form.addRow("Edge Chord Tol", self.edge_tol_spin)

        self.edge_angle_spin = QtWidgets.QDoubleSpinBox()
        self.edge_angle_spin.setRange(0, 1)
        self.edge_angle_spin.setDecimals(4)
        self.edge_angle_spin.setValue(0.25)
        adv_form.addRow("Edge Angle Tol", self.edge_angle_spin)

        self.face_tol_spin = QtWidgets.QDoubleSpinBox()
        self.face_tol_spin.setRange(0, 1)
        self.face_tol_spin.setDecimals(4)
        self.face_tol_spin.setValue(0.01)
        adv_form.addRow("Face Plane Tol", self.face_tol_spin)

        self.face_angle_spin = QtWidgets.QDoubleSpinBox()
        self.face_angle_spin.setRange(0, 1)
        self.face_angle_spin.setDecimals(4)
        self.face_angle_spin.setValue(0.25)
        adv_form.addRow("Face Angle Tol", self.face_angle_spin)


        self.refacet_tabs.addTab(self.basic_tab, "Basic")
        self.refacet_tabs.addTab(self.advanced_tab, "Advanced")

        refacet_layout.addWidget(self.refacet_tabs)
        layout.addWidget(self.refacet_box)

        # Utilities Group
        self.util_box = QtWidgets.QGroupBox("Utilities")
        util_layout = QtWidgets.QVBoxLayout(self.util_box)
        self.util_buttons = {}
        for label in [
            "Auto UV Layout",
            "Cut/Sew UV Seams",
            "Select Plasticity Face(s)",
            "Select Plasticity Edges",
            "Paint Plasticity Faces"]:
            btn = QtWidgets.QPushButton(label)
            self.util_buttons[label] = btn
            util_layout.addWidget(btn)

        layout.addWidget(self.util_box)

        # Console Output
        self.console_output = QtWidgets.QTextEdit()
        self.console_output.setReadOnly(True)
        self.console_output.setFixedHeight(120)
        layout.addWidget(self.console_output)

        content_widget.setLayout(layout)

        # --- Connect Signals ---
        self.connect_btn.clicked.connect(self.toggle_connection)
        self.live_btn.clicked.connect(self.toggle_live_link)
        self.refresh_btn.clicked.connect(self.on_refresh_clicked)
        self.refacet_btn.clicked.connect(self.on_refacet_clicked)
        self.util_buttons["Auto UV Layout"].clicked.connect(self.auto_uv_layout)
        self.util_buttons["Cut/Sew UV Seams"].clicked.connect(self.merge_uv_seams)
        self.util_buttons["Select Plasticity Face(s)"].clicked.connect(self.select_plasticity_faces)
        self.util_buttons["Select Plasticity Edges"].clicked.connect(self.select_plasticity_edges)
        self.util_buttons["Paint Plasticity Faces"].clicked.connect(self.paint_plasticity_faces)

        self.set_buttons_active(False)   # disables all tool buttons
        self.live_btn.setEnabled(False)  # disables live link
        self.connect_btn.setEnabled(True)  # connect is always clickable
        self.server_field.setEnabled(True)


    def on_dock_close(self):
        """Handle dock widget being closed"""
        self.deleteLater()
        self.dock = None

    # Placeholder logic handlers
    def toggle_connection(self):
        if self.connect_btn.isChecked():
            self.connect_btn.setText("Connecting...")
            self.connect_btn.setEnabled(False)
            self.server_field.setEnabled(False)
            self.execute_connect()
        else:
            self.execute_disconnect()

    def execute_connect(self):
        server = self.server_field.text().strip()
        if server:
            try:
                self.client.connect(server)
                # Set a timer to check connection status
                QtCore.QTimer.singleShot(3000, self.check_connection_status)
            except Exception as e:
                self.console_output.append(f"Connection error: {e}")
                self.update_ui_disconnected()

    def check_connection_status(self):
        if not self.client.connected:
            self.console_output.append("Connection failed - check if server is running")
            self.update_ui_disconnected()

    def execute_disconnect(self):
        """Safe disconnection handling"""
        try:
            # Use try/except instead of sip checks
            try:
                self.connect_btn.setEnabled(False)
                self.connect_btn.setText("Disconnecting...")
            except RuntimeError:
                # UI elements already deleted
                return

            # Perform actual disconnection
            if hasattr(self.client, 'disconnect'):
                self.client.disconnect()
            
        except Exception as e:
            print(f"Disconnect error: {e}")
        finally:
            self._safe_ui_update()

    def _safe_ui_update(self):
        """Thread-safe UI state update with error handling"""
        def update():
            try:
                self.connect_btn.setEnabled(True)
                self.connect_btn.setChecked(False)
                self.connect_btn.setText("Connect")
                self.server_field.setEnabled(True)
            except RuntimeError:
                # Widgets already deleted, ignore
                pass
            except Exception as e:
                print(f"UI update error: {e}")

        # Only schedule if the UI still exists
        if hasattr(self, 'connect_btn'):
            QtCore.QTimer.singleShot(0, update)       

    def set_buttons_active(self, active):
        """Top-level groups — disabling these disables all child widgets"""
        self.refresh_live_box.setEnabled(active)
        self.refacet_box.setEnabled(active)
        self.util_box.setEnabled(active)
        self.server_field.setEnabled(not active)

    def update_ui_disconnected(self):
        self.connected = False
        self.connect_btn.setEnabled(True)
        self.connect_btn.setChecked(False)
        self.connect_btn.setText("Connect")
        self.server_field.setEnabled(True)
        self.live_btn.setEnabled(False)
        self.set_buttons_active(False)


    def update_ui_connected(self):
        self.connected = True
        self.connect_btn.setEnabled(True)
        self.connect_btn.setChecked(True)
        self.connect_btn.setText("Disconnect")
        self.server_field.setEnabled(False)
        self.live_btn.setEnabled(True)
        self.set_buttons_active(True)


        
    def toggle_live_link(self):
        """Toggle the live link state and update the button text."""
        print("Toggle Live Link")
        if self.live_btn.isChecked():
            self.live_btn.setText("End Link")
            self.execute_live_link_activate()
        else:
            self.live_btn.setText("Live Link")
            self.execute_live_link_deactivate()

    def execute_live_link_activate(self):
        """Execute the logic to activate live link."""
        print("Live Link activated...")
        self.client.subscribe_all()
        return

    def execute_live_link_deactivate(self):
        """Execute the logic to deactivate live link."""
        print("Live Link deactivated...")
        self.client.unsubscribe_all()
        return

    def on_refresh_clicked(self):
        print("Refreshing...")
        only_visible = self.visible_check.isChecked()
        if only_visible:
            print("visible")
            self.client.list_visible()
        else:
            print("all")
            self.client.list_all()

    def on_refacet_clicked(self):
        print("Refacet Clicked")

    def auto_uv_layout(self):
        print("Auto UV Layout")

    def merge_uv_seams(self):
        print("Merge UV Seams")

    def select_plasticity_faces(self):
        print("Select Plasticity Faces")

    def select_plasticity_edges(self):
        print("Select Plasticity Edges")

    def paint_plasticity_faces(self):
        print("Paint Plasticity Faces")


    @QtCore.Slot(object)
    def execute_in_main_thread(self, func):
        try:
            print("Executing on main thread:", func)
            func()
        except Exception as e:
            print("Error in main thread execution:", e)



    def process_main_thread_queue(self):
        while not main_thread_queue.empty():
            try:
                func = main_thread_queue.get_nowait()
                func()
            except Exception as e:
                print(f"Failed to execute queued function: {e}")

