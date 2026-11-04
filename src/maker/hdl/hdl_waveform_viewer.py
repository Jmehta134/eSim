"""HDL VCD Waveform Viewer & Digital Timing Analyzer.

Provides application-aware light/dark theme support, VCD parsing, bus line 
rendering, and horizontal-only panning/zooming toolbar for eSim Verilog 
simulation outputs.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict

import matplotlib.pyplot as plt
from matplotlib.backends.backend_qtagg import (
    FigureCanvasQTAgg,
    NavigationToolbar2QT,
)
from matplotlib.figure import Figure
from matplotlib.patches import Polygon
import numpy as np
from PyQt6 import QtCore, QtGui, QtWidgets
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QScrollArea,
    QSplitter,
    QStatusBar,
    QVBoxLayout,
    QWidget,
)

from configuration.Appconfig import Appconfig

# Dual-fallback imports for shared eSim plotting components
try:
    from ngspiceSimulation._palette import current_palette, matplotlib_rc_overrides
    from ngspiceSimulation.constants import DEFAULT_DPI, DEFAULT_FIGURE_SIZE, VIBRANT_COLOR_PALETTE
    from ngspiceSimulation.trace import Trace, CustomListWidget
except ImportError:
    try:
        from waveWindow._palette import current_palette, matplotlib_rc_overrides
        from waveWindow.constants import DEFAULT_DPI, DEFAULT_FIGURE_SIZE, VIBRANT_COLOR_PALETTE
        from waveWindow.trace import Trace, CustomListWidget
    except ImportError:
        from ._palette import current_palette, matplotlib_rc_overrides
        from .constants import DEFAULT_DPI, DEFAULT_FIGURE_SIZE, VIBRANT_COLOR_PALETTE
        from .trace import Trace, CustomListWidget

logger = logging.getLogger(__name__)

# =============================================================
# FALLBACK THEME COLORS
# =============================================================

DARK_FALLBACK = {
    "window": "#292D33",
    "panel": "#32373E",
    "panel_light": "#3A4048",
    "wave": "#24282E",
    "grid": "#41474F",
    "border": "#505760",
    "text": "#DDE1E5",
    "muted": "#A0A8B2",
    "signal_text": "#D5DAE0",
    "signal_panel": "#30353C",
    "input": "#252A30",
    "input_text": "#E0E3E7",
    "selection": "#555F6B",
    "selection_text": "#FFFFFF",
    "button": "#444B54",
    "button_hover": "#535D68",
    "unknown": "#D19B70",
}

LIGHT_FALLBACK = {
    "window": "#F0F2F5",
    "panel": "#FFFFFF",
    "panel_light": "#F5F6F8",
    "wave": "#FFFFFF",
    "grid": "#DCE1E7",
    "border": "#C7CDD5",
    "text": "#252A32",
    "muted": "#707984",
    "signal_text": "#303741",
    "signal_panel": "#F5F6F8",
    "input": "#FFFFFF",
    "input_text": "#252A32",
    "selection": "#DCE7F5",
    "selection_text": "#202A38",
    "button": "#E8EBF0",
    "button_hover": "#DCE1E8",
    "unknown": "#B66A27",
}


# =============================================================
# HORIZONTAL-ONLY NAVIGATION TOOLBAR
# =============================================================


class HorizontalNavigationToolbar(NavigationToolbar2QT):
    """Matplotlib toolbar restricted to horizontal axis panning and zooming."""

    def __init__(self, canvas, parent=None):
        super().__init__(canvas, parent)
        self._vertical_limits = {}
        self._locking_vertical = False

    def _save_vertical_limits(self):
        self._vertical_limits = {
            ax: ax.get_ylim() for ax in self.canvas.figure.axes
        }

    def _restore_vertical_limits(self):
        if self._locking_vertical:
            return
        self._locking_vertical = True
        try:
            for ax, limits in self._vertical_limits.items():
                if ax in self.canvas.figure.axes:
                    ax.set_ylim(limits)
        finally:
            self._locking_vertical = False

    def press_pan(self, event):
        self._save_vertical_limits()
        super().press_pan(event)

    def mouse_move(self, event):
        super().mouse_move(event)
        if self._vertical_limits:
            self._restore_vertical_limits()

    def release_pan(self, event):
        super().release_pan(event)
        self._restore_vertical_limits()
        self.canvas.draw_idle()

    def press_zoom(self, event):
        self._save_vertical_limits()
        super().press_zoom(event)

    def release_zoom(self, event):
        super().release_zoom(event)
        self._restore_vertical_limits()
        self.canvas.draw_idle()


# =============================================================
# MATPLOTLIB CANVAS
# =============================================================


class WaveFigureCanvas(FigureCanvasQTAgg):
    """Qt Widget canvas embedding Matplotlib plots."""

    def __init__(self, figure):
        super().__init__(figure)
        self.setObjectName("WaveCanvas")
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self.setAutoFillBackground(True)


# =============================================================
# MAIN HDL WAVEFORM VIEWER WINDOW
# =============================================================


class waveWindow(QWidget):
    """HDL Digital Waveform Viewer Window."""

    PLOT_TOP_MARGIN = 0.00
    PLOT_BOTTOM_MARGIN = 0.20
    NAME_HEADER_HEIGHT = 24
    TIME_AXIS_LABEL_PAD = 8

    DIGITAL_LOW = 0.28
    DIGITAL_HIGH = 0.72
    BUS_LOW = 0.22
    BUS_HIGH = 0.78

    DIGITAL_LINE_WIDTH = 1.75
    BUS_LINE_WIDTH = 1.65
    BUS_TEXT_SIZE = 9.0
    MINIMUM_LANE_HEIGHT = 18

    THEME_EVENTS = (
        QtCore.QEvent.Type.PaletteChange,
        QtCore.QEvent.Type.ApplicationPaletteChange,
    )

    def __init__(self, file_path, project_name, parent=None):
        super().__init__(parent)
        self.file_path = file_path
        self.project_name = project_name

        self.setObjectName("WaveWindow")
        self.setMinimumSize(550, 300)

        self.obj_appconfig = Appconfig()
        app = QtWidgets.QApplication.instance()
        self._palette = current_palette(app)

        self._theme = {}
        self._is_dark_theme = True
        self._applying_theme = False
        self._theme_update_pending = False
        self.grid_enabled = True

        self.traces: Dict[int, Trace] = {}
        self.panes = []

        self.vcd_times = np.array([], dtype=np.float64)
        self.vcd_signals: Dict[str, Dict[str, Any]] = {}

        self.color_palette = VIBRANT_COLOR_PALETTE.copy()
        self._last_xlim = None
        self._rendering = False
        self._previous_grid_color = None

        self.create_main_frame()
        self._rebuild_theme_colors()
        self.apply_theme()
        self._setup_matplotlib_style()
        self.load_vcd_data()

    # =========================================================
    # THEME COLOR GENERATION
    # =========================================================

    @staticmethod
    def _valid_color(color):
        if isinstance(color, QtGui.QColor):
            return color.isValid()
        return QtGui.QColor(str(color)).isValid()

    @staticmethod
    def _qcolor(color):
        if isinstance(color, QtGui.QColor):
            return QtGui.QColor(color)
        return QtGui.QColor(str(color))

    def _get_palette_color(self, palette, role, fallback):
        color = palette.color(role)
        if color.isValid():
            return color.name()
        return fallback

    def _rebuild_theme_colors(self):
        app = QtWidgets.QApplication.instance()
        if app is None:
            return

        palette = app.palette()
        window_color = palette.color(QtGui.QPalette.ColorRole.Window)
        self._is_dark_theme = window_color.lightness() < 128
        fallback = DARK_FALLBACK if self._is_dark_theme else LIGHT_FALLBACK
        role = QtGui.QPalette.ColorRole

        colors = {
            "window": self._get_palette_color(
                palette, role.Window, fallback["window"]
            ),
            "panel": self._get_palette_color(
                palette, role.Base, fallback["panel"]
            ),
            "panel_light": self._get_palette_color(
                palette, role.AlternateBase, fallback["panel_light"]
            ),
            "wave": self._get_palette_color(
                palette, role.Base, fallback["wave"]
            ),
            "text": self._get_palette_color(
                palette, role.WindowText, fallback["text"]
            ),
            "signal_text": self._get_palette_color(
                palette, role.Text, fallback["signal_text"]
            ),
            "button": self._get_palette_color(
                palette, role.Button, fallback["button"]
            ),
            "button_text": self._get_palette_color(
                palette, role.ButtonText, fallback["text"]
            ),
            "border": self._get_palette_color(
                palette, role.Mid, fallback["border"]
            ),
            "muted": self._get_palette_color(
                palette, role.PlaceholderText, fallback["muted"]
            ),
            "selection": self._get_palette_color(
                palette, role.Highlight, fallback["selection"]
            ),
            "selection_text": self._get_palette_color(
                palette, role.HighlightedText, fallback["selection_text"]
            ),
            "signal_panel": fallback["signal_panel"],
            "input": fallback["input"],
            "input_text": fallback["input_text"],
            "button_hover": fallback["button_hover"],
            "unknown": fallback["unknown"],
        }

        wave_color = self._qcolor(colors["wave"])
        grid_color = (
            wave_color.lighter(155)
            if self._is_dark_theme
            else wave_color.darker(115)
        )
        colors["grid"] = grid_color.name()

        border_color = self._qcolor(colors["border"])
        border_color = (
            border_color.lighter(115)
            if self._is_dark_theme
            else border_color.darker(108)
        )
        colors["border"] = border_color.name()

        self._theme = colors

    def _c(self, key):
        return self._theme.get(key, DARK_FALLBACK.get(key, "#FFFFFF"))

    # =========================================================
    # THEME APPLICATION
    # =========================================================

    def apply_theme(self):
        if not self._theme:
            self._rebuild_theme_colors()

        self._applying_theme = True
        try:
            c = self._c
            self.setStyleSheet(f"""
            QWidget#WaveWindow {{
                background-color: {c("window")};
                color: {c("text")};
                font-family: "Segoe UI";
                font-size: 9pt;
            }}
            QWidget#WaveHeader {{
                background-color: {c("panel_light")};
                border: 1px solid {c("border")};
                border-radius: 4px;
            }}
            QLabel#AppTitle {{
                color: {c("text")};
                font-size: 11pt;
                font-weight: 700;
            }}
            QLabel#AppSubtitle {{
                color: {c("muted")};
                font-size: 8pt;
            }}
            QLabel#HeaderBadge {{
                background-color: {c("button")};
                color: {c("button_text")};
                border: 1px solid {c("border")};
                border-radius: 3px;
                padding: 5px 9px;
            }}
            QWidget#SidePanel {{
                background-color: {c("panel")};
                border: 1px solid {c("border")};
                border-radius: 4px;
            }}
            QWidget#WaveCenter {{
                background-color: {c("wave")};
                border: 1px solid {c("border")};
                border-radius: 4px;
            }}
            QWidget#SignalNamePanel {{
                background-color: {c("signal_panel")};
                border-right: 1px solid {c("border")};
            }}
            QWidget#SignalRowsContainer {{
                background-color: {c("signal_panel")};
            }}
            QLabel#PanelHeading {{
                color: {c("text")};
                font-size: 8pt;
                font-weight: 700;
                padding: 3px;
            }}
            QLabel#MutedLabel {{
                color: {c("muted")};
                font-size: 8pt;
            }}
            QLabel#SignalName {{
                color: {c("signal_text")};
                background-color: transparent;
                padding-left: 7px;
                font-family: "Consolas";
                font-size: 8.5pt;
                font-weight: 600;
            }}
            QLineEdit#SignalSearch {{
                background-color: {c("input")};
                border: 1px solid {c("border")};
                border-radius: 3px;
                padding: 7px;
                color: {c("input_text")};
                selection-background-color: {c("selection")};
                selection-color: {c("selection_text")};
            }}
            QListWidget#SignalList {{
                background-color: {c("panel_light")};
                border: 1px solid {c("border")};
                border-radius: 3px;
                color: {c("signal_text")};
                padding: 3px;
                outline: none;
            }}
            QListWidget#SignalList::item {{
                padding: 5px;
                border-bottom: 1px solid {c("border")};
            }}
            QListWidget#SignalList::item:hover {{
                background-color: {c("button_hover")};
            }}
            QListWidget#SignalList::item:selected {{
                background-color: {c("selection")};
                color: {c("selection_text")};
            }}
            QPushButton {{
                background-color: {c("button")};
                color: {c("button_text")};
                border: 1px solid {c("border")};
                border-radius: 3px;
                padding: 6px 10px;
            }}
            QPushButton:hover {{
                background-color: {c("button_hover")};
            }}
            QPushButton:pressed {{
                background-color: {c("selection")};
                color: {c("selection_text")};
            }}
            QCheckBox {{
                color: {c("signal_text")};
                spacing: 7px;
            }}
            QCheckBox::indicator {{
                width: 14px;
                height: 14px;
            }}
            QWidget#NavigationContainer {{
                background-color: {c("panel_light")};
                border: 1px solid {c("border")};
                border-radius: 3px;
            }}
            QWidget#NavigationContainer QToolButton {{
                background-color: transparent;
                color: {c("text")};
                border: none;
                padding: 4px;
            }}
            QWidget#NavigationContainer QToolButton:hover {{
                background-color: {c("button_hover")};
                border-radius: 3px;
            }}
            QStatusBar#WaveStatusBar {{
                background-color: {c("panel_light")};
                border: 1px solid {c("border")};
                color: {c("text")};
            }}
            QSplitter::handle {{
                background-color: {c("border")};
                width: 3px;
            }}
            QScrollArea {{
                background-color: {c("wave")};
                border: none;
            }}
            QScrollBar:horizontal {{
                background-color: {c("panel")};
                height: 11px;
                margin: 0px;
            }}
            QScrollBar::handle:horizontal {{
                background-color: {c("border")};
                min-width: 30px;
                border-radius: 4px;
            }}
            QScrollBar::handle:horizontal:hover {{
                background-color: {c("muted")};
            }}
            QScrollBar::add-line:horizontal,
            QScrollBar::sub-line:horizontal {{
                width: 0px;
            }}
            QToolTip {{
                background-color: {c("panel_light")};
                color: {c("text")};
                border: 1px solid {c("border")};
                padding: 5px;
            }}
            QGroupBox {{
                color: {c("text")};
            }}
            """)

            self.name_rows_container.setStyleSheet(
                f"background-color: {c('signal_panel')};"
            )
            self.name_panel.setStyleSheet(f"""
                QWidget#SignalNamePanel {{
                    background-color: {c('signal_panel')};
                    border-right: 1px solid {c('border')};
                }}
            """)
            self.wave_scroll.setStyleSheet(f"""
                QScrollArea {{
                    background-color: {c('wave')};
                    border: none;
                }}
            """)
            self.wave_scroll.viewport().setStyleSheet(
                f"background-color: {c('wave')};"
            )

            self.canvas.setStyleSheet(f"""
                background-color: {c('wave')};
                border: none;
            """)

            self._update_canvas_palette()
            self._retheme_matplotlib()
            self.update()
        finally:
            self._applying_theme = False

    def changeEvent(self, event):
        super().changeEvent(event)
        if event.type() in self.THEME_EVENTS and not self._applying_theme:
            self._schedule_theme_update()

    def _schedule_theme_update(self):
        if self._theme_update_pending:
            return
        self._theme_update_pending = True
        QtCore.QTimer.singleShot(0, self._retheme_on_palette_change)

    def _retheme_on_palette_change(self):
        self._theme_update_pending = False
        app = QtWidgets.QApplication.instance()
        if app is None:
            return
        try:
            self._palette = current_palette(app)
            self._rebuild_theme_colors()
            self._setup_matplotlib_style()
            self.apply_theme()
        except Exception:
            logger.exception("Failed to update waveform viewer theme")

    def _update_canvas_palette(self):
        color = QtGui.QColor(self._c("wave"))
        palette = self.canvas.palette()
        palette.setColor(QtGui.QPalette.ColorRole.Window, color)
        palette.setColor(QtGui.QPalette.ColorRole.Base, color)
        self.canvas.setPalette(palette)
        self.canvas.setAutoFillBackground(True)

    def _retheme_matplotlib(self):
        if not hasattr(self, "fig"):
            return
        c = self._c
        self.fig.patch.set_facecolor(c("wave"))

        for ax in self.fig.axes:
            ax.set_facecolor(c("wave"))
            for spine in ax.spines.values():
                spine.set_color(c("border"))
                spine.set_linewidth(0.6)

            ax.tick_params(
                axis="x",
                colors=c("text"),
                labelsize=9,
                length=4,
                width=0.8,
                pad=5,
            )
            ax.tick_params(
                axis="y", colors=c("muted"), left=False, labelleft=False
            )
            ax.xaxis.label.set_color(c("text"))
            ax.yaxis.label.set_color(c("text"))

            for gridline in ax.get_xgridlines():
                gridline.set_color(c("grid"))
                gridline.set_linewidth(0.65)
                gridline.set_alpha(0.95)

            for line in ax.lines:
                try:
                    is_old_grid = line.get_color() == self._previous_grid_color
                except AttributeError:
                    is_old_grid = False
                if is_old_grid:
                    line.set_color(c("grid"))

            ax.set_axisbelow(True)

        self._previous_grid_color = c("grid")
        self.canvas.draw_idle()

    # =========================================================
    # INTERFACE SETUP
    # =========================================================

    def create_main_frame(self):
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(8, 8, 8, 6)
        main_layout.setSpacing(6)

        # Header
        header = QWidget()
        header.setObjectName("WaveHeader")
        header_layout = QHBoxLayout(header)
        header_layout.setContentsMargins(12, 7, 12, 7)

        title_layout = QVBoxLayout()
        title_layout.setSpacing(1)
        title = QLabel("WAVEFORM ANALYZER")
        title.setObjectName("AppTitle")

        subtitle = QLabel(f"{self.project_name}  |  Digital Simulation")
        subtitle.setObjectName("AppSubtitle")

        title_layout.addWidget(title)
        title_layout.addWidget(subtitle)
        header_layout.addLayout(title_layout)
        header_layout.addStretch()

        self.signal_count_label = QLabel("0 SIGNALS")
        self.signal_count_label.setObjectName("HeaderBadge")
        header_layout.addWidget(self.signal_count_label)
        main_layout.addWidget(header)

        # Main Splitter (2 Panes: Signal Browser + Waveform View)
        self.splitter = QSplitter(Qt.Orientation.Horizontal)
        self.splitter.setChildrenCollapsible(False)

        # 1. Left Signal Browser Panel
        left_widget = QWidget()
        left_widget.setObjectName("SidePanel")
        left_layout = QVBoxLayout(left_widget)
        left_layout.setContentsMargins(8, 10, 8, 8)
        left_layout.setSpacing(7)

        heading = QLabel("SIGNAL BROWSER")
        heading.setObjectName("PanelHeading")
        left_layout.addWidget(heading)

        self.search_box = QLineEdit()
        self.search_box.setObjectName("SignalSearch")
        self.search_box.setPlaceholderText("Search signals...")
        self.search_box.textChanged.connect(self.filter_waveforms)
        left_layout.addWidget(self.search_box)

        self.waveform_list = CustomListWidget()
        self.waveform_list.setObjectName("SignalList")
        self.waveform_list.setUniformItemSizes(True)
        self.waveform_list.itemClicked.connect(self.on_waveform_toggle)
        left_layout.addWidget(self.waveform_list, 1)

        self.select_all_btn = QPushButton("Deselect All")
        self.select_all_btn.clicked.connect(self.toggle_select_all_waveforms)
        left_layout.addWidget(self.select_all_btn)
        self.splitter.addWidget(left_widget)

        # 2. Center Waveform View Panel
        center_widget = QWidget()
        center_widget.setObjectName("WaveCenter")
        center_layout = QVBoxLayout(center_widget)
        center_layout.setContentsMargins(5, 5, 5, 5)
        center_layout.setSpacing(4)

        waveform_header = QHBoxLayout()
        waveform_title = QLabel("WAVEFORM VIEW")
        waveform_title.setObjectName("PanelHeading")
        waveform_header.addWidget(waveform_title)
        waveform_header.addStretch()

        self.waveform_info_label = QLabel("Digital Timing Diagram")
        self.waveform_info_label.setObjectName("MutedLabel")
        waveform_header.addWidget(self.waveform_info_label)
        center_layout.addLayout(waveform_header)

        # Matplotlib Figure
        self.fig = Figure(figsize=DEFAULT_FIGURE_SIZE, dpi=DEFAULT_DPI)
        self.canvas = WaveFigureCanvas(self.fig)
        self.nav_toolbar = HorizontalNavigationToolbar(self.canvas, self)

        toolbar_container = QWidget()
        toolbar_container.setObjectName("NavigationContainer")
        toolbar_layout = QHBoxLayout(toolbar_container)
        toolbar_layout.setContentsMargins(4, 2, 4, 2)
        toolbar_layout.addWidget(self.nav_toolbar)
        toolbar_layout.addStretch()
        center_layout.addWidget(toolbar_container)

        # Signal Names & Waveform Content Layout
        waveform_content = QWidget()
        waveform_content_layout = QHBoxLayout(waveform_content)
        waveform_content_layout.setContentsMargins(0, 0, 0, 0)
        waveform_content_layout.setSpacing(0)

        # Fixed signal-name panel
        self.name_panel = QWidget()
        self.name_panel.setObjectName("SignalNamePanel")
        self.name_panel.setFixedWidth(155)

        self.name_panel_layout = QVBoxLayout(self.name_panel)
        self.name_panel_layout.setContentsMargins(0, 0, 0, 0)
        self.name_panel_layout.setSpacing(0)

        self.name_header = QLabel("SIGNAL")
        self.name_header.setObjectName("PanelHeading")
        self.name_header.setFixedHeight(self.NAME_HEADER_HEIGHT)
        self.name_header.setAlignment(
            Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft
        )
        self.name_panel_layout.addWidget(self.name_header)

        self.name_rows_container = QWidget()
        self.name_rows_container.setObjectName("SignalRowsContainer")
        self.name_rows_layout = QVBoxLayout(self.name_rows_container)
        self.name_rows_layout.setContentsMargins(0, 0, 0, 0)
        self.name_rows_layout.setSpacing(0)
        self.name_panel_layout.addWidget(self.name_rows_container, 1)

        waveform_content_layout.addWidget(self.name_panel)

        # Scroll Area
        self.wave_scroll = QScrollArea()
        self.wave_scroll.setObjectName("WaveScroll")
        self.wave_scroll.setWidgetResizable(False)
        self.wave_scroll.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAsNeeded
        )
        self.wave_scroll.setVerticalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )
        self.wave_scroll.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
        self.wave_scroll.setWidget(self.canvas)
        waveform_content_layout.addWidget(self.wave_scroll, 1)

        center_layout.addWidget(waveform_content, 1)
        self.splitter.addWidget(center_widget)

        # Splitter sizing for 2 panels: 220px fixed left, remaining flexible right
        self.splitter.setSizes([220, 950])
        self.splitter.setStretchFactor(0, 0)
        self.splitter.setStretchFactor(1, 1)

        main_layout.addWidget(self.splitter, 1)

        self.status_bar = QStatusBar()
        self.status_bar.setObjectName("WaveStatusBar")
        self.status_bar.showMessage("Ready")
        main_layout.addWidget(self.status_bar)

        self.setWindowTitle(f"Waveform Analyzer - {self.project_name}")

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if hasattr(self, "fig") and not self._rendering:
            QtCore.QTimer.singleShot(50, self.refresh_plot)

    # =========================================================
    # VCD LOADING & PARSING
    # =========================================================

    def load_vcd_data(self):
        candidates = [
            os.path.join(self.file_path, "sim_out.vcd"),
            os.path.join(self.file_path, "counter.vcd"),
        ]
        workspace = os.path.join(
            os.path.expanduser("~"), "eSim-Workspace", str(self.project_name)
        )
        candidates.extend([
            os.path.join(workspace, "sim_out.vcd"),
            os.path.join(workspace, "counter.vcd"),
        ])

        vcd_path = next(
            (path for path in candidates if os.path.exists(path)), None
        )

        if not vcd_path:
            logger.error(
                "VCD file not found for project %s", self.project_name
            )
            self.status_bar.showMessage("No VCD file found")
            return

        self.vcd_times, self.vcd_signals = self._parse_native_vcd(vcd_path)
        self.populate_waveform_list()
        self.refresh_plot()
        self.status_bar.showMessage("Ready")

    def _parse_native_vcd(self, filepath):
        times = []
        signals = {}
        symbol_to_keys = {}
        scope_stack = []

        def make_unique_name(name):
            if name not in signals:
                return name
            suffix = 2
            while f"{name} [{suffix}]" in signals:
                suffix += 1
            return f"{name} [{suffix}]"

        def update_symbol(symbol, value):
            for key in symbol_to_keys.get(symbol, []):
                signals[key]["curr"] = value
                if times and signals[key]["values"]:
                    signals[key]["values"][-1] = value

        with open(filepath, "r", errors="ignore") as f:
            for raw_line in f:
                line = raw_line.strip()
                if not line:
                    continue

                if line.startswith("$scope"):
                    parts = line.split()
                    if len(parts) >= 3:
                        scope_stack.append(parts[2])
                    continue

                if line.startswith("$upscope"):
                    if scope_stack:
                        scope_stack.pop()
                    continue

                if line.startswith("$var"):
                    parts = line.split()
                    if len(parts) < 6:
                        continue

                    try:
                        width = int(parts[2])
                    except ValueError:
                        continue

                    symbol = parts[3]
                    reference = parts[4]
                    display_scope = scope_stack[1:]
                    hierarchical_name = (
                        ".".join(display_scope + [reference])
                        if display_scope
                        else reference
                    )
                    hierarchical_name = make_unique_name(hierarchical_name)
                    initial_value = "0" * width if width > 1 else "0"

                    signals[hierarchical_name] = {
                        "name": hierarchical_name,
                        "reference": reference,
                        "scope": ".".join(scope_stack),
                        "width": width,
                        "values": [],
                        "curr": initial_value,
                        "symbol": symbol,
                    }
                    symbol_to_keys.setdefault(symbol, []).append(
                        hierarchical_name
                    )
                    continue

                if line.startswith("$"):
                    continue

                if line.startswith("#"):
                    try:
                        timestamp = float(line[1:])
                    except ValueError:
                        continue
                    times.append(timestamp)
                    for sig in signals.values():
                        sig["values"].append(sig["curr"])
                    continue

                if line[0] in ("b", "B"):
                    parts = line.split()
                    if len(parts) >= 2:
                        value = parts[0][1:]
                        symbol = parts[1]
                        if symbol in symbol_to_keys:
                            for key in symbol_to_keys[symbol]:
                                width = signals[key]["width"]
                                value = value.lower()
                                if len(value) < width:
                                    value = value.zfill(width)
                                signals[key]["curr"] = value
                                if times and signals[key]["values"]:
                                    signals[key]["values"][-1] = value
                    continue

                if line[0] in ("0", "1", "x", "X", "z", "Z"):
                    value = line[0].lower()
                    symbol = line[1:]
                    if symbol in symbol_to_keys:
                        update_symbol(symbol, value)
                    continue

        logger.info(
            "Parsed VCD: %d timestamps, %d hierarchical signals",
            len(times),
            len(signals),
        )
        return np.array(times, dtype=np.float64), signals

    # =========================================================
    # SIGNAL LIST MANAGEMENT
    # =========================================================

    def populate_waveform_list(self):
        self.waveform_list.clear()
        self.traces.clear()

        for idx, (signal_key, sig) in enumerate(self.vcd_signals.items()):
            color = self.color_palette[idx % len(self.color_palette)]
            display_name = sig["name"]

            self.traces[idx] = Trace(
                index=idx, name=display_name, color=color, visible=True
            )

            item = QtWidgets.QListWidgetItem(display_name)
            item.setCheckState(Qt.CheckState.Checked)
            item.setData(Qt.ItemDataRole.UserRole, signal_key)
            item.setToolTip(
                f"Scope: {sig['scope']}\n"
                f"Reference: {sig['reference']}\n"
                f"Width: {sig['width']} bit(s)"
            )
            self.waveform_list.addItem(item)
            item.setSelected(True)  # Keep highlight synced on load

        self._update_viewer_information()

    def filter_waveforms(self, text):
        for i in range(self.waveform_list.count()):
            item = self.waveform_list.item(i)
            item.setHidden(text.lower() not in item.text().lower())

    def on_waveform_toggle(self, item):
        row = self.waveform_list.row(item)
        if row in self.traces:
            new_visible = not self.traces[row].visible
            self.traces[row].visible = new_visible
            
            # Sync both the checkmark and the blue highlight
            item.setCheckState(Qt.CheckState.Checked if new_visible else Qt.CheckState.Unchecked)
            item.setSelected(new_visible)
            
            self.refresh_plot()

    def toggle_select_all_waveforms(self):
        state = (
            Qt.CheckState.Unchecked
            if self.select_all_btn.text() == "Deselect All"
            else Qt.CheckState.Checked
        )
        is_visible = (state == Qt.CheckState.Checked)

        for i in range(self.waveform_list.count()):
            item = self.waveform_list.item(i)
            item.setCheckState(state)
            item.setSelected(is_visible)  # Sync highlight

        for trace in self.traces.values():
            trace.visible = is_visible

        self.select_all_btn.setText(
            "Select All" if not is_visible else "Deselect All"
        )
        self.refresh_plot()

    # =========================================================
    # BUS DRAWING & CONVERSION
    # =========================================================

    def _value_to_integer(self, value):
        value = str(value).strip().lower()
        if value.startswith("b"):
            value = value[1:]
        if not value or any(bit in value for bit in ("x", "z")):
            return None
        if not set(value).issubset({"0", "1"}):
            return None
        try:
            return int(value, 2)
        except ValueError:
            return None

    def _draw_bus(self, ax, x, values, color, width, lane):
        if len(x) == 0 or len(values) == 0:
            return

        y_top = lane + self.BUS_HIGH
        y_bottom = lane + self.BUS_LOW
        y_mid = lane + 0.5

        digits = max(1, (width + 3) // 4)
        converted = [self._value_to_integer(value) for value in values]

        segments = []
        start = 0

        for i in range(1, len(converted)):
            if converted[i] != converted[i - 1]:
                segments.append((start, i, converted[i - 1]))
                start = i
        segments.append((start, len(converted), converted[-1]))

        span = x[-1] - x[0] if len(x) > 1 else 1.0
        step = span / max(1, len(x) - 1) if len(x) > 1 else 0.1
        diagonal = step * 0.22

        for seg_index, (start_idx, end_idx, value) in enumerate(segments):
            x_start = x[start_idx]
            x_end = x[end_idx] if end_idx < len(x) else x[-1] + step

            left_x = x_start + diagonal if seg_index > 0 else x_start
            right_x = x_end - diagonal if end_idx < len(x) else x_end

            if right_x < left_x:
                left_x = x_start
                right_x = x_end

            points = [
                (x_start, y_mid),
                (left_x, y_top),
                (right_x, y_top),
                (x_end, y_mid),
                (right_x, y_bottom),
                (left_x, y_bottom),
            ]

            polygon = Polygon(
                points,
                closed=True,
                facecolor="none",
                edgecolor=color,
                linewidth=self.BUS_LINE_WIDTH,
                joinstyle="miter",
            )
            ax.add_patch(polygon)

            label = "X" if value is None else f"{value:0{digits}X}"
            mid_x = (x_start + x_end) / 2

            if right_x - left_x > step * 0.7:
                ax.text(
                    mid_x,
                    y_mid,
                    label,
                    color=color,
                    ha="center",
                    va="center",
                    fontsize=self.BUS_TEXT_SIZE,
                    fontfamily="monospace",
                    fontweight="bold",
                    clip_on=True,
                    zorder=5,
                )

    # =========================================================
    # LAYOUT ALIGNMENT & RENDERING
    # =========================================================

    def _rebuild_signal_names(self, visible_traces, canvas_height):
        while self.name_rows_layout.count():
            item = self.name_rows_layout.takeAt(0)
            widget = item.widget()
            if widget:
                widget.deleteLater()

        count = len(visible_traces)
        if count == 0:
            return

        top_px = 0
        bottom_px = round(canvas_height * self.PLOT_BOTTOM_MARGIN)
        plot_height = max(
            1, canvas_height - self.NAME_HEADER_HEIGHT - bottom_px
        )
        lane_height = plot_height / count

        self.name_rows_layout.setContentsMargins(0, top_px, 0, bottom_px)

        for trace in visible_traces:
            label = QLabel(trace.name)
            label.setObjectName("SignalName")
            label.setFixedHeight(max(1, round(lane_height)))
            label.setAlignment(
                Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft
            )
            label.setToolTip(trace.name)
            self.name_rows_layout.addWidget(label)

    def _style_axis(self, ax):
        c = self._c
        ax.set_facecolor(c("wave"))

        for spine in ax.spines.values():
            spine.set_color(c("border"))
            spine.set_linewidth(0.6)

        ax.tick_params(
            axis="x",
            colors=c("text"),
            labelsize=9,
            length=4,
            width=0.8,
            pad=5,
            labelbottom=True,
            bottom=True,
        )
        ax.tick_params(axis="y", left=False, labelleft=False)
        ax.set_axisbelow(True)

        ax.grid(
            getattr(self, "grid_enabled", True),
            axis="x",
            color=c("grid"),
            linewidth=0.65,
            alpha=0.95,
        )
        ax.grid(False, axis="y")

        ax.spines["left"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)

    def _update_viewer_information(self):
        total = len(self.vcd_signals)
        visible = sum(1 for trace in self.traces.values() if trace.visible)

        self.signal_count_label.setText(f"{total} SIGNALS")
        self.waveform_info_label.setText(f"{visible} active traces")

    def refresh_plot(self, *args):
        if not hasattr(self, "fig") or self._rendering:
            return

        self._rendering = True
        try:
            was_empty = getattr(self, '_is_empty_view', False)
            
            old_xlim = None
            if not was_empty and self.fig.axes:
                old_xlim = self.fig.axes[0].get_xlim()
            elif not was_empty:
                old_xlim = self._last_xlim

            visible_traces = [
                trace for trace in self.traces.values() if trace.visible
            ]
            x = np.asarray(self.vcd_times, dtype=float)

            canvas_width = max(500, self.wave_scroll.viewport().width())
            canvas_height = max(160, self.wave_scroll.viewport().height())
            plot_width = max(1000, canvas_width)

            self.fig.clear()
            self.fig.patch.set_facecolor(self._c("wave"))
            self.fig.set_size_inches(
                plot_width / DEFAULT_DPI,
                canvas_height / DEFAULT_DPI,
                forward=True,
            )

            self._rebuild_signal_names(visible_traces, canvas_height)

            # Handle Empty View
            if not visible_traces or len(x) == 0:
                self._is_empty_view = True
                ax = self.fig.add_axes([0.015, 0.12, 0.975, 0.80])
                ax.set_facecolor(self._c("wave"))
                ax.text(
                    0.5,
                    0.5,
                    "No signals selected",
                    color=self._c("muted"),
                    ha="center",
                    va="center",
                    transform=ax.transAxes,
                )
                ax.set_xticks([])
                ax.set_yticks([])

                self.canvas.setMinimumSize(plot_width, canvas_height)
                self.canvas.resize(plot_width, canvas_height)
                self._update_canvas_palette()
                self.canvas.draw()
                self._update_viewer_information()
                return

            self._is_empty_view = False
            
            # Shared Axis Setup
            count = len(visible_traces)
            left = 0.015
            right = 0.990
            bottom = self.PLOT_BOTTOM_MARGIN
            top = 1.0 - self.NAME_HEADER_HEIGHT / canvas_height

            ax = self.fig.add_axes([left, bottom, right - left, top - bottom])
            self.panes = [ax]

            ax.set_ylim(0, count)
            ax.set_yticks([])

            default_xlim = (x[0], x[-1]) if len(x) > 1 else (x[0], x[0] + 1)
            ax.set_xlim(default_xlim)

            if old_xlim is not None:
                try:
                    left_limit = max(default_xlim[0], old_xlim[0])
                    right_limit = min(default_xlim[1], old_xlim[1])
                    if right_limit > left_limit:
                        ax.set_xlim(left_limit, right_limit)
                except Exception:
                    logger.debug(
                        "Unable to restore previous x limits", exc_info=True
                    )

            # Draw Signals
            signal_data = list(self.vcd_signals.values())
            for i, trace in enumerate(visible_traces):
                sig_info = signal_data[trace.index]
                values = list(sig_info["values"])

                if len(values) < len(x):
                    fill_value = values[-1] if values else "0"
                    values.extend([fill_value] * (len(x) - len(values)))
                values = values[: len(x)]

                lane = count - i - 1
                color = trace.color
                width = sig_info["width"]

                # Single-Bit Signal
                if width == 1:
                    y_values = []
                    for value in values:
                        v_str = str(value).lower()
                        if v_str == "1":
                            y_values.append(lane + self.DIGITAL_HIGH)
                        elif v_str == "0":
                            y_values.append(lane + self.DIGITAL_LOW)
                        else:
                            y_values.append(lane + 0.5)

                    ax.step(
                        x,
                        y_values,
                        where="post",
                        color=color,
                        linewidth=self.DIGITAL_LINE_WIDTH,
                        solid_capstyle="butt",
                        zorder=3,
                    )

                    for k, value in enumerate(values):
                        if str(value).lower() in ("x", "z"):
                            ax.plot(
                                x[k],
                                lane + 0.5,
                                marker="o",
                                markersize=2.8,
                                color=self._c("unknown"),
                                zorder=4,
                            )
                # Multi-Bit Bus Signal
                else:
                    self._draw_bus(ax, x, values, color, width, lane)

            # Horizontal Lane Separators
            for i in range(1, count):
                ax.axhline(
                    i,
                    color=self._c("grid"),
                    linewidth=0.55,
                    alpha=0.85,
                    zorder=0,
                )

            self._style_axis(ax)

            # Timestamp X-Axis
            ax.set_xlabel(
                "Simulation Time",
                color=self._c("text"),
                fontsize=9,
                fontweight="bold",
                labelpad=self.TIME_AXIS_LABEL_PAD,
            )
            ax.xaxis.set_label_position("bottom")
            ax.xaxis.set_ticks_position("bottom")
            ax.tick_params(
                axis="x", which="both", labelbottom=True, bottom=True
            )

            self._previous_grid_color = self._c("grid")
            self.canvas.setMinimumSize(plot_width, canvas_height)
            self.canvas.resize(plot_width, canvas_height)
            self._update_canvas_palette()
            self.canvas.draw()
            self._last_xlim = ax.get_xlim()
            self._update_viewer_information()
        finally:
            self._rendering = False

    def _setup_matplotlib_style(self):
        try:
            plt.rcParams.update(matplotlib_rc_overrides(self._palette))
        except Exception:
            logger.exception("Failed to apply Matplotlib theme")

    def closeEvent(self, event):
        if hasattr(self, "fig"):
            plt.close(self.fig)
        super().closeEvent(event)


# PEP 8 Alias for backward compatibility
WaveWindow = waveWindow