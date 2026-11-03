import datetime
import os
import re
import shutil
import subprocess
import tempfile
from PyQt6 import QtCore, QtGui, QtWidgets
from frontEnd.theme_utils import zoom_px, on_zoom_changed

# eSim is PyQt6-only. These aliases keep the call sites below terse.
TEXT_SELECTABLE = QtCore.Qt.TextInteractionFlag.TextSelectableByMouse
WIN_MAX = QtCore.Qt.WindowType.WindowMaximizeButtonHint
WIN_MIN = QtCore.Qt.WindowType.WindowMinimizeButtonHint
ORIENT_HORIZ = QtCore.Qt.Orientation.Horizontal
TAB_RIGHT = QtWidgets.QTabBar.ButtonPosition.RightSide
TOOLBTN_INSTANT = QtWidgets.QToolButton.ToolButtonPopupMode.InstantPopup

from PyQt6.Qsci import QsciScintilla
# Reuse eSim's shared editor lexers + theme so HDL here looks exactly like the
# project code editor (same QsciLexerVerilog/VHDL colours), instead of a
# bespoke regex highlighter.
from codeEditor import lexers, theme

# Shared d_cosim toolchain resolver: it already knows where iverilog lives
# (env ESIM_IVERILOG -> ~/.nghdl/config.ini [COSIM] -> PATH), recorded by the
# installer. The verifier reuses it so both features find the same iverilog
# instead of maintaining a second, PATH-only locator that misses prefix builds.
try:
    from . import CosimConfig
    from . import verilog_library
    from .hdl import icarus, jobs
    from .hdl.vcd import parse_vcd_for_plot, to_csv
    from .hdl.ports import (autodump_source, dump_file_name, extract_ports,
                            find_modules, generate_stub_testbench, has_dump,
                            has_finish, is_generated_testbench,
                            is_self_contained_testbench, order_modules,
                            reserved_modules, reserved_name_reason,
                            strip_comments, testbench_matches,
                            top_module_name)
    from .hdl.hdl_waveform_viewer import waveWindow
except ImportError:  # launched with src/ on sys.path (absolute-import style)
    from maker import CosimConfig
    from maker import verilog_library
    from maker.hdl import icarus, jobs
    from maker.hdl.vcd import parse_vcd_for_plot, to_csv
    from maker.hdl.ports import (autodump_source, dump_file_name,
                                 extract_ports, find_modules,
                                 generate_stub_testbench, has_dump, has_finish,
                                 is_generated_testbench,
                                 is_self_contained_testbench, order_modules,
                                 reserved_modules, reserved_name_reason,
                                 strip_comments, testbench_matches,
                                 top_module_name)
    from maker.hdl.hdl_waveform_viewer import waveWindow

# Wall-clock caps for the toolchain so a runaway testbench (e.g. a clock with
# no $finish) can't hang the worker thread indefinitely. Generous: this tool's
# designs are small, so a compile/sim that blows these is almost always a stuck
# run, not a slow one. The user can still Cancel sooner.
COMPILE_TIMEOUT_S = 60
SIM_TIMEOUT_S = 120

# Simulated nanoseconds a generated watchdog lets a testbench with no $finish
# run before stopping it. Long enough for a real design to do something, short
# enough that the user gets a waveform instead of hitting SIM_TIMEOUT_S with
# nothing to show.
NO_FINISH_GUARD_NS = 200000

# Console lines streamed live from a running simulation before the rest is
# summarised. A $display in a clocked loop can emit hundreds of thousands of
# lines; inserting each into a QTextEdit is itself a way to freeze the GUI.
MAX_STREAMED_LINES = 2000


def _no_glow(btn):
    """Opt a small chrome button out of the global Aurora button glow.

    frontEnd.motion seeds every QPushButton with a QGraphicsDropShadowEffect.
    On the verifier's tightly-packed chrome (move arrows, Add Module, Copy) that
    neon halo paints outside the widget and bleeds over neighbours. The design
    contract for this page is borders-only, so we both set ``noMotion`` (so a
    later motion re-install skips it) and strip any effect already attached.
    """
    btn.setProperty("noMotion", True)
    if btn.graphicsEffect() is not None:
        btn.setGraphicsEffect(None)


def _make_verifier_teardown(state):
    """destroyed-slot teardown for the Verify stage's worker thread.

    ``closeEvent`` joins the in-flight compile/sim thread on an explicit close,
    but dock destruction (Close Project / tab close -> deleteLater of the whole
    Model Creation tree) never delivers a closeEvent to nested content -- the
    QThread would then be destroyed while still running ("QThread: Destroyed
    while thread is still running", which can cause a native crash).

    Wired to ``destroyed``, this cancels + joins whatever run is live. It reads
    only the mutable ``state`` dict (plain job/cancel/tmpdir refs), never the
    dying widget; the job is a child QThread still alive when ``destroyed`` fires
    (children are deleted AFTER the parent emits destroyed), so the join lands
    before the crash. Idempotent with closeEvent's own join."""
    def _teardown(*_a):
        job = state.get("job")
        cancel = state.get("cancel")
        try:
            if job is not None and job.isRunning():
                if cancel is not None:
                    cancel.cancel()
                job.wait(3000)
        except RuntimeError:
            # Wrapper already gone on the Qt side; nothing to join.
            pass
        tmp = state.get("tmpdir")
        if tmp:
            shutil.rmtree(tmp, ignore_errors=True)
        state["job"] = state["cancel"] = state["tmpdir"] = None
    return _teardown


# VcdPlotWindow subclasses plotWindow, which pulls matplotlib (+numpy). Defining
# it at module load made the FIRST Model Creation / Makerchip open pay that
# multi-second cold import even though plots only appear in the Verify stage.
# Build the subclass lazily on first use instead, cached module-wide, so
# authoring/UI opens import nothing heavy.
_VcdPlotWindow = None


def make_vcd_plot_window(timestamps, signals_data, signal_types,
                         project_name="Verilog Simulation", parent=None):
    """Return a VcdPlotWindow, importing matplotlib/plotWindow lazily."""
    global _VcdPlotWindow
    if _VcdPlotWindow is None:
        import numpy as np
        import os
        from ngspiceSimulation.plot_window import plotWindow

        class VcdPlotWindow(plotWindow):
            def __init__(self, timestamps, signals_data, signal_types,
                         project_name="Verilog Simulation", parent=None):
                self.timestamps = timestamps
                self.signals_data = signals_data
                self.signal_types = signal_types
                # Link directly to the permanent path instead of a dummy path
                user_home = os.path.expanduser("~")
                real_path = os.path.join(user_home, "eSim-Workspace", str(project_name))
                super().__init__(real_path, project_name, parent)

            def _parse_tran_start_time(self):
                return 0.0
                
            def _convert_vcd_val(self, val):
                if isinstance(val, (int, float)):
                    return float(val)
                val_str = str(val).strip().lower()
                if val_str.startswith('b'):
                    val_str = val_str[1:]
                if val_str in ('0', '1'):
                    return float(val_str)
                if set(val_str).issubset({'0', '1'}):
                    try:
                        return float(int(val_str, 2))
                    except ValueError:
                        return 0.0
                return 0.0

            def load_simulation_data(self):
                self._cursor_interp_cache.clear()
                self._drawn_signature = None
                self._tran_start_time = self._parse_tran_start_time()

                class MockDataExt:
                    def __init__(self):
                        self.x = np.array([])
                        self.y = []
                        self.NBList = []
                        self.volts_length = 0

                    def numVals(self):
                        return [len(self.y), self.volts_length]

                self.obj_dataext = MockDataExt()

                if self.timestamps and self.signals_data:
                    self.obj_dataext.x = np.array(
                        self.timestamps, dtype=np.float64)
                    for name, y_vals in self.signals_data.items():
                        self.obj_dataext.NBList.append(name)
                        numeric_y = [self._convert_vcd_val(v) for v in y_vals]
                        self.obj_dataext.y.append(
                            np.array(numeric_y, dtype=np.float64))
                    self.obj_dataext.volts_length = len(
                        self.obj_dataext.NBList)

                self.plot_type = [1, 0]
                self._rebuild_nb_sorted()
                self.data_info = self.obj_dataext.numVals()
                self.volts_length = self.data_info[1]

                self.analysis_label.setText("Digital HDL Waveform Analysis")
                self.populate_waveform_list()
                self._apply_persisted_layout()
                self._show_default_traces()

                if hasattr(self, 'radio_timing'):
                    self.radio_timing.setEnabled(True)
                    self.radio_timing.setChecked(True)

            def _show_default_traces(self):
                """Draw something the moment the waveform opens.

                Traces default to hidden, which suits the ngspice flow (you
                pick nodes off a schematic you drew). Here it means the tab a
                simulation just opened is an EMPTY plot beside a list of 48
                signals -- ``$dumpvars`` captures every internal reg too -- and
                the user has to guess which ones they wanted. So preselect the
                testbench's own top-level signals: for a generated testbench
                those are exactly the design's ports, which is what the user
                pressed Simulate to look at. A persisted layout wins; this only
                fills an otherwise blank plot."""
                if hasattr(self, 'traces') and self.traces:
                    if not any(t.visible for t in self.traces.values()):
                        top_signals = [i for i, t in sorted(self.traces.items()) 
                                       if '.' not in t.name and not t.name.startswith('esim_')]
                        chosen = (top_signals or sorted(self.traces))[:12]
                        for i in chosen:
                            self.traces[i].visible = True
                    if hasattr(self, 'waveform_list'):
                        from PyQt6 import QtCore
                        for row in range(self.waveform_list.count()):
                            item = self.waveform_list.item(row)
                            idx = item.data(QtCore.Qt.ItemDataRole.UserRole)
                            if isinstance(idx, int) and idx in self.traces:
                                self.update_list_item_appearance(item, idx)

            def update_plot(self):
                if not hasattr(self, 'fig'): 
                    return
                self.fig.clear()
                visible_traces = [t for t in self.traces.values() if t.visible]
                
                if not visible_traces or len(self.obj_dataext.x) == 0:
                    ax = self.fig.add_subplot(111)
                    ax.text(0.5, 0.5, 'No digital signals selected', ha='center', va='center', transform=ax.transAxes)
                    self.canvas.draw_idle()
                    return

                n = len(visible_traces)
                gs = self.fig.add_gridspec(n, 1, hspace=0.25)
                self.panes = []

                for i, t in enumerate(visible_traces):
                    ax = self.fig.add_subplot(gs[i], sharex=(self.panes[0] if self.panes else None))
                    self.panes.append(ax)

                    y_vals = self.obj_dataext.y[t.index]
                    max_y = np.max(y_vals) if len(y_vals) > 0 else 1.0

                    ax.step(self.obj_dataext.x, y_vals, where='post', color=t.color, linewidth=1.8)
                    
                    if max_y <= 1.0:
                        ax.set_ylim(-0.2, 1.2)
                        ax.set_yticks([0, 1])
                    else:
                        ax.set_ylim(-0.5, max_y + 0.5)

                    ax.set_title(t.name, loc='left', color=t.color, fontsize=9, fontweight='bold', pad=2)
                    
                    show_grid = self.grid_check.isChecked() if hasattr(self, 'grid_check') else True
                    ax.grid(show_grid, linestyle='--', alpha=0.5)
                    
                    if i < n - 1:
                        ax.tick_params(labelbottom=False)

                self.canvas.draw_idle()

            refresh_plot = update_plot
            draw_plot = update_plot

        _VcdPlotWindow = VcdPlotWindow
    return _VcdPlotWindow(timestamps, signals_data, signal_types,
                          project_name, parent)

# Default Verilog Design Example
DEFAULT_DESIGN = """module counter (
    input clk,
    input rst,
    output reg [3:0] count
);

  always @(posedge clk or posedge rst) begin
    if (rst) 
      count <= 0;
    else 
      count <= count + 1;
  end

endmodule
"""

# Default Testbench Example
DEFAULT_TB = """`timescale 1ns/1ps

module tb_counter;
  reg clk;
  reg rst;
  wire [3:0] count;

  counter uut (
    .clk(clk),
    .rst(rst),
    .count(count)
  );

  always #5 clk = ~clk;

  initial begin
    clk = 0;
    rst = 1;
    #20;
    rst = 0;
    #200;
    $finish;
  end

  initial begin
    $dumpfile("sim_out.vcd");
    $dumpvars(0, tb_counter);
  end

endmodule
"""


#: The "// --- name.v ---" separator get_design_code() puts in front of each
#: block, as a pattern that matches a whole run of them at the top of a block.
_BLOCK_BANNER_RE = re.compile(r'\A(?:[ \t]*//[ \t]*---[ \t].*?[ \t]---[ \t]*\r?\n)+')


def _strip_block_banner(code):
    """``code`` without any separator banners already at the top of it.

    get_design_code() labels each block with ``// --- <tab name> ---`` so that
    a multi-file design stays readable once the blocks are concatenated. That
    labelled text is also what collect_into_bus() pushes into the DesignBus,
    which autosaves it to the Verilog library, which loads it back into the
    editor -- so every trip round that loop used to add another banner line to
    the user's own source. Real designs came back with six copies of the same
    comment stacked at the top of the file.

    Stripping first makes the labelling idempotent while keeping it. A leading
    comment of the same shape that the user wrote themselves is absorbed into
    the banner for that block, which is a fair trade against unbounded growth
    in a file they own.
    """
    return _BLOCK_BANNER_RE.sub('', code or '')


class HdlEditor(QsciScintilla):
    """In-memory Verilog/VHDL editor tab built on QsciScintilla.

    Reuses eSim's shared lexers + theme (the same QsciLexerVerilog/VHDL the
    project code editor uses) for syntax highlighting, and gets line numbers,
    folding, brace matching and multi-selection from Scintilla for free --
    replacing the old bespoke QPlainTextEdit + regex highlighter + hand-painted
    line-number gutter. ``toPlainText``/``setPlainText`` are kept as thin
    aliases so the surrounding widget code reads naturally."""

    #: Scintilla indicator id for compile-error squiggles. 8/9 are taken by
    #: the shared theme's find highlights, so use 10 here.
    ERROR_INDICATOR = 10

    def __init__(self, filename="design.v", parent=None):
        super().__init__(parent)
        self.filepath = None
        self._mono = theme.editor_font()
        self.setUtf8(True)
        self.setMarginType(0, QsciScintilla.MarginType.NumberMargin)
        self.setMarginLineNumbers(0, True)
        self.setMarginWidth(0, "0000")
        self.setCaretLineVisible(True)
        self.setBraceMatching(QsciScintilla.BraceMatch.SloppyBraceMatch)
        self.setAutoIndent(True)
        self.setIndentationsUseTabs(False)
        self.setTabWidth(4)
        self.setFolding(QsciScintilla.FoldStyle.BoxedTreeFoldStyle)
        # Push a blank spacer margin (index 3, right of the fold margin) between
        # the fold +/- box and the code so the box no longer sticks to the first
        # character of each line.
        self.setMarginWidth(3, 8)

        self._lexer = lexers.make_lexer(filename, self._mono, self)
        self.setLexer(self._lexer)
        theme.apply(self, self._lexer, self._mono)
        self._blend_margins()

        self.indicatorDefine(
            QsciScintilla.IndicatorStyle.SquiggleIndicator, self.ERROR_INDICATOR)
        self.setIndicatorForegroundColor(
            QtGui.QColor("#e51400"), self.ERROR_INDICATOR)

        # Drop Scintilla's native square sunken frame: the editor sits inside the
        # tab pane, whose rounded #verilogEditorTabs::pane padding now shows the
        # rounded corners. A native frame would re-draw square corners over them.
        self.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)

        # Hide Scintilla's horizontal scroll bar. It painted a thick grey bar
        # flush across the editor's bottom edge, squaring off the rounded card
        # and reading as a crop. HDL lines are short; vertical scroll stays.
        self.SendScintilla(QsciScintilla.SCI_SETHSCROLLBAR, 0)

    def _blend_margins(self):
        # The shared theme gives the line-number / fold gutter a grey tint
        # (MARGIN_BG). That's fine in the standalone code editor, but here the
        # editor sits in a rounded white card: the grey gutter read as a full-
        # height grey band with a square bottom corner. Paint the gutter the
        # same paper colour so it melts into the card and the corners round.
        paper = QtGui.QColor(theme.PAPER)
        self.setMarginsBackgroundColor(paper)
        self.setFoldMarginColors(paper, paper)

    def changeEvent(self, event):
        # Re-theme the editor content when the app palette flips dark<->light
        # so the Verilog tabs track the Aurora theme live, matching the project
        # code editor. Guard on _mono: the event can arrive mid-construction.
        # Deferred by a tick -- same reason as codeEditor.CodeEditor: the event
        # is delivered from inside the app-wide repolish, and re-skinning the
        # editor there re-enters the style walk (see theme_utils.defer_restyle).
        if (event.type() == QtCore.QEvent.Type.PaletteChange
                and hasattr(self, "_mono")):
            from frontEnd.theme_utils import defer_restyle
            defer_restyle(self, self._retheme_content)
        super().changeEvent(event)

    def _retheme_content(self):
        """Deferred body of the PaletteChange branch of changeEvent."""
        try:
            theme.apply(self, self._lexer, self._mono)
            self._blend_margins()
        except RuntimeError:
            pass    # editor torn down between the palette event and this tick

    # -- QPlainTextEdit-compatible text accessors (keep call sites terse) --
    def toPlainText(self):
        return self.text()

    def setPlainText(self, content):
        self.setText(content)

    # -- compile-error highlighting (Scintilla indicators, not ExtraSelection) --
    def clear_error_highlights(self):
        last = max(0, self.lines() - 1)
        self.clearIndicatorRange(
            0, 0, last, self.lineLength(last), self.ERROR_INDICATOR)

    def mark_error_line(self, line_num):
        i = line_num - 1
        if 0 <= i < self.lines():
            self.fillIndicatorRange(
                i, 0, i, max(1, self.lineLength(i)), self.ERROR_INDICATOR)

    def goto_line(self, line_num):
        i = max(0, line_num - 1)
        self.setCursorPosition(i, 0)
        self.ensureLineVisible(i)
        self.setFocus()


class ConsoleEdit(QtWidgets.QTextEdit):
    error_clicked = QtCore.pyqtSignal(int, str)
    
    def mouseDoubleClickEvent(self, e):
        cursor = self.cursorForPosition(e.pos())
        cursor.select(QtGui.QTextCursor.SelectionType.LineUnderCursor)
        line_text = cursor.selectedText()

        # Match any .v or .sv filename in iverilog error format: filename.v:15: message
        match = re.search(r'([\w.-]+\.(?:v|sv)):(\d+):', line_text)
        if match:
            filename = match.group(1)
            line_num = int(match.group(2))
            self.error_clicked.emit(line_num, filename)

        super().mouseDoubleClickEvent(e)


class PinnedTabBar(QtWidgets.QTabBar):
    """Chrome-style drag-to-reorder tab bar whose LAST tab is pinned in place.

    The design tabs are the compile order (``_design_sources`` walks the bar), so
    dragging them is a real edit, not decoration. The testbench tab is different:
    this file addresses it as ``count() - 1`` everywhere (close_tab, rename_tab,
    the TB label sync, the missing close button), so it has to stay last -- a
    press on it never starts a drag, and a carried tab stops at its left edge.

    The drag is painted by hand rather than through ``setMovable(True)``. Qt
    turns off ``paintWithOffsets`` as soon as an application stylesheet is
    installed (it cannot know where a styled tab is drawn), so with the Aurora
    sheet the built-in drag offsets only the tab's *child widgets*: the close
    button slides away while the tab body and its label stay frozen in the slot.
    Here the carried tab is a pixmap the bar draws under the cursor, its vacated
    slot is filled with the strip background (the gap), and the tab it displaces
    slides into the freed slot instead of teleporting.
    """

    #: displaced neighbour slide, and the carried tab settling on drop
    _SLIDE_MS = 150
    _SETTLE_MS = 130

    def __init__(self, parent=None, pin_last=True,
                 background_object="verilogRoot"):
        super().__init__(parent)
        self._pin_last = pin_last
        self._background_object = background_object
        self.setMovable(False)
        # QStyle paints a native base line behind the styled rounded tabs. It
        # shows through their top corners and inter-tab gaps as a bright line
        # in dark mode (dark in light mode). Our pane/tab borders already own
        # the connected edge, so the native base must not be drawn.
        self.setDrawBase(False)
        self._drag_locked = False
        self._press_pos = None
        self._press_index = -1
        # Live drag: the carried tab's index, its pixmap, the grab offset inside
        # it, and where its left edge currently sits.
        self._drag_index = -1
        self._drag_pix = None
        self._drag_size = None
        self._grab_dx = 0
        self._ghost_x = 0
        # One in-flight slide at a time: {"index", "pix", "size", "from", "to",
        # "t"}. A second swap ends the first slide rather than stacking.
        self._flight = None
        self._flight_anim = None
        self._gap_color = QtGui.QColor(0, 0, 0, 0)
        # Set while grabbing a tab: see _clean_grab.
        self._suspend_overlay = False

    # ------------------------------------------------------------------ #
    #  Geometry helpers
    # ------------------------------------------------------------------ #
    def _pinned(self):
        # ``count()`` is deliberately one past the last tab in free mode, so
        # every existing tab remains draggable and the neighbour tests below
        # naturally include the final tab.
        return self.count() - 1 if self._pin_last else self.count()

    def _drag_rect(self):
        rect = self.tabRect(self._drag_index)
        return QtCore.QRect(self._ghost_x, rect.y(),
                            self._drag_size.width(), self._drag_size.height())

    def _flight_rect(self):
        f = self._flight
        start, end, t = f["from"], f["to"], f["t"]
        x = start.x() + (end.x() - start.x()) * t
        y = start.y() + (end.y() - start.y()) * t
        return QtCore.QRect(int(round(x)), int(round(y)),
                            f["size"].width(), f["size"].height())

    def _moving_indexes(self):
        """Tabs this bar paints itself, whose real slots must read as empty."""
        moving = []
        if self._drag_index >= 0:
            moving.append(self._drag_index)
        if self._flight is not None:
            moving.append(self._flight["index"])
        return moving

    def _sample_gap_color(self):
        """The strip colour behind the tabs -- what a vacated slot must show.

        Rendered, not guessed: the bar itself is transparent (grabbing it yields
        alpha-0 pixels that fill nothing), and the colour lives in the
        stylesheet, not the palette, so #verilogRoot is asked to paint one pixel
        of its own background with children excluded. Exact in either theme, and
        it follows a re-theme without a token to keep in sync."""
        fallback = self.palette().color(QtGui.QPalette.ColorRole.Window)
        root = self
        while (root is not None
               and root.objectName() != self._background_object):
            root = root.parentWidget()
        if root is None:
            return fallback
        pixel = QtGui.QPixmap(1, 1)
        pixel.fill(QtCore.Qt.GlobalColor.transparent)
        # Via global coords: mapFrom() warns (and returns nonsense) unless the
        # two widgets sit in the same parent chain, which a re-parented dock
        # cannot promise.
        origin = root.mapFromGlobal(self.mapToGlobal(QtCore.QPoint(0, 0)))
        root.render(pixel, QtCore.QPoint(0, 0),
                    QtGui.QRegion(QtCore.QRect(origin, QtCore.QSize(1, 1))),
                    QtWidgets.QWidget.RenderFlag.DrawWindowBackground)
        color = pixel.toImage().pixelColor(0, 0)
        if color.alpha() == 0:
            return fallback
        color.setAlpha(255)
        return color

    def _clean_grab(self, rect):
        """Grab a tab as the stylesheet draws it, with this bar's own overlay
        suppressed. ``grab`` runs paintEvent, so a plain grab taken mid-drag
        bakes whatever is floating over that tab into the copy -- which is how
        the displaced neighbour ended up carrying a slice of the dragged tab."""
        self._suspend_overlay = True
        try:
            return self.grab(rect)
        finally:
            self._suspend_overlay = False

    def _tab_button(self, index):
        return self.tabButton(index, TAB_RIGHT)

    def _hide_moving_buttons(self):
        """Close buttons are real child widgets: they paint at the tab's layout
        position no matter what this bar draws. While a tab is carried (or
        sliding) its button rides along inside the pixmap, so the live one is
        hidden -- otherwise it stays behind in the empty slot."""
        for index in self._moving_indexes():
            button = self._tab_button(index)
            if button is not None:
                button.hide()

    def _show_button(self, index):
        button = self._tab_button(index)
        if button is not None:
            button.show()

    # ------------------------------------------------------------------ #
    #  Drag lifecycle
    # ------------------------------------------------------------------ #
    def _start_drag(self, index, press_x):
        rect = self.tabRect(index)
        self._gap_color = self._sample_gap_color()
        self._drag_pix = self._clean_grab(rect)
        self._drag_size = rect.size()
        self._grab_dx = press_x - rect.x()
        self._ghost_x = rect.x()
        self._drag_index = index
        self._hide_moving_buttons()
        self.setCursor(QtCore.Qt.CursorShape.ClosedHandCursor)
        self.update()

    def _track_drag(self, cursor_x):
        width = self._drag_size.width()
        # The carried tab stays inside the strip, and stops at the pinned tab:
        # clamping the ghost is what keeps the testbench last without any
        # snap-back.
        if self._pin_last:
            limit = self.tabRect(self._pinned()).left() - width
        elif self.count():
            # Free bars stop at the end of the final tab, rather than at the
            # widget's far edge (which may contain a wide empty strip).
            last = self.tabRect(self.count() - 1)
            limit = last.right() + 1 - width
        else:
            limit = 0
        self._ghost_x = max(0, min(cursor_x - self._grab_dx, max(0, limit)))
        # Swap once the LEADING EDGE of the carried tab passes the neighbour's
        # midpoint, against the immediate neighbour only.
        #
        # Not the carried tab's centre against the neighbour's centre: those
        # are (width_self + width_other) / 2 apart, but the ghost is clamped so
        # it cannot cross the pinned tab, and the clamp bites first whenever the
        # carried tab is the WIDER of the two. A tab with a long label could
        # then never be dragged past a shorter last neighbour at all -- the
        # drag simply did nothing, with no way to tell why. Leading edge vs
        # midpoint is reachable in both directions whatever the widths.
        #
        # Hit-testing the cursor instead (tabAt) thrashes: right after a swap
        # the pointer is still inside the tab it just displaced, so the pair
        # would trade places again on the very next mouse move. Measuring from
        # the ghost keeps that property -- after a swap the carried tab has
        # fully covered its neighbour, so the reverse test cannot also be true.
        current = self._drag_index
        if current + 1 < self._pinned():
            nxt = self.tabRect(current + 1)
            if self._ghost_x + width > nxt.x() + nxt.width() // 2:
                self._swap_with(current + 1)
                self.update()
                return
        if current > 0:
            prev = self.tabRect(current - 1)
            if self._ghost_x < prev.x() + prev.width() // 2:
                self._swap_with(current - 1)
        self.update()

    def _swap_with(self, target):
        """Reorder live, and slide the tab that was pushed out of the way."""
        self._end_flight()
        rect = self.tabRect(target)
        pix = self._clean_grab(rect)
        source = self._drag_index
        self.moveTab(source, target)
        self._drag_index = target
        # moveTab shifts everything between the two slots by one; the tab that
        # was at `target` is now one step back towards where the carried tab
        # came from.
        landed = target - 1 if target > source else target + 1
        if not 0 <= landed < self.count():
            self._hide_moving_buttons()
            return
        self._begin_flight(landed, pix, rect.size(), rect, self.tabRect(landed),
                           self._SLIDE_MS)

    def _begin_flight(self, index, pix, size, start, end, duration):
        self._flight = {"index": index, "pix": pix, "size": size,
                        "from": start, "to": end, "t": 0.0}
        self._hide_moving_buttons()
        anim = QtCore.QVariantAnimation(self)
        anim.setStartValue(0.0)
        anim.setEndValue(1.0)
        anim.setDuration(duration)
        anim.setEasingCurve(QtCore.QEasingCurve.Type.OutCubic)
        anim.valueChanged.connect(self._on_flight_value)
        anim.finished.connect(self._end_flight)
        self._flight_anim = anim
        anim.start(QtCore.QAbstractAnimation.DeletionPolicy.DeleteWhenStopped)

    def _on_flight_value(self, value):
        if self._flight is not None:
            self._flight["t"] = float(value)
            self.update()

    def _end_flight(self):
        anim, self._flight_anim = self._flight_anim, None
        if anim is not None:
            try:
                anim.valueChanged.disconnect(self._on_flight_value)
                anim.finished.disconnect(self._end_flight)
                anim.stop()
            except (RuntimeError, TypeError):
                pass
        flight, self._flight = self._flight, None
        if flight is not None:
            self._show_button(flight["index"])
            self.update()

    def _end_drag(self):
        """Settle the carried tab into the slot it ended up in."""
        if self._drag_index < 0:
            return
        index, pix, size = self._drag_index, self._drag_pix, self._drag_size
        start = self._drag_rect()
        self._drag_index = -1
        self._drag_pix = None
        self._drag_size = None
        end = self.tabRect(index)
        self._end_flight()
        self.setCursor(QtCore.Qt.CursorShape.OpenHandCursor)
        if start == end or pix is None:
            self._show_button(index)
            self.update()
            return
        self._begin_flight(index, pix, size, start, end, self._SETTLE_MS)

    # ------------------------------------------------------------------ #
    #  Events
    # ------------------------------------------------------------------ #
    def mousePressEvent(self, event):
        pos = event.position().toPoint()
        self._press_index = self.tabAt(pos)
        self._press_pos = pos
        self._drag_locked = (self._pin_last
                             and self._press_index == self._pinned())
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        pos = event.position().toPoint()
        if event.buttons() == QtCore.Qt.MouseButton.NoButton:
            # Hover: the cursor is the affordance. Open hand over a tab that can
            # actually be carried, plain arrow over the pinned testbench so the
            # user is not invited to drag something that will not move.
            over = self.tabAt(pos)
            pinned = self._pin_last and over == self._pinned()
            self.setCursor(QtCore.Qt.CursorShape.ArrowCursor if pinned
                           else QtCore.Qt.CursorShape.OpenHandCursor)
            super().mouseMoveEvent(event)
            return
        if self._drag_locked:
            # Pinned tab: swallow the move so no drag can start from it.
            return
        if self._drag_index < 0:
            if (self._press_index < 0 or self._press_pos is None
                    or self._press_index >= self._pinned()):
                super().mouseMoveEvent(event)
                return
            if abs(pos.x() - self._press_pos.x()) < QtWidgets.QApplication.startDragDistance():
                super().mouseMoveEvent(event)
                return
            self._start_drag(self._press_index, self._press_pos.x())
        self._track_drag(pos.x())

    def mouseReleaseEvent(self, event):
        was_dragging = self._drag_index >= 0
        self._end_drag()
        self._drag_locked = False
        self._press_index = -1
        self._press_pos = None
        super().mouseReleaseEvent(event)
        if was_dragging:
            # A drag is not a click: keep the tab the user carried selected.
            self.update()

    def leaveEvent(self, event):
        self.unsetCursor()
        super().leaveEvent(event)

    def paintEvent(self, event):
        super().paintEvent(event)
        if self._suspend_overlay:
            return
        if self._drag_index < 0 and self._flight is None:
            return
        painter = QtGui.QPainter(self)
        # Blank the slots of the tabs drawn by hand first, so the styled
        # originals never double up with the moving copies.
        for index in self._moving_indexes():
            painter.fillRect(self.tabRect(index), self._gap_color)
        if self._flight is not None:
            painter.drawPixmap(self._flight_rect(), self._flight["pix"])
        if self._drag_index >= 0 and self._drag_pix is not None:
            # The carried tab rides on top of everything, including a slide.
            painter.drawPixmap(self._drag_rect(), self._drag_pix)
        painter.end()


class HierarchyList(QtWidgets.QListWidget):
    """Compile-order list whose rows can be dragged up and down.

    Every row carries an item *widget* (name + move arrows). Qt's built-in
    ``InternalMove`` drop takes the row out of the model and re-creates it, which
    destroys that widget and leaves a blank row behind -- so the drop is
    intercepted instead: this view never mutates its own model. It reports the
    order the user asked for and the verifier rebuilds (and animates) the rows.
    """

    #: (ordered names, row the dragged/moved item landed on)
    orderChanged = QtCore.pyqtSignal(list, int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setDragEnabled(True)
        self.setAcceptDrops(True)
        self.setDropIndicatorShown(True)
        self.setDragDropMode(QtWidgets.QAbstractItemView.DragDropMode.InternalMove)
        self.setDefaultDropAction(QtCore.Qt.DropAction.MoveAction)
        self.viewport().setAcceptDrops(True)
        self.setSelectionMode(
            QtWidgets.QAbstractItemView.SelectionMode.SingleSelection)

    def names(self):
        return [self.item(i).data(QtCore.Qt.ItemDataRole.UserRole)
                for i in range(self.count())]

    def startDrag(self, actions):
        item = self.currentItem()
        widget = self.itemWidget(item) if item is not None else None
        if widget is None:
            super().startDrag(actions)
            return
        # The delegate paints nothing (all the content lives in the item widget),
        # so Qt's default drag pixmap would be an empty rectangle. Carry a grab
        # of the real row instead, picked up under the cursor where it was held.
        rect = self.visualItemRect(item)
        drag = QtGui.QDrag(self)
        drag.setMimeData(self.model().mimeData([self.indexFromItem(item)]))
        drag.setPixmap(widget.grab())
        drag.setHotSpot(
            self.viewport().mapFromGlobal(QtGui.QCursor.pos()) - rect.topLeft())
        # Ghost the row while it is in flight so the list shows where it came
        # from without duplicating it under the drag pixmap.
        effect = QtWidgets.QGraphicsOpacityEffect(widget)
        effect.setOpacity(0.35)
        widget.setGraphicsEffect(effect)
        try:
            drag.exec(QtCore.Qt.DropAction.MoveAction)
        finally:
            try:
                widget.setGraphicsEffect(None)
            except RuntimeError:
                # Dropped: the row was rebuilt during exec() and this widget is
                # already gone on the C++ side. Nothing to un-ghost.
                pass

    def dropEvent(self, event):
        if event.source() is not self:
            event.ignore()
            return
        names = self.names()
        src = self.currentRow()
        if src < 0 or not names:
            event.ignore()
            return
        pos = event.position().toPoint()
        index = self.indexAt(pos)
        below = QtWidgets.QAbstractItemView.DropIndicatorPosition.BelowItem
        if index.isValid():
            slot = index.row() + (1 if self.dropIndicatorPosition() == below else 0)
        else:
            slot = len(names)          # dropped on the empty space under the rows
        dest = self.drop_row(src, slot, len(names))
        # IgnoreAction, not MoveAction: an accepted move would have the view
        # delete and re-insert the row (and its item widget) behind our back.
        event.setDropAction(QtCore.Qt.DropAction.IgnoreAction)
        event.accept()
        if dest != src:
            names.insert(dest, names.pop(src))
            # Next tick, not now: this runs inside QDrag.exec()'s nested loop,
            # and the rebuild deletes the very item widget the drag is still
            # holding (its grab is the drag pixmap). Let the drag unwind first.
            QtCore.QTimer.singleShot(
                0, lambda n=names, d=dest: self.orderChanged.emit(n, d))

    @staticmethod
    def drop_row(src, slot, count):
        """Row ``src`` lands on, for a drop between rows at gap ``slot``.

        ``slot`` counts gaps (0 == above the first row, ``count`` == past the
        last). A gap below the source is one row further left once the source
        has left its own slot, hence the decrement."""
        dest = slot - 1 if slot > src else slot
        return max(0, min(dest, count - 1))

    def keyPressEvent(self, event):
        # Ctrl+Up / Ctrl+Down: the same reorder from the keyboard, so the order
        # is not mouse-only (and matches the arrow buttons' tooltips).
        ctrl = event.modifiers() & QtCore.Qt.KeyboardModifier.ControlModifier
        keys = (QtCore.Qt.Key.Key_Up, QtCore.Qt.Key.Key_Down)
        if ctrl and event.key() in keys:
            names = self.names()
            src = self.currentRow()
            step = -1 if event.key() == QtCore.Qt.Key.Key_Up else 1
            dest = src + step
            if 0 <= src < len(names) and 0 <= dest < len(names):
                names.insert(dest, names.pop(src))
                self.orderChanged.emit(names, dest)
            event.accept()
            return
        super().keyPressEvent(event)


class VerilogVerifier(QtWidgets.QWidget):
    #: emitted after a compile+simulate run finishes cleanly, so the host
    #: (Flow Navigator) can mark Verify complete and nudge toward Convert.
    simulationSucceeded = QtCore.pyqtSignal()

    #: emitted with a freshly-built VcdPlotWindow when a run produces a
    #: waveform. The host (Flow Navigator -> DockArea) gives the plot a real,
    #: full-width home as its own eSim dock tab, instead of cramming it into the
    #: bottom split. Carries the plot widget; the host owns/parents it.
    waveformReady = QtCore.pyqtSignal(object)

    def __init__(self, parent=None):
        super().__init__(parent)
        
        self.popup_dialog = None
        self.current_timestamps = None
        self.current_signals_data = None
        self.current_signal_types = None

        # Shared design (injected by the Flow Navigator). This stage renders from
        # / collects into it; it never reaches around to disk to sync.
        self.bus = None
        self._rendered_content = None

        # Async run state: the in-flight compile/sim worker thread and its
        # cancel token (None when idle). See _start_job / cancel_running_job.
        self._active_job = None
        self._active_cancel = None
        self._busy_buttons = []
        # Console lines streamed from the run in flight (see _stream_line).
        self._streamed = 0
        # Temp dir of the in-flight run, tracked so it can be reaped even if the
        # stage is torn down mid-run (the done/fail closures may never fire).
        self._active_tmpdir = None
        # Plain mirror of the teardown-critical run state (job / cancel token /
        # tmpdir), read by the destroyed-slot teardown WITHOUT touching this
        # (possibly dying) widget. closeEvent covers explicit close; this covers
        # dock destruction, which skips closeEvent.
        self._teardown_state = {"job": None, "cancel": None, "tmpdir": None}
        self.destroyed.connect(_make_verifier_teardown(self._teardown_state))
        # Filled by _design_sources at each compile: the on-disk source name
        # iverilog will echo in diagnostics -> the editor that holds it. Lets
        # error highlighting / jump-to-line survive labels the diagnostic regex
        # can't carry (spaces, parens — e.g. a duplicate-disambiguated tab).
        self._compile_editors = {}

        self.init_ui()

    def prompt_install_or_path(self, tool_name):
        win_link = "https://bleyer.org/icarus/"
        lin_link = "sudo apt update && sudo apt install -y iverilog"
        
        msg = QtWidgets.QMessageBox(self)
        msg.setWindowTitle(f"{tool_name} Not Found")
        win_hint = r"C:\iverilog\bin"
        lin_hint = "/usr/bin or /usr/local/bin"
        
        full_text = (
            f"The executable '{tool_name}' was not found on your system PATH.\n\n"
            f"You can install it from/using:\n- Windows: {win_link}\n- Linux: {lin_link}\n\n"
            f"(Note: eSim does not endorse or vouch for the security of external links. "
            f"Please verify and download at your own discretion.)\n\n"
            f"If it is already installed but auto-detect failed, you can manually locate the executable '{tool_name}'.\n\n"
            f"Hint: On Windows, it is usually found in {win_hint}. On Linux, check {lin_hint}."
        )
        msg.setText(full_text)
        msg.setTextInteractionFlags(TEXT_SELECTABLE)
        
        btn_select = msg.addButton("Manually Locate Executable", QtWidgets.QMessageBox.ButtonRole.ActionRole)
        msg.addButton(QtWidgets.QMessageBox.StandardButton.Cancel)
        
        msg.exec()
        
        if msg.clickedButton() == btn_select:
            filter_str = f"{tool_name}.exe (*.exe);;All Files (*)" if os.name == 'nt' else f"{tool_name} ({tool_name});;All Files (*)"
            filepath, _ = QtWidgets.QFileDialog.getOpenFileName(self, f"Locate {tool_name} Executable", "", filter_str)
            if filepath:
                return filepath
        return None

    def silent_find_iverilog(self):
        # Single source of truth: CosimConfig resolves env -> ~/.nghdl/config.ini
        # [COSIM] -> bundled library/bin -> PATH -> platform default. Shared with
        # the d_cosim build so both features find the same iverilog, including
        # prefix builds that are not on PATH.
        return CosimConfig.iverilog_binary()

    def silent_find_vvp(self):
        return CosimConfig.vvp_binary()

    def find_iverilog(self):
        path = self.silent_find_iverilog()
        if path:
            return path

        path = self.prompt_install_or_path("iverilog")
        if path:
            # Persist into config.ini (not private QSettings) so the d_cosim
            # build sees the manually-located compiler too.
            CosimConfig.set_manual_paths(iverilog_path=path)
            self.check_iverilog_lock()
            return path
        return None

    def find_vvp(self):
        path = self.silent_find_vvp()
        if path:
            return path

        path = self.prompt_install_or_path("vvp")
        if path:
            CosimConfig.set_manual_paths(vvp_path=path)
            return path
        return None

    def attempt_manual_unlock(self):
        # Try to automatically detect first in case it was just installed
        path = self.silent_find_iverilog()

        if not path:
            path = self.prompt_install_or_path("iverilog")

        if path:
            # Auto-pair vvp from the same dir and persist both in config.ini so
            # d_cosim shares the manually-located toolchain.
            vvp_path = os.path.join(
                os.path.dirname(path), "vvp.exe" if os.name == 'nt' else "vvp")
            CosimConfig.set_manual_paths(
                iverilog_path=path,
                vvp_path=vvp_path if os.path.exists(vvp_path) else None)
            self.check_iverilog_lock()

    def check_iverilog_lock(self):
        if not self.silent_find_iverilog():
            self.lock_ui()
        else:
            self.unlock_ui()

    def lock_ui(self):
        # Disable every editing/build/run affordance in one sweep (buttons,
        # menu actions and the F5 shortcut all share setEnabled); the only live
        # control is "Locate Icarus Verilog".
        for w in self._lockable:
            w.setEnabled(False)

        self.btn_unlock.setVisible(True)

        msg = (
            "Icarus Verilog Dependency Missing\n"
            "---------------------------------\n"
            "The Verilog Simulator requires Icarus Verilog to compile and simulate code.\n\n"
            "Installation Options:\n"
            "- Windows: https://bleyer.org/icarus/\n"
            "  (IMPORTANT: You MUST check the 'Install MinGW Dependencies (DLL Files)' box during installation!)\n"
            "- Linux: sudo apt update && sudo apt install -y iverilog\n\n"
            "(Note: eSim does not endorse or vouch for the security of external links. \n"
            "Please verify and download at your own discretion.)\n\n"
            "Once installed, use the 'Locate Icarus Verilog' button below to enable the tool."
        )
        self.console.setPlainText(msg)
        self._set_console_error(True)

    def unlock_ui(self):
        for w in self._lockable:
            w.setEnabled(True)

        self.btn_unlock.setVisible(False)

        self.console.clear()
        self._set_console_error(False)
        self.log("Icarus Verilog found. Tools ready.")

    def _set_console_error(self, on):
        """Toggle the console between normal and error Aurora styling.

        Swaps the objectName so the global QSS (#verilogConsole vs
        #verilogConsoleError) repaints it; an explicit unpolish/polish is
        needed because Qt only re-evaluates selectors on demand. Replaces the
        old hard-coded inline light stylesheets so the console follows the
        active theme.
        """
        self.console.setObjectName(
            "verilogConsoleError" if on else "verilogConsole")
        self.console.style().unpolish(self.console)
        self.console.style().polish(self.console)

    def init_ui(self):
        # Aurora theme drives the look now: tag the root so the global QSS
        # (#verilogRoot + its child selectors) paints this surface, and let the
        # app sheet style the buttons / tabs / lists / splitters the old
        # hard-light inline stylesheet used to hand-paint. Stripping it lets the
        # panel follow dark/light theme switches instead of staying white.
        self.setObjectName("verilogRoot")

        main_layout = QtWidgets.QVBoxLayout(self)
        # Hug the panel edges: the default layout margin left a wasted gutter
        # (read as a thin idle border) down the left of the Verify surface. The
        # Flow Navigator host already supplies the outer breathing room.
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        main_splitter = QtWidgets.QSplitter(QtCore.Qt.Orientation.Vertical)
        main_layout.addWidget(main_splitter)
        
        # 1. Top widget containing the tabbed editors
        top_container = QtWidgets.QWidget()
        top_container.setObjectName("verilogTopContainer")
        top_layout = QtWidgets.QVBoxLayout(top_container)
        top_layout.setContentsMargins(0, 0, 0, 0)
        
        top_h_splitter = QtWidgets.QSplitter(ORIENT_HORIZ)
        top_h_splitter.setObjectName("verifierTopSplitter")
        top_layout.addWidget(top_h_splitter)
        
        # Sidebar for module hierarchy
        sidebar_widget = QtWidgets.QWidget()
        sidebar_widget.setObjectName("verilogSidebar")
        sidebar_layout = QtWidgets.QVBoxLayout(sidebar_widget)
        sidebar_layout.setContentsMargins(0, 0, 0, 0)
        
        lbl_sidebar = QtWidgets.QLabel("Module Hierarchy")
        # Font comes from QSS (#verilogSidebarTitle). It used to be pinned here
        # to QFont("Segoe UI", 10) -- a hard-coded family and an absolute point
        # size, so this one label ignored the app's type scale, ignored zoom
        # entirely, and rendered in a different face from everything around it.
        lbl_sidebar.setObjectName("verilogSidebarTitle")
        sidebar_layout.addWidget(lbl_sidebar)
        
        self.btn_auto_detect = QtWidgets.QPushButton("Auto-Detect")
        self.btn_auto_detect.setProperty("cssClass", "secondary")
        _no_glow(self.btn_auto_detect)
        self.btn_auto_detect.clicked.connect(self.auto_detect_hierarchy)
        sidebar_layout.addWidget(self.btn_auto_detect)
        
        self.hierarchy_list = HierarchyList()
        self.hierarchy_list.setObjectName("verilogHierarchyList")
        # The QSS white+border-radius styles the frame, but a QListWidget's
        # opaque viewport paints a square white rect over the frame's rounded
        # bottom corners (only the top survives, via the 7px padding inset).
        # Transparent viewport lets the frame's rounded white show on all four
        # corners. No horizontal bar so the corners stay intact.
        self.hierarchy_list.viewport().setStyleSheet("background: transparent;")
        self.hierarchy_list.setHorizontalScrollBarPolicy(
            QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.hierarchy_list.itemClicked.connect(self.hierarchy_single_clicked)
        self.hierarchy_list.itemDoubleClicked.connect(self.hierarchy_double_clicked)
        self.hierarchy_list.setContextMenuPolicy(QtCore.Qt.ContextMenuPolicy.CustomContextMenu)
        self.hierarchy_list.customContextMenuRequested.connect(self.show_hierarchy_context_menu)
        # Drag a row (or Ctrl+Up/Down) -> new compile order, rebuilt + animated.
        self.hierarchy_list.orderChanged.connect(self.apply_hierarchy_order)
        # Live refs for the row-slide animation (see _finish_hierarchy_anim).
        self._hierarchy_anim = None
        self._hierarchy_ghosts = []
        sidebar_layout.addWidget(self.hierarchy_list)
        
        top_h_splitter.addWidget(sidebar_widget)

        self.editor_tabs = QtWidgets.QTabWidget()
        self.editor_tabs.setObjectName("verilogEditorTabs")
        # Chrome-style reorder: design tabs drag left/right (Qt slides the
        # displaced tabs), the testbench stays pinned last. setTabBar must
        # happen before any tab is added.
        self.editor_tabs.setTabBar(PinnedTabBar())
        self.editor_tabs.tabBar().tabMoved.connect(self._on_tab_moved)
        self.editor_tabs.setTabsClosable(True)
        self.editor_tabs.tabCloseRequested.connect(self.close_tab)
        # Double-click used to open the rename dialog. Tab names come from the
        # code now, so there is nothing here to rename by hand.
        # Track the last-active design tab so auto_generate_tb knows which module to target
        self._last_active_design_idx = None
        def _on_tab_changed(idx):
            w = self.editor_tabs.widget(idx)
            if w in getattr(self, 'design_views', []):
                self._last_active_design_idx = idx
        self.editor_tabs.currentChanged.connect(_on_tab_changed)
        
        # Add right-click context menu to the tab bar
        self.editor_tabs.tabBar().setContextMenuPolicy(QtCore.Qt.ContextMenuPolicy.CustomContextMenu)
        self.editor_tabs.tabBar().customContextMenuRequested.connect(self.show_tab_context_menu)
        
        corner_widget = QtWidgets.QWidget()
        corner_layout = QtWidgets.QHBoxLayout(corner_widget)
        corner_layout.setContentsMargins(2, 0, 6, 0)

        # Plain ASCII "+": the ➕ emoji rendered via Segoe UI Emoji on Windows
        # as an off-palette colour glyph that clashed with the theme.
        self.btn_add_module = QtWidgets.QPushButton("+ Add Module")
        # Own objectName + compact #addModuleBtn QSS: the global QPushButton
        # min-height (32) + padding made this taller than the tab bar, so it
        # clipped at the top. The styled rule caps its height to fit the bar.
        self.btn_add_module.setObjectName("addModuleBtn")
        self.btn_add_module.setFlat(True)
        self.btn_add_module.setCursor(QtCore.Qt.CursorShape.PointingHandCursor)
        _no_glow(self.btn_add_module)
        self.btn_add_module.clicked.connect(self.add_module_tab)
        corner_layout.addWidget(self.btn_add_module)

        # No per-widget "Fullscreen" pop-out here: the whole Model Creation
        # panel already has a native fullscreen toggle in the Flow Navigator
        # header, and the waveform now opens as its own full-width eSim tab
        # (see waveformReady), so the editor never needs a popout crutch.
        self.editor_tabs.setCornerWidget(corner_widget)

        top_h_splitter.addWidget(self.editor_tabs)
        # The editor is the work surface; the hierarchy sidebar stays a slim,
        # fixed companion. Stretch 0/1 means every pixel of a wider window goes
        # to the editor, not the sidebar. Sidebar is still drag-collapsible.
        top_h_splitter.setStretchFactor(0, 0)
        top_h_splitter.setStretchFactor(1, 1)
        # Seed the sidebar a touch wider so the hierarchy card's right edge lands
        # under the Verilog|VHDL toggle and its rows fit the index badge + name +
        # both move arrows without feeling cramped. Still drag-resizable.
        # zoom_px, not a bare 232: the QSS metrics inside this sidebar scale
        # with zoom but a Python-set width does not, so at 150% the buttons
        # grew and the box that holds them did not -- which is how "Remove
        # Models" ended up wider than the button drawn around it.
        on_zoom_changed(
            sidebar_widget,
            lambda z, w=sidebar_widget: w.setMinimumWidth(zoom_px(232, z)))
        top_h_splitter.setSizes([zoom_px(248), zoom_px(752)])

        # Design editor (Module 1)
        self.design_views = []
        self.add_module_tab("design.v", DEFAULT_DESIGN)

        # Testbench editor (always pinned to the end)
        self.tb_view = HdlEditor("tb_design.v")
        self.tb_view.filepath = None
        # The startup template is eSim's, not the user's, so it is replaceable
        # (see _testbench_is_ours). Remembering it verbatim is what makes the
        # very first edit the user types count as adoption.
        self._tb_generated_text = DEFAULT_TB
        self.tb_view.setText(DEFAULT_TB)
        self.editor_tabs.addTab(self.tb_view, "Testbench (tb_design.v)")
        # Disable close button for tb_view
        self.editor_tabs.tabBar().setTabButton(self.editor_tabs.count()-1, TAB_RIGHT, None)
        
        # Dynamically update the testbench tab name
        self.tb_view.textChanged.connect(self.update_tb_tab_name)
        self.update_tb_tab_name()

        
        # Controls panel under editors. Two zones, like a real IDE's build bar:
        # file actions fold into one "File" menu on the left (they're occasional
        # and keyboard-driven); the right cluster holds the build/run actions a
        # user actually touches every cycle, ending in the primary Simulate.
        controls_layout = QtWidgets.QHBoxLayout()
        # Breathing room: without margins the bar sat flush between the editor
        # card above and the console card below, reading as clutter.
        controls_layout.setContentsMargins(2, 8, 2, 8)
        controls_layout.setSpacing(8)
        top_layout.addLayout(controls_layout)

        # --- File menu: open/save/export, demoted off the main bar -----------
        # These are QActions so the same object carries the menu entry, its
        # keyboard shortcut, and its enabled state (one source of truth for the
        # lock flow). Standard editor shortcuts so power users never open it.
        self.act_open_source = QtGui.QAction("Open Source .v…", self)
        self.act_open_source.setShortcut(QtGui.QKeySequence.StandardKey.Open)
        self.act_open_source.triggered.connect(self.load_source_files)

        self.act_open_tb = QtGui.QAction("Open Testbench .v…", self)
        self.act_open_tb.triggered.connect(self.load_tb_file)

        self.act_save = QtGui.QAction("Save", self)
        self.act_save.setShortcut(QtGui.QKeySequence.StandardKey.Save)
        self.act_save.triggered.connect(self.save_file)

        self.act_save_as = QtGui.QAction("Save As…", self)
        # Explicit key: StandardKey.SaveAs is unbound on Windows and on the
        # Qt 6.4 (Ubuntu 24.04) fallback theme, leaving the action shortcutless.
        self.act_save_as.setShortcut(QtGui.QKeySequence("Ctrl+Shift+S"))
        self.act_save_as.triggered.connect(self.save_as_file)

        # Export is rare and only meaningful after a sim; it lives in the menu,
        # disabled until a run produces data (see render_waveform / export_csv).
        self.act_export_csv = QtGui.QAction("Export Waveform CSV…", self)
        self.act_export_csv.triggered.connect(self.export_csv)
        self.act_export_csv.setEnabled(False)

        file_menu = QtWidgets.QMenu(self)
        file_menu.addAction(self.act_open_source)
        file_menu.addAction(self.act_open_tb)
        file_menu.addSeparator()
        file_menu.addAction(self.act_save)
        file_menu.addAction(self.act_save_as)
        file_menu.addSeparator()
        file_menu.addAction(self.act_export_csv)
        # The shortcuts must fire from anywhere in the panel, not only while the
        # (collapsed) menu is open, so attach the actions to the widget too.
        self.addActions([self.act_open_source, self.act_open_tb,
                         self.act_save, self.act_save_as])

        self.file_button = QtWidgets.QToolButton()
        self.file_button.setObjectName("verifierFileBtn")
        self.file_button.setText("File")
        self.file_button.setMenu(file_menu)
        self.file_button.setPopupMode(TOOLBTN_INSTANT)
        controls_layout.addWidget(self.file_button)

        sep = QtWidgets.QFrame()
        sep.setFrameShape(QtWidgets.QFrame.Shape.VLine)
        sep.setFrameShadow(QtWidgets.QFrame.Shadow.Sunken)
        controls_layout.addWidget(sep)

        # --- Build-prep actions (visible, secondary) -------------------------
        self.btn_syntax = QtWidgets.QPushButton("Check Syntax")
        self.btn_syntax.setProperty("cssClass", "secondary")
        self.btn_syntax.clicked.connect(self.check_syntax)
        controls_layout.addWidget(self.btn_syntax)

        self.btn_stub = QtWidgets.QPushButton("Auto-Generate Testbench")
        self.btn_stub.setProperty("cssClass", "secondary")
        self.btn_stub.clicked.connect(self.auto_generate_tb)
        controls_layout.addWidget(self.btn_stub)

        # Push the run cluster to the right edge -- the eye lands on Simulate.
        controls_layout.addStretch(1)

        # Shown only when iverilog can't be found; lets the user point at it.
        self.btn_unlock = QtWidgets.QPushButton("Locate Icarus Verilog…")
        self.btn_unlock.setProperty("cssClass", "danger")
        self.btn_unlock.clicked.connect(self.attempt_manual_unlock)
        self.btn_unlock.setVisible(False)
        controls_layout.addWidget(self.btn_unlock)

        # Live run status. A compile+simulate can legitimately take minutes on
        # a big design, and with only a greyed-out button to look at, "running"
        # and "hung" are indistinguishable -- which is what the freeze reports
        # were really about. A moving bar plus a phase label and an elapsed
        # clock says which of the two it is, every second.
        self.lbl_progress = QtWidgets.QLabel("")
        self.lbl_progress.setObjectName("verifierRunStatus")
        self.lbl_progress.setVisible(False)
        controls_layout.addWidget(self.lbl_progress)

        self.bar_progress = QtWidgets.QProgressBar()
        self.bar_progress.setObjectName("verifierRunBar")
        self.bar_progress.setRange(0, 0)        # indeterminate: it is running
        self.bar_progress.setTextVisible(False)
        self.bar_progress.setFixedWidth(zoom_px(110))
        self.bar_progress.setVisible(False)
        controls_layout.addWidget(self.bar_progress)

        # Drives the elapsed clock in lbl_progress while a job is in flight.
        self._run_timer = QtCore.QTimer(self)
        self._run_timer.setInterval(1000)
        self._run_timer.timeout.connect(self._tick_progress)
        self._run_started = None
        self._run_phase = ""

        # Shown only while a compile/sim runs on the worker thread; clicking it
        # kills the underlying iverilog/vvp process via the active CancelToken.
        self.btn_cancel = QtWidgets.QPushButton("Cancel")
        self.btn_cancel.clicked.connect(self.cancel_running_job)
        self.btn_cancel.setVisible(False)
        controls_layout.addWidget(self.btn_cancel)

        # The primary action: accented and right-anchored. F5 = run, the near-
        # universal "go" key, so it works without reaching for the mouse.
        # U+25B8 (small triangle) always takes the monochrome text glyph;
        # U+25B6 got promoted to a colour emoji by Segoe UI Emoji on Windows.
        self.btn_simulate = QtWidgets.QPushButton("▸  Simulate")
        # Aurora paints the accented primary button via the verifierPrimary
        # cssClass (replaces the old #primaryAction inline-styled green).
        self.btn_simulate.setProperty("cssClass", "verifierPrimary")
        self.btn_simulate.clicked.connect(self.simulate_and_wave)
        controls_layout.addWidget(self.btn_simulate)

        self.sc_simulate = QtGui.QShortcut(QtGui.QKeySequence("F5"), self)
        self.sc_simulate.activated.connect(self.simulate_and_wave)

        # No "Send to Makerchip" button: the design is shared with the Author
        # and Convert stages through the Flow Navigator, which syncs it
        # automatically on tab switch (see render_from_bus / collect_into_bus).
        # There is nothing to hand off.

        # Single source of truth for the iverilog lock: everything that must be
        # dead while the toolchain is missing. QWidgets, QActions and QShortcut
        # all share setEnabled(), so one loop in lock_ui/unlock_ui covers them.
        self._lockable = [
            self.editor_tabs, self.hierarchy_list, self.btn_add_module,
            self.btn_auto_detect, self.btn_syntax, self.btn_stub,
            self.btn_simulate, self.file_button, self.sc_simulate,
            self.act_open_source, self.act_open_tb,
            self.act_save, self.act_save_as,
        ]

        main_splitter.addWidget(top_container)
        
        # Find/Replace Toolbar
        self.find_widget = QtWidgets.QWidget()
        self.find_widget.setObjectName("verilogFindBar")
        find_layout = QtWidgets.QHBoxLayout(self.find_widget)
        find_layout.setContentsMargins(0, 0, 0, 0)
        
        self.find_input = QtWidgets.QLineEdit()
        self.find_input.setPlaceholderText("Find...")
        self.replace_input = QtWidgets.QLineEdit()
        self.replace_input.setPlaceholderText("Replace with...")
        
        self.btn_find_next = QtWidgets.QPushButton("Find Next")
        self.btn_find_next.clicked.connect(self.find_next)
        
        self.btn_replace = QtWidgets.QPushButton("Replace")
        self.btn_replace.clicked.connect(self.replace_text)
        
        self.btn_close_find = QtWidgets.QPushButton("X")
        self.btn_close_find.setFixedWidth(zoom_px(30))
        self.btn_close_find.clicked.connect(self.find_widget.hide)
        
        find_layout.addWidget(self.find_input)
        find_layout.addWidget(self.replace_input)
        find_layout.addWidget(self.btn_find_next)
        find_layout.addWidget(self.btn_replace)
        find_layout.addWidget(self.btn_close_find)
        
        self.find_widget.hide()
        top_layout.addWidget(self.find_widget)
        
        shortcut = QtGui.QShortcut(QtGui.QKeySequence("Ctrl+F"), self)
        shortcut.activated.connect(self.show_find_toolbar)
        
        # 2. Bottom band: the console log. The waveform used to live beside it
        #    in a horizontal split; it now opens as its own full-width eSim tab
        #    (waveformReady), so the console is this band's only child and goes
        #    straight into the splitter -- no wrapper container needed.
        # Console output styled like Vivado TCL Console
        self.console_tabs = QtWidgets.QTabWidget()
        self.console_tabs.setObjectName("verilogBottomContainer")
        self.console = ConsoleEdit()
        self.console.setReadOnly(True)
        self.console.setPlaceholderText("Console logs will appear here...\nDouble-click a syntax error (e.g. design.v:5: error) to jump directly to the line.")
        # No setFont here: the mono family and size come from #verilogConsole in
        # the QSS, which is the only path that tracks zoom. A hard-coded
        # QFont("Consolas", 11) stayed 11pt at every zoom level.
        self.console.error_clicked.connect(self.jump_to_error)
        # Aurora styles the console via #verilogConsole (and #verilogConsoleError
        # for the error state, toggled in _set_console_error). No inline sheet,
        # so it tracks dark/light theme switches.
        self.console.setObjectName("verilogConsole")
        self.console_tabs.addTab(self.console, "Console Output")

        # Copy lives in the console tab's corner now that there is no popout
        # bar. Text-only: the 📋 emoji drew as a loud colour clipboard on
        # Windows, out of place on a quiet corner action.
        self.btn_copy_console = QtWidgets.QPushButton("Copy")
        self.btn_copy_console.setObjectName("verifierConsoleCopy")
        self.btn_copy_console.setFlat(True)
        self.btn_copy_console.setCursor(QtCore.Qt.CursorShape.PointingHandCursor)
        _no_glow(self.btn_copy_console)
        self.btn_copy_console.clicked.connect(lambda: QtWidgets.QApplication.clipboard().setText(self.console.toPlainText()))
        self.console_tabs.setCornerWidget(self.btn_copy_console)

        main_splitter.addWidget(self.console_tabs)

        # Editor dominates; the console keeps a calm, roughly-fixed band and the
        # editor absorbs all extra height as the window grows (Vivado/VS Code
        # behaviour). Stretch factors -- not magic pixels -- drive the resize;
        # the initial split only seeds a sensible ~3:1 starting proportion.
        main_splitter.setStretchFactor(0, 1)
        main_splitter.setStretchFactor(1, 0)
        main_splitter.setSizes([600, 200])

        # Check for Icarus Verilog on boot and lock if missing
        self.check_iverilog_lock()

        # Aurora micro-interactions for the IDE: button glow, tab-bar kinetics
        # and context-menu motion. All gated on the motion pref (no-op off).
        try:
            from frontEnd.motion import (
                install_button_motion, motion_enabled, apply_panel_depth,
                install_tab_kinetics, install_context_menu_motion)
            # Static panel depth is always on (matches the design: only the
            # animated glow / kinetics are perf-gated behind the motion pref).
            # NOT on the tab widgets: a QTabBar is transparent between the tabs'
            # rounded top corners, so the shadow rendered behind the widget
            # showed *through* those notches as a hard tick above every tab
            # boundary -- dark on the light theme, pale on the dark one. The
            # tab cards get their depth from their border + pane instead.
            apply_panel_depth(self.hierarchy_list, blur=28, y=8, alpha=82)
            if motion_enabled():
                install_button_motion(self)
                install_tab_kinetics(self)
                install_context_menu_motion(self)
                # ...except on the editor tab bar. The kinetic filter puts an
                # accent drop-shadow on the whole bar while the mouse is down,
                # which is exactly the shadow-through-the-notches tick again --
                # now flashing on every press and behind every drag frame. This
                # bar paints its own drag feedback and sets its own cursors, so
                # it needs nothing from the filter.
                kinetics = getattr(self, "_esim_tab_kinetic_filter", None)
                if kinetics is not None:
                    self.editor_tabs.tabBar().removeEventFilter(kinetics)
        except Exception:
            pass

    def _unique_tab_name(self, name):
        """Disambiguate a tab label so no two design tabs share one. Tab labels
        are the keys the hierarchy list and get_design_code match on, so a
        collision (e.g. loading two ``top.v`` from different folders) would
        otherwise drop a module. ``foo.v`` -> ``foo (2).v``."""
        existing = {self.editor_tabs.tabText(i)
                    for i in range(self.editor_tabs.count())}
        if name not in existing:
            return name
        stem, dot, ext = name.rpartition('.')
        base, suffix = (stem, '.' + ext) if dot else (name, '')
        n = 2
        while f"{base} ({n}){suffix}" in existing:
            n += 1
        return f"{base} ({n}){suffix}"

    def add_module_tab(self, name=None, content="", filepath=None):
        if not name:
            name = f"module_{len(self.design_views) + 1}.v"
        # The code decides the label, not the other way round: a tab called
        # design.v holding `module counter` taught users that the two names
        # were separate things they had to reconcile, which is exactly the
        # confusion that made renaming a tab look like it should do something.
        name = self._module_tab_name(content) or name
        name = self._unique_tab_name(name)

        # Name the editor after the tab so the lexer (Verilog vs VHDL) is chosen
        # from its extension, matching the project code editor.
        editor = HdlEditor(name)
        editor.filepath = filepath
        editor.setText(content)

        # Append BEFORE insertTab so _on_tab_changed sees it immediately
        self.design_views.append(editor)

        # Insert before testbench tab
        tb_index = self.editor_tabs.count() - 1 if self.editor_tabs.count() > 0 else 0
        self.editor_tabs.insertTab(tb_index, editor, name)
        self.editor_tabs.setCurrentWidget(editor)
        editor.setToolTip(
            "Named after the module in this file. Rename the module in the "
            "code and the tab follows.")
        # Keep following the code as it is edited, exactly as the testbench tab
        # already does (update_tb_tab_name).
        editor.textChanged.connect(
            lambda e=editor: self.update_module_tab_name(e))

        self.update_hierarchy_list()

    @staticmethod
    def _module_tab_name(code):
        """``<top module>.v`` for ``code``, or "" when nothing parses yet.

        Name only (top_module_name), not extract_ports: this runs on every
        keystroke, and hdlparse's port pass is heavy enough to be felt as
        typing lag on a large design."""
        name = top_module_name(code or "")
        return f"{name}.v" if name else ""

    def update_module_tab_name(self, editor):
        """Re-label a design tab after the module it now contains.

        Silent while the design is being typed and has no readable module --
        the tab simply keeps its current name rather than flickering through
        half-written ones."""
        idx = self.editor_tabs.indexOf(editor)
        if idx < 0 or idx == self.editor_tabs.count() - 1:
            return                          # gone, or the pinned testbench
        wanted = self._module_tab_name(editor.toPlainText())
        if not wanted or wanted == self.editor_tabs.tabText(idx):
            return
        self.editor_tabs.setTabText(idx, self._unique_tab_name(wanted))
        self.update_hierarchy_list()

    def _on_tab_moved(self, _from, _to):
        """Keep the model in step after a tab was dragged to a new slot.

        Two jobs. Re-pin the testbench last if a drag somehow got past
        PinnedTabBar's clamp (every ``count() - 1`` test in this file depends on
        it). Then mirror the new on-screen order into ``design_views``, which is
        what ``design_views[-1]`` and the get_design_code fallback read -- the
        compile order itself comes from the bar, so it already followed."""
        if getattr(self, '_repinning', False):
            return
        bar = self.editor_tabs.tabBar()
        last = self.editor_tabs.count() - 1
        tb_view = getattr(self, 'tb_view', None)
        tb = self.editor_tabs.indexOf(tb_view) if tb_view is not None else -1
        if 0 <= tb < last:
            self._repinning = True
            try:
                bar.moveTab(tb, last)
            finally:
                self._repinning = False
        views = set(getattr(self, 'design_views', []))
        self.design_views = [self.editor_tabs.widget(i)
                             for i in range(self.editor_tabs.count())
                             if self.editor_tabs.widget(i) in views]
        current = self.editor_tabs.currentIndex()
        if self.editor_tabs.widget(current) in views:
            self._last_active_design_idx = current

    def close_tab(self, index):
        # Prevent closing Testbench
        if index == self.editor_tabs.count() - 1:
            return

        widget = self.editor_tabs.widget(index)
        if widget in self.design_views:
            self.design_views.remove(widget)
        self.editor_tabs.removeTab(index)
        # Reset last-active design index if the closed tab was the remembered one
        if getattr(self, '_last_active_design_idx', None) is not None:
            if self._last_active_design_idx == index:
                self._last_active_design_idx = None
            elif self._last_active_design_idx > index:
                self._last_active_design_idx -= 1
        self.update_hierarchy_list()
        
    # NOTE: there is deliberately no "Rename Module" action any more. It only
    # ever called setTabText -- it did not touch the module in the code, the
    # file on disk, or the model the design builds into -- so renaming a tab
    # looked like it had worked and changed nothing at all. Tab labels are now
    # derived from the code (update_module_tab_name), which makes renaming the
    # module in the source the one way to rename anything, and makes it rename
    # everything.

    def show_tab_context_menu(self, pos):
        index = self.editor_tabs.tabBar().tabAt(pos)
        if index >= 0 and index < self.editor_tabs.count() - 1: # Prevent operations on testbench
            menu = QtWidgets.QMenu(self)
            delete_action = menu.addAction("Delete Module")
            action = menu.exec(self.editor_tabs.tabBar().mapToGlobal(pos))
            if action == delete_action:
                self.close_tab(index)

    def hierarchy_single_clicked(self, item):
        name = item.data(QtCore.Qt.ItemDataRole.UserRole)
        # Switch to the matching module tab
        for i in range(self.editor_tabs.count()):
            if self.editor_tabs.tabText(i) == name:
                self.editor_tabs.setCurrentIndex(i)
                break

    def hierarchy_double_clicked(self, item):
        """Double-click focuses the module's tab (it used to open the rename
        dialog that could not actually rename anything)."""
        self.hierarchy_single_clicked(item)

    def show_hierarchy_context_menu(self, pos):
        item = self.hierarchy_list.itemAt(pos)
        if item:
            name = item.data(QtCore.Qt.ItemDataRole.UserRole)
            menu = QtWidgets.QMenu(self)
            delete_action = menu.addAction("Delete Module")
            action = menu.exec(self.hierarchy_list.mapToGlobal(pos))
            if action == delete_action:
                # Find index in editor_tabs
                for i in range(self.editor_tabs.count()):
                    if self.editor_tabs.tabText(i) == name:
                        self.close_tab(i)
                        break

    def move_hierarchy_item(self, item, direction):
        row = self.hierarchy_list.row(item)
        names = self.hierarchy_list.names()
        dest = row - 1 if direction == "up" else row + 1
        if row < 0 or not (0 <= dest < len(names)):
            return
        names.insert(dest, names.pop(row))
        self.apply_hierarchy_order(names, dest)

    def apply_hierarchy_order(self, names, focus_row=None):
        """Re-render the hierarchy in ``names`` order and slide the rows there.

        The single entry point for every reorder -- move arrows, row drag,
        Ctrl+Up/Down. The list itself is rebuilt outright (item widgets cannot
        survive a model move), so without the slide a reorder is a silent
        repaint: the row lands in its new place with nothing to say it moved.
        Each row that changed position is covered by a snapshot of where it used
        to be, and that snapshot is animated to where the row now is."""
        before = self._snapshot_hierarchy_rows()
        self.update_hierarchy_list(names)
        if focus_row is not None and 0 <= focus_row < self.hierarchy_list.count():
            self.hierarchy_list.setCurrentRow(focus_row)
            self.hierarchy_list.setFocus()
        if before:
            self._animate_hierarchy_rows(before)

    def _snapshot_hierarchy_rows(self):
        """{name: (rect, pixmap)} for the rows on screen right now, or {} when
        the slide should be skipped (motion off, panel hidden, nothing laid out
        yet -- a grab of an unmapped widget is an empty pixmap)."""
        # Any slide still running belongs to the pre-rebuild rows: finish it
        # while those widgets are alive, or its cleanup would touch corpses.
        self._finish_hierarchy_anim()
        try:
            from frontEnd.motion import motion_enabled
            if not motion_enabled():
                return {}
        except Exception:
            pass
        if not self.hierarchy_list.isVisible():
            return {}
        rows = {}
        for i in range(self.hierarchy_list.count()):
            item = self.hierarchy_list.item(i)
            widget = self.hierarchy_list.itemWidget(item)
            rect = self.hierarchy_list.visualItemRect(item)
            if widget is None or rect.isEmpty():
                continue
            rows[item.data(QtCore.Qt.ItemDataRole.UserRole)] = (
                QtCore.QRect(rect), widget.grab())
        return rows

    def _animate_hierarchy_rows(self, before):
        """FLIP the rebuilt rows: park a snapshot of each moved row at its old
        geometry, hide the real row, and animate the snapshot into place."""
        viewport = self.hierarchy_list.viewport()
        group = QtCore.QParallelAnimationGroup(self.hierarchy_list)
        for i in range(self.hierarchy_list.count()):
            item = self.hierarchy_list.item(i)
            name = item.data(QtCore.Qt.ItemDataRole.UserRole)
            if name not in before:
                continue
            old_rect, pixmap = before[name]
            new_rect = self.hierarchy_list.visualItemRect(item)
            if new_rect == old_rect or new_rect.isEmpty():
                continue
            ghost = QtWidgets.QLabel(viewport)
            ghost.setPixmap(pixmap)
            ghost.setGeometry(old_rect)
            ghost.show()
            ghost.raise_()
            widget = self.hierarchy_list.itemWidget(item)
            if widget is not None:
                widget.hide()
            self._hierarchy_ghosts.append((ghost, widget))
            anim = QtCore.QPropertyAnimation(ghost, b"geometry", group)
            anim.setDuration(190)
            anim.setStartValue(old_rect)
            anim.setEndValue(new_rect)
            # Fast out of the old slot, settling into the new one -- the same
            # curve the rest of the Aurora motion uses.
            anim.setEasingCurve(QtCore.QEasingCurve.Type.OutCubic)
            group.addAnimation(anim)
        if group.animationCount() == 0:
            self._finish_hierarchy_anim()
            group.deleteLater()
            return
        self._hierarchy_anim = group
        group.finished.connect(self._finish_hierarchy_anim)
        group.start()

    def _finish_hierarchy_anim(self):
        """Drop the ghosts and un-hide the real rows. Idempotent, and safe to
        call after a rebuild has already deleted the widgets it restores."""
        group, self._hierarchy_anim = self._hierarchy_anim, None
        if group is not None:
            try:
                group.finished.disconnect(self._finish_hierarchy_anim)
                group.stop()
                group.deleteLater()
            except (RuntimeError, TypeError):
                pass
        ghosts, self._hierarchy_ghosts = self._hierarchy_ghosts, []
        for ghost, widget in ghosts:
            for w in (ghost, widget):
                if w is None:
                    continue
                try:
                    if w is widget:
                        w.show()
                    else:
                        w.deleteLater()
                except RuntimeError:
                    # Rebuilt out from under us; the C++ side is already gone.
                    pass

    def update_hierarchy_list(self, sorted_names=None):
        self.hierarchy_list.clear()
        
        # If sorted_names provided, use that order. Otherwise, use tab order.
        if sorted_names:
            names = sorted_names
        else:
            names = []
            for i in range(self.editor_tabs.count()):
                # Only include tabs that are in design_views (excludes testbench)
                if hasattr(self, 'design_views') and self.editor_tabs.widget(i) in self.design_views:
                    names.append(self.editor_tabs.tabText(i))
                
        for row, name in enumerate(names):
            item = QtWidgets.QListWidgetItem()
            item.setData(QtCore.Qt.ItemDataRole.UserRole, name)
            widget = QtWidgets.QWidget()
            widget.setObjectName("hierarchyRow")
            # Open hand over the row body: the row is draggable, and nothing
            # else on it says so.
            widget.setCursor(QtCore.Qt.CursorShape.OpenHandCursor)
            widget.setToolTip("Drag to reorder — this is the compile order")
            layout = QtWidgets.QHBoxLayout(widget)
            # Zero vertical margin: the row fills the item rect exactly (item has
            # no vertical padding), so the hover/selected pill hugs the content
            # instead of floating taller than the row.
            layout.setContentsMargins(12, 0, 10, 0)
            layout.setSpacing(8)

            lbl = QtWidgets.QLabel(name)
            lbl.setObjectName("hierarchyName")

            # Aurora styles these tiny move buttons via #hierarchyMoveBtn (size,
            # padding and colour), so no inline override is needed. NoFocus kills
            # the focus ring the last-clicked arrow otherwise kept.
            vcenter = QtCore.Qt.AlignmentFlag.AlignVCenter

            # Solid ▲/▼ in a 26px hit target: the old ▴/▾ at 22px drew a
            # 5px-tall hairline glyph that was hard to see and harder to hit.
            btn_up = QtWidgets.QPushButton("▲")
            btn_up.setObjectName("hierarchyMoveBtn")
            btn_up.setFixedSize(zoom_px(26), zoom_px(26))
            btn_up.setFocusPolicy(QtCore.Qt.FocusPolicy.NoFocus)
            btn_up.setCursor(QtCore.Qt.CursorShape.PointingHandCursor)
            btn_up.setToolTip("Move up (Ctrl+Up)")
            # Greyed out at the ends of the list: the honest answer to "did my
            # click do anything?" when there is nowhere left to go.
            btn_up.setEnabled(row > 0)
            _no_glow(btn_up)
            btn_up.clicked.connect(lambda checked, i=item: self.move_hierarchy_item(i, "up"))

            btn_down = QtWidgets.QPushButton("▼")
            btn_down.setObjectName("hierarchyMoveBtn")
            btn_down.setFixedSize(zoom_px(26), zoom_px(26))
            btn_down.setFocusPolicy(QtCore.Qt.FocusPolicy.NoFocus)
            btn_down.setCursor(QtCore.Qt.CursorShape.PointingHandCursor)
            btn_down.setToolTip("Move down (Ctrl+Down)")
            btn_down.setEnabled(row < len(names) - 1)
            _no_glow(btn_down)
            btn_down.clicked.connect(lambda checked, i=item: self.move_hierarchy_item(i, "down"))

            # Explicit AlignVCenter on every cell: with fixed-size buttons a bare
            # addWidget lets the layout stagger the two arrows at different Y
            # (the "displaced down button"). Pin them all to the row centre.
            layout.addWidget(lbl, 0, vcenter)
            layout.addStretch()
            layout.addWidget(btn_up, 0, vcenter)
            layout.addWidget(btn_down, 0, vcenter)

            # Row height == item sizeHint height (set below) so the selection
            # pill and the row are pixel-identical.
            widget.setMinimumHeight(36)
            item.setSizeHint(QtCore.QSize(0, 36))

            self.hierarchy_list.addItem(item)
            self.hierarchy_list.setItemWidget(item, widget)

    def auto_detect_hierarchy(self):
        # Order the design tabs top-down (a module before the modules it
        # instantiates). The heuristic -- comment/string stripping, balanced
        # #(...) param overrides, duplicate-name and cycle tolerance -- lives in
        # the pure, tested ``order_modules``; this method only feeds it the tab
        # labels + code and renders the result.
        named_codes = []
        for i in range(self.editor_tabs.count()):
            widget = self.editor_tabs.widget(i)
            if widget not in getattr(self, 'design_views', []):
                continue
            named_codes.append((self.editor_tabs.tabText(i),
                                widget.toPlainText()))
        if not named_codes:
            self.update_hierarchy_list([])
            return
        self.update_hierarchy_list(order_modules(named_codes))

    # Console palette. One colour per semantic level so runs read like a real
    # tool console instead of a uniform gray wall.
    _LOG_COLORS = {
        'info':    "#57606A",   # tool chatter: paths, progress, load/save
        'ok':      "#1A7F37",   # outcomes that mean "you're good"
        'error':   "#CF222E",   # failures
        'warn':    "#9A6700",   # cancellations, notes, hints
        'head':    "#0969DA",   # section headings (Diagnostics:)
        'output':  "#24292E",   # raw simulator stdout ($display et al.)
    }

    def _append_console(self, text, color, bold=False):
        cursor = self.console.textCursor()
        cursor.movePosition(QtGui.QTextCursor.MoveOperation.End)
        fmt = QtGui.QTextCharFormat()
        fmt.setForeground(QtGui.QColor(color))
        if bold:
            fmt.setFontWeight(QtGui.QFont.Weight.DemiBold)
        cursor.insertText(text.rstrip("\n") + "\n", fmt)
        self.console.setTextCursor(cursor)
        self.console.ensureCursorVisible()

    @staticmethod
    def _classify_log(text):
        """Infer a semantic level from the message so the ~30 existing call
        sites colour themselves without each one being touched."""
        t = text.strip().lower()
        if t.startswith(("error", "failed")) or " error:" in t \
                or t.endswith(("errors:", "failed:")):
            return 'error'
        if t.startswith("note:") or "cancell" in t:
            return 'warn'
        if t.startswith(("syntax ok", "waveform ready", "successfully",
                         "generated testbench")) or t.endswith("tools ready."):
            return 'ok'
        if t.endswith(":") and " " not in t:   # bare headings e.g. "Diagnostics:"
            return 'head'
        return 'info'

    def log(self, text, level=None):
        """Append a status line, coloured by semantic level. When level is
        omitted it is inferred from the message text."""
        level = level or self._classify_log(text)
        self._append_console(text, self._LOG_COLORS[level],
                             bold=(level in ('ok', 'error', 'head')))

    def log_section(self, title):
        """Visual break between runs: a dim rule with the action name, so
        consecutive Check Syntax / Simulate runs don't blur together."""
        stamp = datetime.datetime.now().strftime("%H:%M:%S")
        if self.console.toPlainText().strip():
            self._append_console("", self._LOG_COLORS['info'])
        self._append_console(f"─── {title} · {stamp} " + "─" * 40,
                             self._LOG_COLORS['info'])

    def log_sim(self, text):
        """Append raw tool output, colouring diagnostics per line: error lines
        red, warnings amber, everything else full contrast."""
        for line in text.rstrip("\n").split("\n"):
            low = line.lower()
            if "error" in low:
                self._append_console(line, self._LOG_COLORS['error'])
            elif "warning" in low:
                self._append_console(line, self._LOG_COLORS['warn'])
            else:
                self._append_console(line, self._LOG_COLORS['output'])

    def clear_waveform_state(self):
        """Reset the 'have waveform data' state. The plot itself now lives in a
        separate eSim tab owned by the host, so there is no inline view to
        clear -- only the Export CSV affordance to disable."""
        if hasattr(self, 'act_export_csv'):
            self.act_export_csv.setEnabled(False)

    def get_current_editor(self):
        return self.editor_tabs.currentWidget()
        
    def show_find_toolbar(self):
        self.find_widget.show()
        self.find_input.setFocus()
        self.find_input.selectAll()
        
    def find_next(self):
        text = self.find_input.text()
        if not text:
            return
        editor = self.get_current_editor()
        # findFirst(expr, regex, case-sensitive, whole-word, wrap, forward).
        # wrap=True handles the wrap-around the old manual cursor reset did.
        editor.findFirst(text, False, False, False, True)

    def replace_text(self):
        text_to_find = self.find_input.text()
        replacement = self.replace_input.text()
        if not text_to_find:
            return

        editor = self.get_current_editor()
        # If the current selection is already the search hit, replace it; then
        # advance to the next match (mirrors the old behaviour).
        if editor.hasSelectedText() and editor.selectedText() == text_to_find:
            editor.replace(replacement)
        self.find_next()

    def jump_to_error(self, line_num, filename):
        """Jump to the error line for the source iverilog named.

        Resolve the editor by the compile-time source name first (what the
        diagnostic actually carries — sanitised, possibly unlike the visible
        label), then the exact tab label, then the canonical testbench aliases.
        Give up rather than fall back to design_views[0]: a wrong jump is worse
        than no jump.
        """
        editor = None
        tab_idx = -1
        basename = os.path.basename(filename)

        # Step 1: the compile-time name map (what iverilog actually echoes —
        # sanitised, so it may differ from the visible tab label).
        editor = getattr(self, '_compile_editors', {}).get(basename)

        # Step 2: exact tab label match (handles any module filename)
        if editor is None:
            for i in range(self.editor_tabs.count()):
                if self.editor_tabs.tabText(i) == basename:
                    editor = self.editor_tabs.widget(i)
                    break

        # Step 3: canonical testbench aliases that iverilog may use
        if editor is None and basename in ('tb.v', 'tb_design.v'):
            editor = self.tb_view

        if editor is not None:
            tab_idx = self.editor_tabs.indexOf(editor)

        # Do not fall back to design_views[0] — a wrong jump is worse than no jump
        if editor is None:
            return

        self.editor_tabs.setCurrentIndex(tab_idx)
        editor.mark_error_line(line_num)
        editor.goto_line(line_num)

    def highlight_errors_from_log(self, log_text, hints=True):
        """Squiggle the error lines in every tab, matched by filename in the log.

        ``hints=False`` suppresses the generic :meth:`analyze_syntax_error`
        advice for callers that have already said something more specific about
        this failure."""
        # Clear all existing error highlights first
        for v in getattr(self, 'design_views', []):
            v.clear_error_highlights()
        self.tb_view.clear_error_highlights()

        # Map the on-disk source name iverilog echoes -> the editor holding it.
        # _compile_editors is authoritative (it IS the name written at compile
        # time, sanitised so the regex below can carry it). Tab labels are added
        # only as a fallback; 'design.v' is intentionally NOT registered — both
        # run paths write per-tab named files, so a generic 'design.v' would be
        # stale and must not silently redirect to design_views[0].
        tab_editors = dict(getattr(self, '_compile_editors', {}))
        tab_editors['tb.v'] = self.tb_view
        tab_editors['tb_design.v'] = self.tb_view
        for i in range(self.editor_tabs.count()):
            tab_editors.setdefault(self.editor_tabs.tabText(i),
                                   self.editor_tabs.widget(i))

        matches = re.finditer(r'([\w./-]+\.(?:v|sv)):(\d+):', log_text)
        for match in matches:
            filename = match.group(1)
            line_num = int(match.group(2))
            # Try to find the editor; look by basename too
            editor = tab_editors.get(filename) or tab_editors.get(os.path.basename(filename))
            if editor is None:
                continue
            editor.mark_error_line(line_num)

        if hints:
            self.analyze_syntax_error(log_text)

    def analyze_syntax_error(self, log_text):
        hints = []
        log_lower = log_text.lower()
        if "syntax error" in log_lower:
            hints.append("note: check for missing ';', misspelled keyword, or unclosed module")
        if "unknown module type" in log_lower:
            hints.append("note: module instantiated but not defined — check spelling and that all source files are loaded")
        if "is not a valid l-value" in log_lower:
            hints.append("note: cannot drive a wire from an 'always' block; declare as 'reg', or use 'assign' for combinational logic")
        if "undeclared identifier" in log_lower or "not declared" in log_lower:
            hints.append("note: signal used but not declared — add a 'wire' or 'reg' declaration")
        if "unconnected" in log_lower and "port" in log_lower:
            hints.append("note: unconnected port on module instance — verify port map is complete")
        if "expecting" in log_lower and "endmodule" in log_lower:
            hints.append("note: incomplete module structure — every 'module' must have a matching 'endmodule'")
        if "multiple drivers" in log_lower:
            hints.append("note: signal driven from multiple sources — check for conflicting 'always' blocks or 'assign' statements")

        if hints:
            self.log("Diagnostics:")
            for hint in hints:
                self.log("  " + hint)


    @staticmethod
    def _read_text(filepath):
        """Read an HDL source file as text. Prefers UTF-8 but falls back to
        latin-1 (which can't raise on byte content) so a file saved in a legacy
        encoding loads -- mojibake the user can see and fix beats a hard failure
        and an empty tab."""
        with open(filepath, 'rb') as f:
            raw = f.read()
        try:
            return raw.decode('utf-8')
        except UnicodeDecodeError:
            return raw.decode('latin-1')

    def load_source_files(self):
        filepaths, _ = QtWidgets.QFileDialog.getOpenFileNames(self, "Open Verilog Source Files", "", "Verilog Files (*.v *.sv);;All Files (*)")
        if filepaths:
            # If the only design tab is the default counter, automatically close it
            if len(self.design_views) == 1:
                first_editor = self.design_views[0]
                if first_editor.toPlainText().strip() == DEFAULT_DESIGN.strip():
                    self.close_tab(0)
                    # DEFAULT_DESIGN and DEFAULT_TB are a matched pair: the stub
                    # testbench instantiates the counter this tab defined. Drop
                    # the design and keep the testbench and it now instantiates
                    # a module nothing defines -- an error against a file the
                    # user never wrote. Only the untouched default is cleared;
                    # anything the user typed or opened is theirs to keep.
                    if self.tb_view.toPlainText().strip() == DEFAULT_TB.strip():
                        self.tb_view.setPlainText("")
                        self.tb_view.filepath = None

            for filepath in filepaths:
                try:
                    code = self._read_text(filepath)
                    name = os.path.basename(filepath)
                    self.add_module_tab(name, code, filepath=filepath)
                    self.log(f"Loaded source file: {name}")
                except Exception as e:
                    self.log(f"Failed to load {filepath}: {e}")

    def load_tb_file(self):
        filepath, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Open Verilog Testbench File", "", "Verilog Files (*.v *.sv);;All Files (*)")
        if filepath:
            try:
                code = self._read_text(filepath)
                # A testbench the user opened is theirs from this moment on,
                # even if its text happens to match one eSim generated
                # earlier: forget the remembered stub so nothing here can
                # overwrite the file they just chose.
                self._tb_generated_text = None
                self.tb_view.setPlainText(code)
                self.tb_view.filepath = filepath
                self.editor_tabs.setCurrentIndex(self.editor_tabs.count() - 1)
                self.log(f"Loaded testbench: {os.path.basename(filepath)}")
            except Exception as e:
                self.log(f"Failed to load testbench: {e}")

    def update_tb_tab_name(self):
        import re
        # Strip comments/strings first so a 'module' word in a banner comment
        # can't rename the tab (the same hijack extract_ports now guards against).
        text = strip_comments(self.tb_view.toPlainText())
        match = re.search(r'\bmodule\s+(\w+)', text)
        idx = self.editor_tabs.indexOf(self.tb_view)
        if match:
            mod_name = match.group(1)
            self.editor_tabs.setTabText(idx, f"Testbench ({mod_name}.v)")
        else:
            self.editor_tabs.setTabText(idx, "Testbench (tb_design.v)")

    def save_file(self):
        editor = self.get_current_editor()
        if hasattr(editor, 'filepath') and editor.filepath:
            try:
                with open(editor.filepath, 'w', encoding='utf-8') as f:
                    f.write(editor.toPlainText())
                self.log(f"Saved {os.path.basename(editor.filepath)}")
            except Exception as e:
                self.log(f"Failed to save file: {e}")
        else:
            self.save_as_file()

    def save_as_file(self):
        filepath, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Save Verilog File", "", "Verilog Files (*.v *.sv);;All Files (*)")
        if filepath:
            try:
                editor = self.get_current_editor()
                with open(filepath, 'w', encoding='utf-8') as f:
                    f.write(editor.toPlainText())
                editor.filepath = filepath
                
                # Optionally rename the tab to match the new file name
                name = os.path.basename(filepath)
                idx = self.editor_tabs.indexOf(editor)
                if idx != -1 and idx != self.editor_tabs.count() - 1:
                    self.editor_tabs.setTabText(idx, name)
                    self.update_hierarchy_list()
                    
                self.log(f"Saved active tab to {name}.")
            except Exception as e:
                self.log(f"Failed to save file: {e}")

    def render_waveform(self, timestamps, signals_data, signal_types):
        self.current_timestamps = timestamps
        self.current_signals_data = signals_data
        self.current_signal_types = signal_types

        if not timestamps or not signals_data:
            self.clear_waveform_state()
            return

        # Build the native eSim plotWindow adapted for VCD, parentless: the host
        # (Flow Navigator -> DockArea) takes ownership when it docks the plot as
        # its own full-width tab, so the verifier keeps no reference that could
        # dangle once that tab is closed/destroyed.
        plot = make_vcd_plot_window(timestamps, signals_data, signal_types,
                                    getattr(self, 'current_project_name', 'Verilog Simulation'))

        if hasattr(self, 'act_export_csv'):
            self.act_export_csv.setEnabled(True)
            
        # The host (Flow Navigator -> DockArea) takes ownership of this
        # parentless plot when it docks it. If nothing is listening (verifier
        # used standalone), drop it rather than leaking an unparented window.
        if self.receivers(self.waveformReady) > 0:
            self.waveformReady.emit(plot)
        else:
            plot.show()
            
    def export_csv(self):
        if not hasattr(self, 'current_timestamps') or not self.current_timestamps:
            self.log("Error: No simulation data to export.")
            return
            
        filename, _ = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Export Simulation Data as CSV",
            "simulation_results.csv",
            "CSV Files (*.csv);;All Files (*)"
        )
        
        if not filename:
            return
            
        try:
            raw_signals_dict = getattr(self, 'current_raw_signals_data',
                                       self.current_signals_data)
            timescale = getattr(self, 'current_timescale', "Time")
            csv_text = to_csv(self.current_timestamps, raw_signals_dict, timescale)
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(csv_text)
            self.log(f"Successfully exported CSV to: {filename}")
        except Exception as e:
            self.log(f"Error exporting CSV: {e}")

    def get_design_code(self):
        code_blocks = []

        # Fallback if hierarchy is empty (e.g. startup glitches)
        if self.hierarchy_list.count() == 0 and hasattr(self, 'design_views'):
            for editor in self.design_views:
                code_blocks.append(editor.toPlainText())
            return "\n".join(code_blocks)

        # Iterate through hierarchy list from top to bottom
        for i in range(self.hierarchy_list.count()):
            name = self.hierarchy_list.item(i).data(QtCore.Qt.ItemDataRole.UserRole)
            # Find the corresponding tab
            for j in range(self.editor_tabs.count()):
                if self.editor_tabs.tabText(j) == name and self.editor_tabs.widget(j) in getattr(self, 'design_views', []):
                    editor = self.editor_tabs.widget(j)
                    code_blocks.append(
                        f"// --- {name} ---\n"
                        f"{_strip_block_banner(editor.toPlainText())}\n")
                    break
        return "\n".join(code_blocks)

    def get_tb_code(self):
        return self.tb_view.toPlainText()

    # Button Event Handlers
    @staticmethod
    def _safe_source_name(label):
        """A diagnostics-safe compile filename for a tab label.

        iverilog echoes the source filename in its errors, and both the
        squiggle (highlight_errors_from_log) and double-click jump match it with
        a ``[\\w./-]`` regex. A label with spaces or parens — e.g. the
        ``foo (2).v`` that duplicate disambiguation can produce — would not
        match, so the error would land nowhere. Sanitise to word chars while
        keeping the .v/.sv extension."""
        base = label if label.endswith(('.v', '.sv')) else label + '.v'
        root, ext = os.path.splitext(base)
        root = re.sub(r'[^\w.-]', '_', root) or 'design'
        return root + ext

    def _design_sources(self):
        """Ordered (filename, code) for each design tab, plus a refreshed
        ``self._compile_editors`` mapping that filename back to its editor.

        Names are sanitised (see :meth:`_safe_source_name`) so diagnostics map
        cleanly, and de-collided after sanitising (two labels can sanitise to
        the same name) so no module silently overwrites another on disk."""
        sources = []
        self._compile_editors = {}
        used = set()
        for i in range(self.editor_tabs.count()):
            widget = self.editor_tabs.widget(i)
            if widget not in getattr(self, 'design_views', []):
                continue
            name = self._safe_source_name(self.editor_tabs.tabText(i))
            if name in used:
                root, ext = os.path.splitext(name)
                n = 2
                while f"{root}_{n}{ext}" in used:
                    n += 1
                name = f"{root}_{n}{ext}"
            used.add(name)
            self._compile_editors[name] = widget
            sources.append((name, widget.toPlainText()))
        return sources

    # ------------------------------------------------------------------ #
    #  Async job plumbing (compile/sim run off the GUI thread)
    # ------------------------------------------------------------------ #
    def _job_running(self):
        return self._active_job is not None and self._active_job.isRunning()

    def _cleanup_tmpdir(self):
        """Reap the in-flight run's temp dir. Idempotent (ignore_errors + nulls
        the handle) so it's safe to call from the done/fail closures AND as a
        closeEvent backstop for a run torn down before those fire."""
        d = self._active_tmpdir
        self._active_tmpdir = None
        self._teardown_state["tmpdir"] = None
        if d:
            shutil.rmtree(d, ignore_errors=True)

    # ------------------------------------------------------------------ #
    #  Run status (phase + elapsed clock, updated once a second)
    # ------------------------------------------------------------------ #
    def _tick_progress(self):
        if self._run_started is None:
            return
        secs = int(self._run_started.elapsed() / 1000)
        self.lbl_progress.setText(
            f"{self._run_phase} {secs // 60}:{secs % 60:02d}")

    def set_phase(self, phase):
        """Name what the in-flight run is doing right now (GUI thread)."""
        self._run_phase = phase
        self._tick_progress()

    def _stream_line(self, stream, text):
        """One line of live simulator output, delivered on the GUI thread."""
        self._streamed += 1
        if self._streamed <= MAX_STREAMED_LINES:
            self.log_sim(text)
        elif self._streamed == MAX_STREAMED_LINES + 1:
            self.log(f"note: output past {MAX_STREAMED_LINES} lines is not "
                     f"shown live; the run continues.")

    def _start_job(self, work, on_done, on_fail, cancel, busy, phase="Working"):
        """Run ``work(report)`` on a worker thread; deliver its result to
        ``on_done`` (or its error to ``on_fail``) back on the GUI thread.

        ``busy`` buttons are disabled and the Cancel button + run status shown
        for the duration; ``cancel`` is the token ``work`` threads into the
        backend so Cancel can kill it. ``report`` is a thread-safe callable the
        backend uses to push phase changes and output lines to the console
        while it runs."""
        self._active_cancel = cancel
        self._teardown_state["cancel"] = cancel
        self._busy_buttons = busy
        self._streamed = 0
        for b in busy:
            b.setEnabled(False)
        self.btn_cancel.setVisible(True)
        self.btn_cancel.setEnabled(True)

        self._run_started = QtCore.QElapsedTimer()
        self._run_started.start()
        self._run_phase = phase
        self.lbl_progress.setText(f"{phase} 0:00")
        self.lbl_progress.setVisible(True)
        self.bar_progress.setVisible(True)
        self._run_timer.start()

        job = jobs.BackgroundJob(work, parent=self)
        # Queued across the thread boundary, so the backend can report freely.
        job.progress.connect(self._on_progress)
        self._active_job = job
        self._teardown_state["job"] = job

        def finish():
            self._active_job = None
            self._active_cancel = None
            self._teardown_state["job"] = None
            self._teardown_state["cancel"] = None
            for b in busy:
                b.setEnabled(True)
            self.btn_cancel.setVisible(False)
            self._run_timer.stop()
            self._run_started = None
            self.lbl_progress.setVisible(False)
            self.bar_progress.setVisible(False)

        def ok(result):
            finish()
            on_done(result)

        def err(msg):
            finish()
            on_fail(msg)

        job.succeeded.connect(ok)
        job.failed.connect(err)
        job.finished.connect(job.deleteLater)
        job.start()

    def _on_progress(self, kind, text):
        """Backend progress, marshalled onto the GUI thread by the job's
        queued ``progress`` signal. ``kind`` is 'phase' or an output stream."""
        if kind == 'phase':
            self.set_phase(text)
        else:
            self._stream_line(kind, text)

    def cancel_running_job(self):
        if self._active_cancel is not None:
            self.log("Cancelling...")
            self.btn_cancel.setEnabled(False)
            self._active_cancel.cancel()

    def closeEvent(self, event):
        # Don't let a worker thread outlive the widget.
        if self._job_running():
            if self._active_cancel is not None:
                self._active_cancel.cancel()
            self._active_job.wait(3000)
        # Backstop the temp dir: if we tore down mid-run, the done/fail closures
        # that normally reap it never fired.
        self._cleanup_tmpdir()
        super().closeEvent(event)

    def check_syntax(self):
        if self._job_running():
            return
        self.log_section("Check Syntax")
        self.log("Checking syntax...")

        # Clear previous error highlights
        for v in self.design_views:
            v.clear_error_highlights()
        self.tb_view.clear_error_highlights()

        if not self.design_views:
            self.log("Error: No design modules found.")
            return

        iverilog = self.find_iverilog()
        if not iverilog:
            self.log("Error: 'iverilog' was not found. Please install it or specify the path.")
            return

        self.log(f"iverilog: {iverilog}")

        sources = self._design_sources()
        if self._report_reserved_names(sources):
            return
        tb_code = self._refresh_generated_testbench(sources)
        has_tb = bool(tb_code and tb_code.strip())

        tmpdir = tempfile.mkdtemp()
        self._active_tmpdir = tmpdir
        self._teardown_state["tmpdir"] = tmpdir
        cancel = icarus.CancelToken()

        def compile_unit(unit_sources, out_name):
            try:
                # Fall back through the older language standards: a 2001-era
                # file using 'bit' or 'logic' as an identifier is valid code
                # that -g2012 rejects with a bare "syntax error".
                res, _std = icarus.compile_with_fallback(
                    iverilog, unit_sources, tmpdir, warnings=True,
                    out_name=out_name, timeout=COMPILE_TIMEOUT_S, cancel=cancel)
                return res
            except subprocess.TimeoutExpired:
                raise RuntimeError(
                    f"Compilation exceeded {COMPILE_TIMEOUT_S}s and was stopped.")

        def work(report):
            # Phase A -- the design on its own, which is the question this
            # button actually asks. iverilog roots at the user's own module, so
            # nothing the testbench tab happens to contain can colour the
            # verdict on their design.
            design = compile_unit(sources, "design.bin")
            if not design.ok or not has_tb or cancel.cancelled:
                return design, None
            # Phase B -- design + testbench: the unit Simulate will build. Only
            # reached once the design is known good, so a failure here belongs
            # to the testbench or to a design/testbench mismatch, and is
            # reported as such instead of as "your code has errors".
            return design, compile_unit(
                sources + [("tb_design.v", tb_code)], "tb.bin")

        def done(res):
            try:
                design, tb = res
                if cancel.cancelled:
                    self.log("Syntax check cancelled.")
                elif not design.ok:
                    self.log("Design errors:")
                    self._report_compile_errors(design)
                elif tb is None:
                    self.log("Syntax OK.")
                    if not has_tb:
                        self.log("note: testbench is empty, so it was not "
                                 "checked. Use Auto-Generate Testbench before "
                                 "simulating.")
                elif tb.ok:
                    self.log("Syntax OK (design and testbench).")
                else:
                    self.log("Design syntax OK — testbench errors:")
                    stale = self._stale_testbench_note(tb, sources)
                    self._report_compile_errors(tb, hints=not stale)
            finally:
                self._cleanup_tmpdir()

        def fail(msg):
            self.log(f"Syntax check error: {msg}")
            self._cleanup_tmpdir()

        self._start_job(work, done, fail, cancel,
                        busy=[self.btn_syntax, self.btn_simulate],
                        phase="Checking")

    def _report_reserved_names(self, design_sources):
        """Name a module that collides with the Verilog language itself, and
        report whether it did (True => do not compile).

        ``module nand`` is a redeclaration of a built-in gate primitive, and
        iverilog says so as "syntax error" on the module line -- which reads
        like the design is malformed and sends people hunting through their
        own code. Rename the module and the same code compiles, which is
        exactly the kind of thing a tool should be able to explain."""
        blocked = False
        for _fname, code in design_sources:
            for module in reserved_modules(code):
                self.log("Error: " + reserved_name_reason(module))
                blocked = True
        return blocked

    def _refresh_generated_testbench(self, design_sources):
        """Re-generate the testbench for the current design if -- and only if
        -- the one on screen is eSim's own and no longer fits.

        Renaming a module is a normal thing to do, and it silently invalidates
        a generated testbench: the instance still names the old module, so
        Check Syntax fails against a file the user never wrote. Regenerating
        eSim's own stub costs nothing and keeps the check about the design.
        A testbench the user wrote or edited is never regenerated here -- it
        is reported by _stale_testbench_note instead, which is the honest
        answer when the two really have diverged.

        Returns the testbench source to compile."""
        tb_code = self.get_tb_code()
        if not tb_code.strip():
            return tb_code
        defined = set()
        for _fname, code in design_sources:
            defined.update(find_modules(code))
        if testbench_matches(tb_code, defined):
            return tb_code
        if not self._testbench_is_ours(tb_code):
            return tb_code
        design_code = "\n".join(code for _n, code in design_sources)
        module_name, ports = extract_ports(design_code)
        if not module_name:
            return tb_code
        fresh = generate_stub_testbench(module_name, ports,
                                        design_code=design_code)
        self._set_testbench(fresh)
        self.log(f"note: the generated testbench was still written for a "
                 f"different module — regenerated it for '{module_name}'.")
        return fresh

    def _report_compile_errors(self, res, hints=True):
        """Log a failed compile: raw tool output, squiggles in the owning tabs,
        then a one-line locator.

        The locator exists because the line number is the single most useful
        token in a compiler run and it sits mid-line inside a message people
        skim past. Restating it as ``tb_design.v:8`` -- and saying the console
        is double-clickable -- turns 'it says something failed' into 'go here'.
        """
        self.log_sim(res.stderr)
        self.highlight_errors_from_log(res.stderr, hints=hints)
        locations = icarus.error_locations(res.output)
        if locations:
            where = ", ".join(locations[:5])
            if len(locations) > 5:
                where += f" (+{len(locations) - 5} more)"
            self.log(f"note: error line(s) at {where} — double-click an error "
                     f"in this console to jump there.")

    def _stale_testbench_note(self, res, design_sources):
        """Name the stale-testbench case outright, and report whether it did.

        The pattern: the testbench instantiates a module no design tab defines,
        because it was generated for -- or shipped with -- a different design.
        iverilog reports that against ``tb_design.v``, a file the user never
        wrote, in wording that reads like their own code is broken. That is how
        a mismatched testbench gets reported as "Check Syntax doesn't work".

        Returns True when a note was logged, so the caller can suppress the
        generic 'module instantiated but not defined' hint, which says the same
        thing less precisely."""
        defined = set()
        for _, code in design_sources:
            defined.update(find_modules(code))
        stale = []
        for fname, _line, module in icarus.unknown_module_sites(res.output):
            if (os.path.basename(fname) == "tb_design.v"
                    and module not in defined and module not in stale):
                stale.append(module)
        if not stale:
            return False
        names = ", ".join(f"'{m}'" for m in stale)
        self.log(f"note: the testbench instantiates {names}, which no design "
                 f"tab defines — it does not match this design. Use "
                 f"Auto-Generate Testbench to build one for the current "
                 f"module, or open your own with Open Testbench.")
        return True

    def auto_generate_tb(self):
        """Generate a testbench stub for the last-active design tab.
        
        If the user was viewing the testbench tab when they clicked the button,
        we remember which design tab was previously active (stored in
        _last_active_design_idx) so we target the correct module.
        """
        self.btn_stub.setEnabled(False)

        active_editor = self.get_current_editor()
        if active_editor == self.tb_view:
            # Use the remembered last-active design tab if available
            if hasattr(self, '_last_active_design_idx') and self._last_active_design_idx is not None:
                idx = self._last_active_design_idx
                widget = self.editor_tabs.widget(idx)
                if widget in getattr(self, 'design_views', []):
                    active_editor = widget
                else:
                    active_editor = self.design_views[0] if self.design_views else None
            elif hasattr(self, 'design_views') and self.design_views:
                active_editor = self.design_views[0]
            else:
                self.log("Error: No design modules found.")
                self.btn_stub.setEnabled(True)
                return

        if active_editor is None:
            self.log("Error: No design modules found.")
            self.btn_stub.setEnabled(True)
            return

        code = active_editor.toPlainText()
        if not code or not code.strip():
            self.log("Error: Current module editor is empty. Write your design first.")
            self.btn_stub.setEnabled(True)
            return

        module_name, ports = extract_ports(code)
        if not module_name:
            self.log("Error: Could not find any Verilog module declaration.")
            self.btn_stub.setEnabled(True)
            return

        tb_code = generate_stub_testbench(module_name, ports, design_code=code)
        self._set_testbench(tb_code)
        # Jump to the testbench tab (always last)
        self.editor_tabs.setCurrentIndex(self.editor_tabs.count() - 1)
        self.log(f"Generated testbench stub for module '{module_name}'.")
        self.btn_stub.setEnabled(True)

    def _ensure_testbench(self, design_sources):
        """Make sure a usable testbench exists before simulating.

        The realistic user path here is: paste a module from somewhere, press
        Simulate. Demanding that they *also* produce a matching testbench --
        with the right instance name, the right port map and the exact
        ``$dumpfile`` eSim wants -- is the step at which that user gives up. So
        an empty testbench, or one left over from a different design, is
        replaced with a generated one instead of failing the run.

        A testbench that does instantiate this design is always left alone: it
        is the user's, and they may well have written it deliberately.

        So is one that does NOT match but is not eSim's to replace. Only two
        things may be overwritten here: an empty testbench, and a testbench
        still carrying eSim's own provenance marker (see
        ports.is_generated_testbench) -- i.e. the stub eSim wrote, which the
        user has not touched. Anything else is somebody's work, and the answer
        to "it no longer matches the design" is to SAY so, not to delete it:
        rename a module and the previous behaviour silently replaced a
        hand-written testbench with boilerplate.

        Returns the testbench source to compile, or None if one cannot be made.
        """
        tb_code = self.get_tb_code()
        defined = set()
        for _name, code in design_sources:
            defined.update(find_modules(code))

        if testbench_matches(tb_code, defined):
            return tb_code

        design_code = "\n".join(code for _n, code in design_sources)
        module_name, ports = extract_ports(design_code)
        if not module_name:
            self.log("Error: no Verilog module found in the design tabs.")
            return None

        empty = not tb_code.strip()
        if not empty and not self._testbench_is_ours(tb_code):
            self.log(f"Error: your testbench does not instantiate "
                     f"'{module_name}', so this design would not be driven by "
                     f"it.")
            self.log("note: eSim did not touch it — it is yours. Update the "
                     "instance in the Testbench tab, or press Auto-Generate "
                     "Testbench to replace it with a generated one.")
            return None

        why = ("the testbench is empty" if empty
               else f"the generated testbench does not instantiate "
                    f"'{module_name}'")
        tb_code = generate_stub_testbench(module_name, ports,
                                          design_code=design_code)
        self._set_testbench(tb_code)
        self.log(f"note: {why} — generated one for '{module_name}' and used "
                 f"it. Open the Testbench tab to edit or replace it.")
        return tb_code

    def _testbench_is_ours(self, tb_code=None):
        """True when the testbench on screen is one eSim wrote and the user has
        not adopted.

        Two independent signals, either of which is enough. The marker comment
        survives restarts and is visible to the user (deleting that line is a
        perfectly good way to say "this is mine now"); the remembered text
        covers a testbench generated in this session whose marker the user has
        not seen yet."""
        code = self.get_tb_code() if tb_code is None else tb_code
        if not code.strip():
            return True
        if code.strip() == DEFAULT_TB.strip():
            return True            # the untouched startup template
        remembered = getattr(self, "_tb_generated_text", None)
        if remembered is not None and code.strip() == remembered.strip():
            return True
        return is_generated_testbench(code)

    def _set_testbench(self, tb_code):
        """Put a generated testbench on screen and remember it verbatim, so a
        later edit by the user is detectable as an edit."""
        self._tb_generated_text = tb_code
        self.tb_view.setPlainText(tb_code)

    def simulate_and_wave(self):
        if self._job_running():
            return
        self.log_section("Simulate")

        design_code = self.get_design_code()
        if not design_code or not design_code.strip():
            self.log("Error: Design editor is empty.")
            return

        iverilog = self.find_iverilog()
        vvp = self.find_vvp()
        if not iverilog or not vvp:
            self.log("Error: 'iverilog' or 'vvp' binaries were not found on the system.")
            return

        # Each design tab is written under its own tab-label filename (via
        # _design_sources) so iverilog error output references the tab the user
        # sees and highlight_errors_from_log can find it.
        design_sources = self._design_sources()

        # A module named after a Verilog primitive cannot compile, whichever
        # button started the run. Said here in eSim's words rather than left to
        # iverilog's "syntax error" on the module line.
        if self._report_reserved_names(design_sources):
            return

        # A design that is already a complete testbench (no ports, drives
        # itself) is simulated as-is; wrapping it would instantiate a port-less
        # module and drive nothing.
        self_contained = (len(design_sources) == 1
                          and is_self_contained_testbench(design_sources[0][1]))
        if self_contained:
            tb_code = design_sources[0][1]
            sources = list(design_sources)
            self.log("Design is a self-contained testbench — running it "
                     "directly.")
        else:
            tb_code = self._ensure_testbench(design_sources)
            if tb_code is None:
                return
            sources = design_sources + [("tb_design.v", tb_code)]

        # Read the waveform back under whatever name the testbench dumps to,
        # and supply a dump (and a stop) when it has neither. Between them,
        # these are the difference between "Simulate produced nothing" and a
        # waveform, for every testbench not written against eSim's defaults.
        vcd_name = dump_file_name(tb_code) or "sim_out.vcd"
        if not has_dump(tb_code):
            guard = None if has_finish(tb_code) else NO_FINISH_GUARD_NS
            sources = sources + [("esim_autodump.v",
                                  autodump_source(vcd_name, guard_ns=guard))]
            note = "no $dumpvars in the testbench — capturing all signals"
            if guard:
                note += (f"; no $finish either — stopping after "
                         f"{guard} ns")
            self.log(f"note: {note}.")

        count = len(design_sources)
        self.log(f"Compiling {count} file(s)"
                 + ("..." if self_contained else " + testbench..."))

        tmpdir = tempfile.mkdtemp()
        self._active_tmpdir = tmpdir
        self._teardown_state["tmpdir"] = tmpdir
        cancel = icarus.CancelToken()
        libdir = CosimConfig.iverilog_libdir()

        def work(report):
            # Cap the live relay HERE, on the worker side: a $display in a
            # clocked loop can emit hundreds of thousands of lines, and one
            # queued signal each would flood the GUI event loop even if the
            # slot then threw them away. The full output is still captured in
            # the result for the summary.
            streamed = [0]

            def relay(kind, text):
                streamed[0] += 1
                if streamed[0] <= MAX_STREAMED_LINES:
                    report(kind, text)

            try:
                report('phase', 'Compiling')
                run = icarus.build_and_simulate(
                    iverilog, vvp, sources, tmpdir, libdir=libdir,
                    vcd_name=vcd_name,
                    compile_timeout=COMPILE_TIMEOUT_S,
                    sim_timeout=SIM_TIMEOUT_S, cancel=cancel,
                    on_line=relay,
                    on_phase=lambda p: report(
                        'phase',
                        'Simulating' if p == 'simulate' else 'Compiling'))
            except subprocess.TimeoutExpired:
                raise RuntimeError(
                    f"Simulation exceeded {SIM_TIMEOUT_S}s and was stopped. "
                    "Add a $finish to your testbench or shorten the run.")
            # Parse the VCD HERE, on the worker thread. It used to run in the
            # done-callback, i.e. on the GUI thread, where a long run's dump
            # locked the whole application up for as long as it took.
            parsed = None
            if run.vcd_content and not cancel.cancelled:
                report('phase', 'Reading waveform')
                parsed = parse_vcd_for_plot(run.vcd_content)
            return run, parsed

        def done(result):
            try:
                run, parsed = result
                self._render_sim_result(run, parsed,
                                        cancelled=cancel.cancelled,
                                        design_sources=design_sources)
            finally:
                self._cleanup_tmpdir()

        def fail(msg):
            self.log(f"Simulation error: {msg}")
            self._cleanup_tmpdir()

        self._start_job(work, done, fail, cancel,
                        busy=[self.btn_simulate, self.btn_syntax],
                        phase="Compiling")

    def _render_sim_result(self, run, parsed=None, cancelled=False,
                           design_sources=()):
        """Log/highlight/plot the outcome of build_and_simulate, on the GUI
        thread. ``parsed`` is the already-parsed VCD (done on the worker thread
        so this method stays cheap). ``design_sources`` is the design half of
        what was compiled, used only to recognise a mismatched testbench."""
        if cancelled:
            self.log("Simulation cancelled.")
            return

        if not run.compile.ok:
            # Simulate always builds design+testbench as one unit, so the same
            # mismatch Check Syntax now isolates can land here too.
            self.log("Compilation failed:")
            stale = self._stale_testbench_note(run.compile, design_sources)
            self._report_compile_errors(run.compile, hints=not stale)
            return

        if run.std != icarus.STD_FALLBACKS[0]:
            self.log(f"note: compiled with {run.std} — this code uses "
                     f"identifiers that are reserved words in newer Verilog "
                     f"standards.")

        sim = run.sim
        # stdout/stderr already reached the console live (see _stream_line);
        # only the tail beyond the streaming cap needs summarising.
        produced = len(((sim.stdout or "") + (sim.stderr or "")).splitlines())
        if produced > MAX_STREAMED_LINES:
            self.log(f"note: {produced - MAX_STREAMED_LINES} further output "
                     f"line(s) not shown.")

        if not sim.ok:
            self.log(f"vvp: exited with code {sim.returncode}")
            if sim.returncode in [3221225781, -1073741515]:
                self.log("note: exit code 0xC0000135 — missing DLL. "
                         "Reinstall Icarus Verilog with 'Install MinGW Dependencies' checked.")

        if parsed is None and run.vcd_content:
            # Fallback for callers that hand over an unparsed run. The normal
            # path parses on the worker thread (see simulate_and_wave) exactly
            # so this does not happen on a big dump.
            try:
                parsed = parse_vcd_for_plot(run.vcd_content)
            except Exception as ex:
                self.log(f"Error: failed to parse VCD — {ex}")

        # --- HARDCODED PERMANENT VCD DISK WRITE & DOCKING ---
        if run.vcd_content:
            import os
            import sys
            
            user_home = os.path.expanduser("~")
            proj_name = getattr(self, 'current_project_name', 'course_task')
            proj_dir = os.path.join(user_home, "eSim-Workspace", str(proj_name))
            os.makedirs(proj_dir, exist_ok=True)
            
            vcd_name = run.vcd_name if run.vcd_name else "sim_out.vcd"
            vcd_path = os.path.join(proj_dir, vcd_name)
            
            with open(vcd_path, "w", encoding="utf-8") as f:
                f.write(run.vcd_content)
                
            src_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
            if src_dir not in sys.path:
                sys.path.insert(0, src_dir)
            
            # Instantiate custom viewer linked directly to the permanent path
            self.wave_win = waveWindow(proj_dir, proj_name)
            
            # Emit to the Flow Navigator dock tab cleanly
            if self.receivers(self.waveformReady) > 0:
                self.waveformReady.emit(self.wave_win)
            else:
                self.wave_win.show()

        if parsed and parsed[0]:
            timestamps, signals_data, signal_types, raw_signals, timescale = parsed
            self.current_raw_signals_data = raw_signals
            self.current_timescale = timescale
            if run.vcd_name and run.vcd_name != "sim_out.vcd":
                self.log(f"Read waveform from {run.vcd_name}.")
            self.log(f"Waveform ready — {len(signals_data)} signal(s), "
                     f"{len(timestamps)} sample(s).")
            self._warn_if_undriven(signals_data, raw_signals)
            if hasattr(self, 'act_export_csv'):
                self.act_export_csv.setEnabled(True)
        elif run.vcd_content:
            self.log("Error: the simulation dumped a waveform with no value "
                     "changes. Check that the testbench drives the design "
                     "before $finish.")
        else:
            self.log("No waveform was produced. The testbench ran but wrote "
                     "no VCD — use Auto-Generate Testbench for one that does.")

        # Compile + simulate both succeeded: let the host advance the flow.
        if run.ok:
            self.simulationSucceeded.emit()

    def _warn_if_undriven(self, signals_data, raw_signals):
        """Say so when the waveform is mostly undefined.

        An all-X trace is the single most common "the waveform is broken"
        report, and it is almost never a plotting problem: it means the
        testbench never drove the inputs. Naming that beats leaving the user
        staring at a flat red plot wondering which half of the tool failed."""
        undriven = [name for name, vals in raw_signals.items()
                    if vals and all(str(v).lower() in ('x', 'z')
                                    for v in vals)]
        if not undriven:
            return
        names = ", ".join(undriven[:5])
        if len(undriven) > 5:
            names += f" (+{len(undriven) - 5} more)"
        scope = ("every signal" if len(undriven) == len(signals_data)
                 else "some signals")
        self.log(f"note: {scope} stayed undefined (X/Z) for the whole run: "
                 f"{names}. That usually means the testbench never drives "
                 f"them — Auto-Generate Testbench writes one that does.")

    def set_design_bus(self, bus):
        """Bind this stage to the shared design (called by the Flow Navigator)."""
        self.bus = bus

    def render_from_bus(self):
        """Show the shared design in the single design tab. Called when this
        stage is entered, so a design authored/loaded elsewhere shows up here
        with no manual 'load' step. Idempotent: if the tab already holds the bus
        content, leave it -- in-progress edits survive a re-entry."""
        if self.bus is None:
            return
        code = self.bus.get_content()
        if not code:
            return
        if self._rendered_content == code:
            return
        # Drop every existing design tab (the testbench is pinned and stays)
        # and install the shared design as the single design tab.
        for editor in list(self.design_views):
            idx = self.editor_tabs.indexOf(editor)
            if idx != -1:
                self.close_tab(idx)
        # Same pairing as load_source_files: the default testbench only makes
        # sense next to the default design, and the design it drives is being
        # replaced right here. Clear it so the incoming design isn't checked
        # against a testbench for someone else's module.
        if self.tb_view.toPlainText().strip() == DEFAULT_TB.strip():
            self.tb_view.setPlainText("")
            self.tb_view.filepath = None
        name = os.path.basename(self.bus.path) if self.bus.path else "design.v"
        self.add_module_tab(name, code, filepath=self.bus.path or None)
        self._rendered_content = code
        self.log(f"Loaded design: {name}")

    def save_testbench_to_library(self):
        """Keep the testbench in the same folder as the design it drives.

        Deliberately NOT part of collect_into_bus, which must stay purely
        in-memory. Ownership inside a design folder is split and never
        overlaps: the DesignBus writes ``<module>.v`` (which already contains
        every design tab, concatenated by get_design_code), and this writes
        ``tb_<module>.v``. The untouched default template is skipped, so a
        design the user never wrote a testbench for does not get somebody
        else's boilerplate filed next to it."""
        if self.bus is None:
            return ""
        module = verilog_library.top_module(self.bus.get_content())
        if not module:
            return ""
        tb = self.tb_view.toPlainText()
        if not tb.strip() or tb.strip() == DEFAULT_TB.strip():
            return ""
        return verilog_library.write_text(
            verilog_library.sibling_path(module, "tb_%s.v" % module), tb)

    def collect_into_bus(self):
        """Push the design under test into the shared design (in memory; NO
        disk). Called when this stage is left. Skips the untouched default
        template and empty / parse-less states, so merely passing through Verify
        never overwrites the design with boilerplate. A design first authored
        here is given a home under ~/eSim-Workspace/verified_verilog so Convert
        can later materialize it."""
        if self.bus is None:
            return
        self.auto_detect_hierarchy()
        code = self.get_design_code()
        if not code or not code.strip():
            return
        if code.strip() == DEFAULT_DESIGN.strip():
            return
        module_name, _ = extract_ports(code)
        if not module_name:
            return
        # Just the in-memory push. The bus gives the design its home in the
        # Verilog library off its own autosave -- named from the module, and
        # re-derived every time the module changes, which the old
        # "set a path once, only if there isn't one" rule never did. That is
        # what left a design permanently filed under the first name it ever
        # had, however many times it was replaced afterwards.
        self.bus.set_content(code)
        self._rendered_content = code
