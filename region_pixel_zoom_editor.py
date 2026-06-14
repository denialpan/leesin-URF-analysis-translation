from __future__ import annotations

import json
import math
import subprocess
import sys
import tempfile
import tkinter as tk
from fractions import Fraction
from pathlib import Path
from tkinter import filedialog, messagebox, simpledialog, ttk


TARGET_WIDTH = 1920
TARGET_HEIGHT = 1080
DEFAULT_JSON = "hud-regions.json"
ZOOM_LEVELS = (1, 2, 3, 4, 6, 8, 10, 12)
HANDLE_SIZE = 6


VIDEO_EXTENSIONS = {
    ".mp4",
    ".mov",
    ".mkv",
    ".webm",
    ".avi",
    ".m4v",
    ".flv",
    ".wmv",
}


def clamp(value: int, low: int, high: int) -> int:
    return max(low, min(high, value))


def parse_timestamp(value: str, fps: float = 60.0) -> float:
    value = value.strip()
    if not value:
        return 0.0

    try:
        seconds = float(value)
    except ValueError:
        parts = value.split(":")
        if len(parts) not in (2, 3, 4):
            raise ValueError(
                "Use seconds, MM:SS, HH:MM:SS, or Resolve timecode HH:MM:SS:FF."
            ) from None

        try:
            if len(parts) == 4:
                hours, minutes, whole_seconds, frames = (int(part) for part in parts)
                if frames < 0 or frames >= math.ceil(fps):
                    raise ValueError(
                        f"The frame field must be between 0 and {math.ceil(fps) - 1} "
                        f"for {fps:g} fps."
                    )
                seconds = hours * 3600 + minutes * 60 + whole_seconds + frames / fps
            else:
                numeric_parts = [float(part) for part in parts]
                if len(numeric_parts) == 3:
                    hours, minutes, final_seconds = numeric_parts
                    seconds = hours * 3600 + minutes * 60 + final_seconds
                else:
                    minutes, final_seconds = numeric_parts
                    seconds = minutes * 60 + final_seconds
        except ValueError as error:
            if str(error).startswith("The frame field"):
                raise
            raise ValueError(
                "Use seconds, MM:SS, HH:MM:SS, or Resolve timecode HH:MM:SS:FF."
            ) from None

    if seconds < 0:
        raise ValueError("The timestamp cannot be negative.")
    return seconds


def probe_frame_rate(path: Path) -> float:
    command = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=avg_frame_rate",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        str(path),
    ]
    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            creationflags=0x08000000 if sys.platform == "win32" else 0,
            check=False,
        )
        if result.returncode == 0 and result.stdout.strip():
            fps = float(Fraction(result.stdout.strip()))
            if fps > 0:
                return fps
    except (FileNotFoundError, ValueError, ZeroDivisionError):
        pass
    return 60.0


class RegionEditor(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("Pixel Zoom Region Editor")
        self.geometry("1400x900")

        self.zoom = tk.IntVar(value=2)
        self.timestamp = tk.StringVar(value="0")
        self.status = tk.StringVar(value="Open a video/image to begin.")
        self.region_name = tk.StringVar()
        self.x_var = tk.IntVar(value=0)
        self.y_var = tk.IntVar(value=0)
        self.w_var = tk.IntVar(value=0)
        self.h_var = tk.IntVar(value=0)

        self.source_path: Path | None = None
        self.source_fps = 60.0
        self.temp_dir = tempfile.TemporaryDirectory(prefix="region_editor_")
        self.preview_generation = 0
        self.preview_path: Path | None = None
        self.base_image: tk.PhotoImage | None = None
        self.zoomed_image: tk.PhotoImage | None = None
        self.canvas_image_id: int | None = None

        self.regions: dict[str, dict[str, int]] = {}
        self.rect_ids: dict[str, int] = {}
        self.handle_ids: list[int] = []
        self.selected_name: str | None = None

        self.drag_mode: str | None = None
        self.drag_start: tuple[int, int] | None = None
        self.drag_original: dict[str, int] | None = None
        self.pending_rect_id: int | None = None

        self._build_ui()
        self.protocol("WM_DELETE_WINDOW", self._on_close)

    def _build_ui(self) -> None:
        main = ttk.Frame(self)
        main.pack(fill=tk.BOTH, expand=True)

        left = ttk.Frame(main)
        left.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        right = ttk.Frame(main, width=320)
        right.pack(side=tk.RIGHT, fill=tk.Y)
        right.pack_propagate(False)

        toolbar = ttk.Frame(left)
        toolbar.pack(side=tk.TOP, fill=tk.X)

        ttk.Button(toolbar, text="Open Video/Image", command=self.open_source).pack(
            side=tk.LEFT, padx=4, pady=4
        )
        ttk.Label(toolbar, text="Time / timecode").pack(side=tk.LEFT, padx=(12, 4))
        ttk.Entry(toolbar, textvariable=self.timestamp, width=15).pack(side=tk.LEFT)
        ttk.Button(toolbar, text="Load Frame", command=self.load_frame).pack(
            side=tk.LEFT, padx=4
        )

        ttk.Label(toolbar, text="Zoom").pack(side=tk.LEFT, padx=(12, 4))
        zoom_box = ttk.Combobox(
            toolbar,
            textvariable=self.zoom,
            width=4,
            values=ZOOM_LEVELS,
            state="readonly",
        )
        zoom_box.pack(side=tk.LEFT)
        zoom_box.bind("<<ComboboxSelected>>", lambda _event: self.refresh_zoom())

        ttk.Button(toolbar, text="Load JSON", command=self.load_json).pack(
            side=tk.LEFT, padx=(12, 4)
        )
        ttk.Button(toolbar, text="Save JSON", command=self.save_json).pack(
            side=tk.LEFT, padx=4
        )

        canvas_frame = ttk.Frame(left)
        canvas_frame.pack(fill=tk.BOTH, expand=True)

        self.canvas = tk.Canvas(
            canvas_frame,
            bg="#1f1f1f",
            highlightthickness=0,
            cursor="crosshair",
        )
        x_scroll = ttk.Scrollbar(canvas_frame, orient=tk.HORIZONTAL, command=self.canvas.xview)
        y_scroll = ttk.Scrollbar(canvas_frame, orient=tk.VERTICAL, command=self.canvas.yview)
        self.canvas.configure(xscrollcommand=x_scroll.set, yscrollcommand=y_scroll.set)

        self.canvas.grid(row=0, column=0, sticky="nsew")
        y_scroll.grid(row=0, column=1, sticky="ns")
        x_scroll.grid(row=1, column=0, sticky="ew")
        canvas_frame.rowconfigure(0, weight=1)
        canvas_frame.columnconfigure(0, weight=1)

        self.canvas.bind("<ButtonPress-1>", self.on_mouse_down)
        self.canvas.bind("<B1-Motion>", self.on_mouse_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_mouse_up)
        self.canvas.bind("<Motion>", self.on_mouse_move)

        ttk.Label(right, text="Regions", font=("Segoe UI", 11, "bold")).pack(
            anchor=tk.W, padx=8, pady=(8, 4)
        )

        self.region_list = tk.Listbox(right, height=14, exportselection=False)
        self.region_list.pack(fill=tk.X, padx=8)
        self.region_list.bind("<<ListboxSelect>>", self.on_region_select)

        buttons = ttk.Frame(right)
        buttons.pack(fill=tk.X, padx=8, pady=6)
        ttk.Button(buttons, text="Rename", command=self.rename_selected).pack(
            side=tk.LEFT, expand=True, fill=tk.X, padx=(0, 3)
        )
        ttk.Button(buttons, text="Delete", command=self.delete_selected).pack(
            side=tk.LEFT, expand=True, fill=tk.X, padx=(3, 0)
        )

        form = ttk.LabelFrame(right, text="Selected Region")
        form.pack(fill=tk.X, padx=8, pady=8)

        self._add_form_row(form, "Name", ttk.Entry(form, textvariable=self.region_name))
        self._add_form_row(form, "X", ttk.Entry(form, textvariable=self.x_var))
        self._add_form_row(form, "Y", ttk.Entry(form, textvariable=self.y_var))
        self._add_form_row(form, "Width", ttk.Entry(form, textvariable=self.w_var))
        self._add_form_row(form, "Height", ttk.Entry(form, textvariable=self.h_var))

        ttk.Button(form, text="Apply Numeric Edits", command=self.apply_numeric_edits).pack(
            fill=tk.X, padx=6, pady=6
        )

        help_box = ttk.LabelFrame(right, text="Controls")
        help_box.pack(fill=tk.X, padx=8, pady=8)
        ttk.Label(
            help_box,
            text=(
                "Draw: drag empty space\n"
                "Select: click rectangle\n"
                "Move: drag selected rectangle\n"
                "Resize: drag corner handles\n"
                "Zoom is nearest-neighbor only"
            ),
            justify=tk.LEFT,
        ).pack(anchor=tk.W, padx=6, pady=6)

        ttk.Label(self, textvariable=self.status, anchor=tk.W).pack(
            side=tk.BOTTOM, fill=tk.X, padx=6, pady=3
        )

    def _add_form_row(self, parent: ttk.Frame, label: str, widget: ttk.Widget) -> None:
        row = ttk.Frame(parent)
        row.pack(fill=tk.X, padx=6, pady=3)
        ttk.Label(row, text=label, width=8).pack(side=tk.LEFT)
        widget.pack(side=tk.LEFT, fill=tk.X, expand=True)

    def open_source(self) -> None:
        path = filedialog.askopenfilename(
            title="Open video or image",
            filetypes=[
                ("Video/Image", "*.mp4 *.mov *.mkv *.webm *.avi *.m4v *.png *.jpg *.jpeg *.bmp"),
                ("All files", "*.*"),
            ],
        )
        if not path:
            return

        self.source_path = Path(path)
        if self.source_path.suffix.lower() in VIDEO_EXTENSIONS:
            self.source_fps = probe_frame_rate(self.source_path)
        else:
            self.source_fps = 60.0
        self.status.set(
            f"Loaded source: {self.source_path.name} ({self.source_fps:g} fps)"
        )
        self.load_frame()

    def load_frame(self) -> None:
        if self.source_path is None:
            messagebox.showinfo("No source", "Open a video or image first.")
            return

        try:
            timestamp = parse_timestamp(self.timestamp.get(), self.source_fps)
        except ValueError as error:
            messagebox.showerror("Invalid timestamp", str(error))
            return

        preview_path = self._extract_preview(self.source_path, timestamp)
        if preview_path is None:
            return

        try:
            new_base_image = tk.PhotoImage(file=str(preview_path), format="PPM")
        except tk.TclError as error:
            messagebox.showerror("Preview error", f"Could not load extracted preview:\n{error}")
            return

        self.preview_path = preview_path
        self.base_image = new_base_image
        self.zoomed_image = None
        if self.canvas_image_id is not None:
            self.canvas.delete(self.canvas_image_id)
            self.canvas_image_id = None

        self.refresh_zoom()
        self.update_idletasks()
        self.status.set(
            f"Frame loaded at {timestamp:.3f} seconds; "
            f"target resolution {TARGET_WIDTH}x{TARGET_HEIGHT}."
        )

    def _extract_preview(self, path: Path, timestamp: float) -> Path | None:
        self.preview_generation += 1
        preview_path = (
            Path(self.temp_dir.name)
            / f"preview_{self.preview_generation:06d}.ppm"
        )
        command = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
        ]

        if path.suffix.lower() in VIDEO_EXTENSIONS:
            command.extend(["-ss", f"{timestamp:.6f}"])

        command.extend(
            [
                "-i",
                str(path),
                "-frames:v",
                "1",
                "-vf",
                f"scale={TARGET_WIDTH}:{TARGET_HEIGHT}:flags=neighbor",
                "-pix_fmt",
                "rgb24",
                "-c:v",
                "ppm",
                "-f",
                "image2pipe",
                "-",
            ]
        )

        try:
            result = subprocess.run(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                creationflags=0x08000000 if sys.platform == "win32" else 0,
                check=False,
            )
        except FileNotFoundError:
            messagebox.showerror("ffmpeg missing", "ffmpeg was not found on PATH.")
            return None

        if result.returncode != 0:
            messagebox.showerror(
                "ffmpeg failed",
                result.stderr.decode(errors="replace").strip()
                or "Could not extract preview frame.",
            )
            return None

        if not result.stdout:
            details = result.stderr.decode(errors="replace").strip()
            message = (
                f"ffmpeg returned no image at {timestamp:.3f} seconds. "
                "The timestamp may be beyond the end of the source video."
            )
            if details:
                message += f"\n\nffmpeg output:\n{details}"
            messagebox.showerror("Preview error", message)
            return None

        preview_path.write_bytes(result.stdout)
        return preview_path

    def refresh_zoom(self) -> None:
        if self.base_image is None:
            return

        zoom = int(self.zoom.get())
        self.zoomed_image = self.base_image.zoom(zoom, zoom)

        if self.canvas_image_id is None:
            self.canvas_image_id = self.canvas.create_image(
                0,
                0,
                image=self.zoomed_image,
                anchor=tk.NW,
                tags=("image",),
            )
        else:
            self.canvas.itemconfigure(self.canvas_image_id, image=self.zoomed_image)

        self.canvas.tag_lower("image")
        self.canvas.configure(
            scrollregion=(0, 0, TARGET_WIDTH * zoom, TARGET_HEIGHT * zoom)
        )
        self.redraw_regions()

    def canvas_to_frame(self, canvas_x: float, canvas_y: float) -> tuple[int, int]:
        zoom = int(self.zoom.get())
        x = int(self.canvas.canvasx(canvas_x) // zoom)
        y = int(self.canvas.canvasy(canvas_y) // zoom)
        return clamp(x, 0, TARGET_WIDTH - 1), clamp(y, 0, TARGET_HEIGHT - 1)

    def frame_to_canvas_rect(self, region: dict[str, int]) -> tuple[int, int, int, int]:
        zoom = int(self.zoom.get())
        x1 = region["x"] * zoom
        y1 = region["y"] * zoom
        x2 = (region["x"] + region["width"]) * zoom
        y2 = (region["y"] + region["height"]) * zoom
        return x1, y1, x2, y2

    def redraw_regions(self) -> None:
        self.canvas.delete("region")
        self.canvas.delete("handle")
        self.rect_ids.clear()
        self.handle_ids.clear()

        for name, region in self.regions.items():
            color = "#00ff66" if name == self.selected_name else "#ffaa00"
            width = 3 if name == self.selected_name else 2
            x1, y1, x2, y2 = self.frame_to_canvas_rect(region)
            rect_id = self.canvas.create_rectangle(
                x1,
                y1,
                x2,
                y2,
                outline=color,
                width=width,
                tags=("region", f"region:{name}"),
            )
            self.rect_ids[name] = rect_id
            self.canvas.create_text(
                x1 + 4,
                y1 + 4,
                text=name,
                fill=color,
                anchor=tk.NW,
                tags=("region", f"region:{name}"),
            )

        self.draw_handles()

    def draw_handles(self) -> None:
        self.canvas.delete("handle")
        self.handle_ids.clear()
        if self.selected_name is None or self.selected_name not in self.regions:
            return

        x1, y1, x2, y2 = self.frame_to_canvas_rect(self.regions[self.selected_name])
        for handle, x, y in (
            ("nw", x1, y1),
            ("ne", x2, y1),
            ("sw", x1, y2),
            ("se", x2, y2),
        ):
            handle_id = self.canvas.create_rectangle(
                x - HANDLE_SIZE,
                y - HANDLE_SIZE,
                x + HANDLE_SIZE,
                y + HANDLE_SIZE,
                fill="#00ff66",
                outline="#003300",
                tags=("handle", f"handle:{handle}"),
            )
            self.handle_ids.append(handle_id)

    def hit_test_region(self, x: int, y: int) -> str | None:
        for name, region in reversed(list(self.regions.items())):
            if (
                region["x"] <= x <= region["x"] + region["width"]
                and region["y"] <= y <= region["y"] + region["height"]
            ):
                return name
        return None

    def hit_test_handle(self, canvas_x: float, canvas_y: float) -> str | None:
        item_ids = self.canvas.find_overlapping(
            self.canvas.canvasx(canvas_x) - HANDLE_SIZE,
            self.canvas.canvasy(canvas_y) - HANDLE_SIZE,
            self.canvas.canvasx(canvas_x) + HANDLE_SIZE,
            self.canvas.canvasy(canvas_y) + HANDLE_SIZE,
        )
        for item_id in item_ids:
            tags = self.canvas.gettags(item_id)
            for tag in tags:
                if tag.startswith("handle:"):
                    return tag.split(":", 1)[1]
        return None

    def on_mouse_down(self, event: tk.Event) -> None:
        if self.base_image is None:
            return

        frame_x, frame_y = self.canvas_to_frame(event.x, event.y)
        self.drag_start = (frame_x, frame_y)

        handle = self.hit_test_handle(event.x, event.y)
        if handle and self.selected_name:
            self.drag_mode = f"resize:{handle}"
            self.drag_original = dict(self.regions[self.selected_name])
            return

        hit_name = self.hit_test_region(frame_x, frame_y)
        if hit_name:
            self.select_region(hit_name)
            self.drag_mode = "move"
            self.drag_original = dict(self.regions[hit_name])
            return

        self.drag_mode = "draw"
        self.drag_original = None
        zoom = int(self.zoom.get())
        x = frame_x * zoom
        y = frame_y * zoom
        self.pending_rect_id = self.canvas.create_rectangle(
            x,
            y,
            x,
            y,
            outline="#00aaff",
            width=2,
            dash=(4, 2),
        )

    def on_mouse_drag(self, event: tk.Event) -> None:
        if self.drag_start is None or self.drag_mode is None:
            return

        frame_x, frame_y = self.canvas_to_frame(event.x, event.y)
        start_x, start_y = self.drag_start

        if self.drag_mode == "draw" and self.pending_rect_id is not None:
            zoom = int(self.zoom.get())
            self.canvas.coords(
                self.pending_rect_id,
                start_x * zoom,
                start_y * zoom,
                frame_x * zoom,
                frame_y * zoom,
            )
            return

        if self.selected_name is None or self.drag_original is None:
            return

        original = self.drag_original
        if self.drag_mode == "move":
            dx = frame_x - start_x
            dy = frame_y - start_y
            new_x = clamp(original["x"] + dx, 0, TARGET_WIDTH - original["width"])
            new_y = clamp(original["y"] + dy, 0, TARGET_HEIGHT - original["height"])
            self.regions[self.selected_name] = {
                **original,
                "x": new_x,
                "y": new_y,
            }
        elif self.drag_mode.startswith("resize:"):
            handle = self.drag_mode.split(":", 1)[1]
            self.regions[self.selected_name] = self.resize_region(
                original,
                handle,
                frame_x,
                frame_y,
            )

        self.update_selected_form()
        self.redraw_regions()

    def on_mouse_up(self, event: tk.Event) -> None:
        if self.drag_mode == "draw" and self.drag_start is not None:
            end_x, end_y = self.canvas_to_frame(event.x, event.y)
            start_x, start_y = self.drag_start
            x1, x2 = sorted((start_x, end_x))
            y1, y2 = sorted((start_y, end_y))
            width = max(1, x2 - x1)
            height = max(1, y2 - y1)

            if width >= 2 and height >= 2:
                name = simpledialog.askstring(
                    "Region name",
                    "Name for this region:",
                    initialvalue=f"region_{len(self.regions) + 1}",
                    parent=self,
                )
                if name:
                    self.regions[name] = {
                        "x": x1,
                        "y": y1,
                        "width": width,
                        "height": height,
                    }
                    self.refresh_region_list()
                    self.select_region(name)

        if self.pending_rect_id is not None:
            self.canvas.delete(self.pending_rect_id)
            self.pending_rect_id = None

        self.drag_mode = None
        self.drag_start = None
        self.drag_original = None
        self.redraw_regions()

    def on_mouse_move(self, event: tk.Event) -> None:
        frame_x, frame_y = self.canvas_to_frame(event.x, event.y)
        self.status.set(
            f"Pixel x={frame_x}, y={frame_y} | regions={len(self.regions)}"
        )

    def resize_region(
        self,
        original: dict[str, int],
        handle: str,
        frame_x: int,
        frame_y: int,
    ) -> dict[str, int]:
        x1 = original["x"]
        y1 = original["y"]
        x2 = original["x"] + original["width"]
        y2 = original["y"] + original["height"]

        if "w" in handle:
            x1 = clamp(frame_x, 0, x2 - 1)
        if "e" in handle:
            x2 = clamp(frame_x, x1 + 1, TARGET_WIDTH)
        if "n" in handle:
            y1 = clamp(frame_y, 0, y2 - 1)
        if "s" in handle:
            y2 = clamp(frame_y, y1 + 1, TARGET_HEIGHT)

        return {
            "x": x1,
            "y": y1,
            "width": max(1, x2 - x1),
            "height": max(1, y2 - y1),
        }

    def refresh_region_list(self) -> None:
        self.region_list.delete(0, tk.END)
        for name in sorted(self.regions):
            region = self.regions[name]
            self.region_list.insert(
                tk.END,
                f"{name}  ({region['x']},{region['y']}) {region['width']}x{region['height']}",
            )

    def on_region_select(self, _event: tk.Event) -> None:
        selection = self.region_list.curselection()
        if not selection:
            return
        sorted_names = sorted(self.regions)
        self.select_region(sorted_names[selection[0]])

    def select_region(self, name: str) -> None:
        if name not in self.regions:
            return
        self.selected_name = name
        self.update_selected_form()
        self.redraw_regions()

        sorted_names = sorted(self.regions)
        index = sorted_names.index(name)
        self.region_list.selection_clear(0, tk.END)
        self.region_list.selection_set(index)
        self.region_list.see(index)

    def update_selected_form(self) -> None:
        if self.selected_name is None or self.selected_name not in self.regions:
            self.region_name.set("")
            return
        region = self.regions[self.selected_name]
        self.region_name.set(self.selected_name)
        self.x_var.set(region["x"])
        self.y_var.set(region["y"])
        self.w_var.set(region["width"])
        self.h_var.set(region["height"])

    def apply_numeric_edits(self) -> None:
        if self.selected_name is None:
            return

        new_name = self.region_name.get().strip()
        if not new_name:
            messagebox.showerror("Invalid name", "Region name cannot be empty.")
            return

        width = clamp(int(self.w_var.get()), 1, TARGET_WIDTH)
        height = clamp(int(self.h_var.get()), 1, TARGET_HEIGHT)
        x = clamp(int(self.x_var.get()), 0, TARGET_WIDTH - width)
        y = clamp(int(self.y_var.get()), 0, TARGET_HEIGHT - height)

        old_name = self.selected_name
        if new_name != old_name and new_name in self.regions:
            messagebox.showerror("Duplicate name", f"Region '{new_name}' already exists.")
            return

        if new_name != old_name:
            del self.regions[old_name]
            self.selected_name = new_name

        self.regions[new_name] = {
            "x": x,
            "y": y,
            "width": width,
            "height": height,
        }
        self.refresh_region_list()
        self.select_region(new_name)

    def rename_selected(self) -> None:
        if self.selected_name is None:
            return
        new_name = simpledialog.askstring(
            "Rename region",
            "New name:",
            initialvalue=self.selected_name,
            parent=self,
        )
        if not new_name or new_name == self.selected_name:
            return
        if new_name in self.regions:
            messagebox.showerror("Duplicate name", f"Region '{new_name}' already exists.")
            return
        self.regions[new_name] = self.regions.pop(self.selected_name)
        self.selected_name = new_name
        self.refresh_region_list()
        self.select_region(new_name)

    def delete_selected(self) -> None:
        if self.selected_name is None:
            return
        if messagebox.askyesno("Delete region", f"Delete '{self.selected_name}'?"):
            del self.regions[self.selected_name]
            self.selected_name = None
            self.refresh_region_list()
            self.update_selected_form()
            self.redraw_regions()

    def load_json(self) -> None:
        path = filedialog.askopenfilename(
            title="Load regions JSON",
            initialfile=DEFAULT_JSON,
            filetypes=[("JSON", "*.json"), ("All files", "*.*")],
        )
        if not path:
            return

        data = json.loads(Path(path).read_text(encoding="utf-8"))
        self.regions = self.normalize_loaded_regions(data)
        self.selected_name = None
        self.refresh_region_list()
        self.redraw_regions()
        self.status.set(f"Loaded {len(self.regions)} regions from {Path(path).name}.")

    def save_json(self) -> None:
        path = filedialog.asksaveasfilename(
            title="Save regions JSON",
            initialfile=DEFAULT_JSON,
            defaultextension=".json",
            filetypes=[("JSON", "*.json"), ("All files", "*.*")],
        )
        if not path:
            return

        output = {
            "resolution": {"width": TARGET_WIDTH, "height": TARGET_HEIGHT},
            "regions": dict(sorted(self.regions.items())),
        }
        Path(path).write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
        self.status.set(f"Saved {len(self.regions)} regions to {Path(path).name}.")

    def normalize_loaded_regions(self, data: object) -> dict[str, dict[str, int]]:
        if isinstance(data, dict) and "regions" in data:
            data = data["regions"]

        if not isinstance(data, dict):
            raise ValueError("Expected a JSON object or an object with a 'regions' field.")

        regions: dict[str, dict[str, int]] = {}
        for name, value in data.items():
            if not isinstance(value, dict):
                continue

            x = int(value.get("x", 0))
            y = int(value.get("y", 0))
            width = int(value.get("width", value.get("w", 1)))
            height = int(value.get("height", value.get("h", 1)))

            width = clamp(width, 1, TARGET_WIDTH)
            height = clamp(height, 1, TARGET_HEIGHT)
            x = clamp(x, 0, TARGET_WIDTH - width)
            y = clamp(y, 0, TARGET_HEIGHT - height)
            regions[str(name)] = {
                "x": x,
                "y": y,
                "width": width,
                "height": height,
            }

        return regions

    def _on_close(self) -> None:
        self.temp_dir.cleanup()
        self.destroy()


def main() -> None:
    app = RegionEditor()
    app.mainloop()


if __name__ == "__main__":
    main()
