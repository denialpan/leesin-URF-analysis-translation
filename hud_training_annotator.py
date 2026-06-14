from __future__ import annotations

import hashlib
import json
import random
import subprocess
import sys
import tempfile
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

from region_pixel_zoom_editor import (
    TARGET_HEIGHT,
    TARGET_WIDTH,
    VIDEO_EXTENSIONS,
    parse_timestamp,
    probe_frame_rate,
)


DEFAULT_REGIONS = "hud-regions.json"
DEFAULT_DATASET = "training-data"
DEFAULT_LABELS = (
    "ready",
    "cooldown",
    "disabled",
    "missing",
    "uncertain",
    "recast",
)
ZOOM_LEVELS = (2, 4, 6, 8, 10, 12, 16)
CREATE_NO_WINDOW = 0x08000000 if sys.platform == "win32" else 0


def safe_component(value: str) -> str:
    cleaned = "".join(
        character if character.isalnum() or character in "-_." else "_"
        for character in value
    ).strip("._")
    return cleaned or "unnamed"


def source_id(path: Path) -> str:
    digest = hashlib.sha1(str(path.resolve()).encode("utf-8")).hexdigest()[:10]
    return f"{safe_component(path.stem)[:60]}_{digest}"


def annotation_key(video: str, frame: int, region: str) -> tuple[str, int, str]:
    return video, frame, region


def probe_video_info(path: Path) -> tuple[float, int, float]:
    command = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=avg_frame_rate,nb_frames,duration",
        "-of",
        "json",
        str(path),
    ]
    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            creationflags=CREATE_NO_WINDOW,
            check=False,
        )
    except FileNotFoundError as error:
        raise RuntimeError("ffprobe was not found on PATH.") from error

    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or "Could not inspect the video.")

    streams = json.loads(result.stdout).get("streams", [])
    if not streams:
        raise RuntimeError("The selected file contains no video stream.")

    stream = streams[0]
    fps = probe_frame_rate(path)
    duration = float(stream.get("duration") or 0)
    frame_count_text = stream.get("nb_frames")
    frame_count = int(frame_count_text) if frame_count_text not in (None, "N/A") else 0
    if frame_count <= 0 and duration > 0:
        frame_count = max(1, round(duration * fps))
    if duration <= 0 and frame_count > 0:
        duration = frame_count / fps
    return fps, frame_count, duration


def load_regions(path: Path) -> dict[str, dict[str, int]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict) and "regions" in data:
        data = data["regions"]
    if not isinstance(data, dict):
        raise ValueError("Expected an object containing HUD regions.")

    regions: dict[str, dict[str, int]] = {}
    for name, value in data.items():
        if not isinstance(value, dict):
            continue
        region = {
            "x": int(value["x"]),
            "y": int(value["y"]),
            "width": int(value.get("width", value.get("w"))),
            "height": int(value.get("height", value.get("h"))),
        }
        if (
            region["width"] <= 0
            or region["height"] <= 0
            or region["x"] < 0
            or region["y"] < 0
            or region["x"] + region["width"] > TARGET_WIDTH
            or region["y"] + region["height"] > TARGET_HEIGHT
        ):
            raise ValueError(f"Region '{name}' is outside 1920x1080.")
        regions[str(name)] = region

    if not regions:
        raise ValueError("No valid regions were found.")
    return regions


class HudTrainingAnnotator(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("HUD Training Annotator")
        self.geometry("1180x820")
        self.minsize(900, 650)

        self.video_path: Path | None = None
        self.regions_path: Path | None = None
        self.dataset_path = Path(DEFAULT_DATASET)
        self.regions: dict[str, dict[str, int]] = {}
        self.annotations: list[dict[str, object]] = []
        self.annotation_lookup: dict[tuple[str, int, str], int] = {}

        self.fps = 60.0
        self.frame_count = 0
        self.duration = 0.0
        self.current_frame = 0
        self.temp_dir = tempfile.TemporaryDirectory(prefix="hud_annotator_")
        self.frame_generation = 0

        self.frame_image: tk.PhotoImage | None = None
        self.crop_image: tk.PhotoImage | None = None
        self.zoomed_crop: tk.PhotoImage | None = None

        self.frame_var = tk.StringVar(value="0")
        self.time_var = tk.StringVar(value="0")
        self.step_var = tk.IntVar(value=1)
        self.zoom_var = tk.IntVar(value=10)
        self.region_var = tk.StringVar()
        self.auto_advance = tk.BooleanVar(value=True)
        self.status = tk.StringVar(value="Load regions and a video to begin.")
        self.video_info = tk.StringVar(value="No video loaded")
        self.annotation_info = tk.StringVar(value="No annotation at this frame")

        self._build_ui()
        self._bind_shortcuts()
        self.protocol("WM_DELETE_WINDOW", self._on_close)
        self._load_default_regions()
        self._load_annotations()

    def _build_ui(self) -> None:
        toolbar = ttk.Frame(self)
        toolbar.pack(fill=tk.X, padx=6, pady=6)

        ttk.Button(toolbar, text="Open Video", command=self.open_video).pack(
            side=tk.LEFT, padx=3
        )
        ttk.Button(toolbar, text="Load Regions", command=self.choose_regions).pack(
            side=tk.LEFT, padx=3
        )
        ttk.Button(toolbar, text="Dataset Folder", command=self.choose_dataset).pack(
            side=tk.LEFT, padx=3
        )
        ttk.Label(toolbar, textvariable=self.video_info).pack(side=tk.LEFT, padx=12)

        navigation = ttk.LabelFrame(self, text="Frame Navigation")
        navigation.pack(fill=tk.X, padx=8, pady=(0, 6))

        ttk.Button(navigation, text="|<", command=lambda: self.go_to_frame(0)).pack(
            side=tk.LEFT, padx=(6, 2), pady=6
        )
        ttk.Button(navigation, text="< Step", command=lambda: self.step_frame(-1)).pack(
            side=tk.LEFT, padx=2
        )
        ttk.Button(navigation, text="Step >", command=lambda: self.step_frame(1)).pack(
            side=tk.LEFT, padx=2
        )
        ttk.Label(navigation, text="Step").pack(side=tk.LEFT, padx=(10, 2))
        ttk.Spinbox(
            navigation, from_=1, to=10000, textvariable=self.step_var, width=7
        ).pack(side=tk.LEFT)

        ttk.Label(navigation, text="Frame").pack(side=tk.LEFT, padx=(14, 2))
        frame_entry = ttk.Entry(navigation, textvariable=self.frame_var, width=10)
        frame_entry.pack(side=tk.LEFT)
        frame_entry.bind("<Return>", lambda _event: self.load_frame_entry())
        ttk.Button(navigation, text="Go", command=self.load_frame_entry).pack(
            side=tk.LEFT, padx=2
        )

        ttk.Label(navigation, text="Time").pack(side=tk.LEFT, padx=(14, 2))
        time_entry = ttk.Entry(navigation, textvariable=self.time_var, width=14)
        time_entry.pack(side=tk.LEFT)
        time_entry.bind("<Return>", lambda _event: self.load_time_entry())
        ttk.Button(navigation, text="Go", command=self.load_time_entry).pack(
            side=tk.LEFT, padx=2
        )

        content = ttk.Frame(self)
        content.pack(fill=tk.BOTH, expand=True, padx=8)

        preview_panel = ttk.LabelFrame(content, text="Selected HUD Crop")
        preview_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 6))

        preview_controls = ttk.Frame(preview_panel)
        preview_controls.pack(fill=tk.X, padx=6, pady=6)
        ttk.Label(preview_controls, text="Region").pack(side=tk.LEFT)
        self.region_box = ttk.Combobox(
            preview_controls,
            textvariable=self.region_var,
            state="readonly",
            width=16,
        )
        self.region_box.pack(side=tk.LEFT, padx=4)
        self.region_box.bind(
            "<<ComboboxSelected>>", lambda _event: self.refresh_crop()
        )
        ttk.Label(preview_controls, text="Pixel zoom").pack(
            side=tk.LEFT, padx=(14, 2)
        )
        zoom_box = ttk.Combobox(
            preview_controls,
            textvariable=self.zoom_var,
            values=ZOOM_LEVELS,
            state="readonly",
            width=5,
        )
        zoom_box.pack(side=tk.LEFT)
        zoom_box.bind("<<ComboboxSelected>>", lambda _event: self.refresh_crop())

        canvas_frame = ttk.Frame(preview_panel)
        canvas_frame.pack(fill=tk.BOTH, expand=True, padx=6, pady=(0, 6))
        self.canvas = tk.Canvas(
            canvas_frame,
            bg="#171717",
            highlightthickness=0,
        )
        x_scroll = ttk.Scrollbar(
            canvas_frame, orient=tk.HORIZONTAL, command=self.canvas.xview
        )
        y_scroll = ttk.Scrollbar(
            canvas_frame, orient=tk.VERTICAL, command=self.canvas.yview
        )
        self.canvas.configure(xscrollcommand=x_scroll.set, yscrollcommand=y_scroll.set)
        self.canvas.grid(row=0, column=0, sticky="nsew")
        y_scroll.grid(row=0, column=1, sticky="ns")
        x_scroll.grid(row=1, column=0, sticky="ew")
        canvas_frame.rowconfigure(0, weight=1)
        canvas_frame.columnconfigure(0, weight=1)

        side_panel = ttk.Frame(content, width=330)
        side_panel.pack(side=tk.RIGHT, fill=tk.Y)
        side_panel.pack_propagate(False)

        label_frame = ttk.LabelFrame(side_panel, text="Assign State")
        label_frame.pack(fill=tk.X)
        for number, label in enumerate(DEFAULT_LABELS, start=1):
            ttk.Button(
                label_frame,
                text=f"{number}  {label.title()}",
                command=lambda selected=label: self.assign_label(selected),
            ).pack(fill=tk.X, padx=8, pady=3)

        ttk.Checkbutton(
            label_frame,
            text="Random region/frame after labeling",
            variable=self.auto_advance,
        ).pack(anchor=tk.W, padx=8, pady=7)

        current_frame = ttk.LabelFrame(side_panel, text="Current Sample")
        current_frame.pack(fill=tk.X, pady=8)
        ttk.Label(
            current_frame,
            textvariable=self.annotation_info,
            justify=tk.LEFT,
            wraplength=300,
        ).pack(anchor=tk.W, padx=8, pady=8)

        history_frame = ttk.LabelFrame(side_panel, text="Recent Annotations")
        history_frame.pack(fill=tk.BOTH, expand=True)
        self.history = tk.Listbox(history_frame, exportselection=False)
        self.history.pack(fill=tk.BOTH, expand=True, padx=6, pady=6)
        self.history.bind("<Double-Button-1>", self.open_history_item)

        ttk.Label(self, textvariable=self.status, anchor=tk.W).pack(
            fill=tk.X, padx=8, pady=5
        )

    def _bind_shortcuts(self) -> None:
        for number, label in enumerate(DEFAULT_LABELS, start=1):
            self.bind(
                str(number),
                lambda event, selected=label: self._label_shortcut(event, selected),
            )
        self.bind("<Left>", lambda _event: self.step_frame(-1))
        self.bind("<Right>", lambda _event: self.step_frame(1))
        self.bind("<Prior>", lambda _event: self.step_frame(-10))
        self.bind("<Next>", lambda _event: self.step_frame(10))

    def _label_shortcut(self, event: tk.Event, label: str) -> None:
        widget_class = event.widget.winfo_class()
        if widget_class in {"Entry", "TEntry", "Spinbox", "TSpinbox"}:
            return
        self.assign_label(label)

    def _load_default_regions(self) -> None:
        path = Path(DEFAULT_REGIONS)
        if path.exists():
            self.load_regions_file(path)

    def choose_regions(self) -> None:
        path = filedialog.askopenfilename(
            title="Load HUD regions",
            initialfile=DEFAULT_REGIONS,
            filetypes=[("JSON", "*.json"), ("All files", "*.*")],
        )
        if path:
            self.load_regions_file(Path(path))

    def load_regions_file(self, path: Path) -> None:
        try:
            regions = load_regions(path)
        except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as error:
            messagebox.showerror("Region error", str(error))
            return

        previous = self.region_var.get()
        self.regions_path = path
        self.regions = regions
        names = list(regions)
        self.region_box["values"] = names
        self.region_var.set(previous if previous in regions else names[0])
        self.status.set(f"Loaded {len(regions)} regions from {path.name}.")
        self.refresh_crop()

    def open_video(self) -> None:
        path = filedialog.askopenfilename(
            title="Open training video",
            filetypes=[
                ("Videos", "*.mp4 *.mov *.mkv *.webm *.avi *.m4v *.flv *.wmv"),
                ("All files", "*.*"),
            ],
        )
        if not path:
            return

        video_path = Path(path)
        if video_path.suffix.lower() not in VIDEO_EXTENSIONS:
            messagebox.showerror("Video error", "Select a supported video file.")
            return

        try:
            self.fps, self.frame_count, self.duration = probe_video_info(video_path)
        except (RuntimeError, ValueError, json.JSONDecodeError) as error:
            messagebox.showerror("Video error", str(error))
            return

        self.video_path = video_path
        self.current_frame = 0
        self.video_info.set(
            f"{video_path.name} | {self.fps:g} fps | "
            f"{self.frame_count:,} frames | {self.duration:.2f}s"
        )
        self._load_annotations()
        self.go_to_frame(0)

    def choose_dataset(self) -> None:
        path = filedialog.askdirectory(
            title="Select training dataset folder",
            initialdir=str(self.dataset_path),
        )
        if not path:
            return
        self.dataset_path = Path(path)
        self._load_annotations()
        self.status.set(f"Dataset folder: {self.dataset_path}")

    @property
    def annotations_path(self) -> Path:
        return self.dataset_path / "annotations.json"

    def _load_annotations(self) -> None:
        self.annotations = []
        if self.annotations_path.exists():
            try:
                data = json.loads(self.annotations_path.read_text(encoding="utf-8"))
                entries = data.get("annotations", data) if isinstance(data, dict) else data
                if isinstance(entries, list):
                    self.annotations = [
                        entry for entry in entries if isinstance(entry, dict)
                    ]
            except (OSError, json.JSONDecodeError):
                messagebox.showwarning(
                    "Dataset warning",
                    f"Could not read {self.annotations_path}. Starting with an empty index.",
                )

        self.annotation_lookup = {}
        for index, entry in enumerate(self.annotations):
            try:
                key = annotation_key(
                    str(entry["video"]),
                    int(entry["frame"]),
                    str(entry["region"]),
                )
                self.annotation_lookup[key] = index
            except (KeyError, TypeError, ValueError):
                continue
        self.refresh_history()
        self.refresh_annotation_info()

    def load_frame_entry(self) -> None:
        try:
            frame = int(self.frame_var.get())
        except ValueError:
            messagebox.showerror("Invalid frame", "Frame must be an integer.")
            return
        self.go_to_frame(frame)

    def load_time_entry(self) -> None:
        try:
            seconds = parse_timestamp(self.time_var.get(), self.fps)
        except ValueError as error:
            messagebox.showerror("Invalid time", str(error))
            return
        self.go_to_frame(round(seconds * self.fps))

    def step_frame(self, direction: int) -> None:
        try:
            step = max(1, int(self.step_var.get()))
        except (ValueError, tk.TclError):
            step = 1
            self.step_var.set(step)
        self.go_to_frame(self.current_frame + direction * step)

    def go_to_frame(self, frame: int) -> None:
        if self.video_path is None:
            return
        maximum = max(0, self.frame_count - 1)
        frame = max(0, min(maximum, int(frame)))
        seconds = frame / self.fps
        extracted = self._extract_frame(seconds)
        if extracted is None:
            return

        try:
            self.frame_image = tk.PhotoImage(file=str(extracted), format="PPM")
        except tk.TclError as error:
            messagebox.showerror("Preview error", str(error))
            return
        finally:
            extracted.unlink(missing_ok=True)

        self.current_frame = frame
        self.frame_var.set(str(frame))
        self.time_var.set(f"{seconds:.6f}")
        self.refresh_crop()
        self.refresh_annotation_info()
        self.status.set(
            f"Frame {frame:,}/{maximum:,} at {seconds:.3f}s. "
            "Shortcuts: 1-6 label, arrows step."
        )

    def _extract_frame(self, seconds: float) -> Path | None:
        if self.video_path is None:
            return None
        self.frame_generation += 1
        output = Path(self.temp_dir.name) / f"frame_{self.frame_generation:06d}.ppm"
        command = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-ss",
            f"{seconds:.9f}",
            "-i",
            str(self.video_path),
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
        try:
            result = subprocess.run(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                creationflags=CREATE_NO_WINDOW,
                check=False,
            )
        except FileNotFoundError:
            messagebox.showerror("ffmpeg missing", "ffmpeg was not found on PATH.")
            return None

        if result.returncode != 0 or not result.stdout:
            details = result.stderr.decode(errors="replace").strip()
            messagebox.showerror(
                "Frame extraction failed",
                details or f"No image was produced at {seconds:.3f} seconds.",
            )
            return None
        output.write_bytes(result.stdout)
        return output

    def refresh_crop(self) -> None:
        region_name = self.region_var.get()
        if self.frame_image is None or region_name not in self.regions:
            return

        region = self.regions[region_name]
        crop = tk.PhotoImage(width=region["width"], height=region["height"])
        crop.tk.call(
            str(crop),
            "copy",
            str(self.frame_image),
            "-from",
            region["x"],
            region["y"],
            region["x"] + region["width"],
            region["y"] + region["height"],
            "-to",
            0,
            0,
        )
        zoom = int(self.zoom_var.get())
        self.crop_image = crop
        self.zoomed_crop = crop.zoom(zoom, zoom)

        self.canvas.delete("all")
        self.canvas.create_image(
            0, 0, image=self.zoomed_crop, anchor=tk.NW, tags=("crop",)
        )
        self.canvas.configure(
            scrollregion=(
                0,
                0,
                region["width"] * zoom,
                region["height"] * zoom,
            )
        )
        self.refresh_annotation_info()

    def current_annotation_key(self) -> tuple[str, int, str] | None:
        if self.video_path is None or not self.region_var.get():
            return None
        return annotation_key(
            str(self.video_path.resolve()),
            self.current_frame,
            self.region_var.get(),
        )

    def assign_label(self, label: str) -> None:
        if self.video_path is None or self.crop_image is None:
            messagebox.showinfo("Nothing to label", "Load a video frame first.")
            return
        region_name = self.region_var.get()
        if region_name not in self.regions:
            return

        region = self.regions[region_name]
        relative_crop = (
            Path("images")
            / safe_component(region_name)
            / safe_component(label)
            / f"{source_id(self.video_path)}_frame_{self.current_frame:08d}.png"
        )
        crop_path = self.dataset_path / relative_crop
        crop_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            self.crop_image.write(str(crop_path), format="png")
        except tk.TclError as error:
            messagebox.showerror("Crop error", f"Could not save PNG crop:\n{error}")
            return

        key = self.current_annotation_key()
        if key is None:
            return
        existing_index = self.annotation_lookup.get(key)
        if existing_index is not None:
            previous_crop = self.dataset_path / str(
                self.annotations[existing_index].get("crop", "")
            )
            if previous_crop != crop_path and previous_crop.is_file():
                previous_crop.unlink()

        entry: dict[str, object] = {
            "video": str(self.video_path.resolve()),
            "frame": self.current_frame,
            "timestamp": round(self.current_frame / self.fps, 9),
            "fps": self.fps,
            "region": region_name,
            "label": label,
            "crop": relative_crop.as_posix(),
            "region_box": dict(region),
        }
        if existing_index is None:
            self.annotations.append(entry)
            self.annotation_lookup[key] = len(self.annotations) - 1
        else:
            self.annotations[existing_index] = entry

        self._save_annotations()
        self.refresh_history()
        self.refresh_annotation_info()
        self.status.set(
            f"Saved {region_name} = {label} at frame {self.current_frame:,}."
        )
        if self.auto_advance.get():
            self.go_to_random_sample()

    def go_to_random_sample(self) -> None:
        if self.video_path is None or not self.regions or self.frame_count <= 0:
            return

        region_names = list(self.regions)
        if len(region_names) > 1:
            alternatives = [
                name for name in region_names if name != self.region_var.get()
            ]
            self.region_var.set(random.choice(alternatives))
        else:
            self.region_var.set(region_names[0])

        if self.frame_count > 1:
            next_frame = random.randrange(self.frame_count - 1)
            if next_frame >= self.current_frame:
                next_frame += 1
        else:
            next_frame = 0
        self.go_to_frame(next_frame)

    def _save_annotations(self) -> None:
        self.dataset_path.mkdir(parents=True, exist_ok=True)
        payload = {
            "format_version": 1,
            "labels": list(DEFAULT_LABELS),
            "annotations": self.annotations,
        }
        temporary = self.annotations_path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        temporary.replace(self.annotations_path)

    def refresh_annotation_info(self) -> None:
        key = self.current_annotation_key()
        if key is None or key not in self.annotation_lookup:
            self.annotation_info.set(
                f"Frame: {self.current_frame}\n"
                f"Region: {self.region_var.get() or '-'}\n"
                "Label: not labeled"
            )
            return
        entry = self.annotations[self.annotation_lookup[key]]
        self.annotation_info.set(
            f"Frame: {entry['frame']}\n"
            f"Region: {entry['region']}\n"
            f"Label: {entry['label']}\n"
            f"Time: {float(entry['timestamp']):.3f}s"
        )

    def refresh_history(self) -> None:
        self.history.delete(0, tk.END)
        relevant = self.annotations
        if self.video_path is not None:
            video = str(self.video_path.resolve())
            relevant = [entry for entry in relevant if entry.get("video") == video]
        for entry in reversed(relevant[-250:]):
            self.history.insert(
                tk.END,
                f"{int(entry.get('frame', 0)):>7}  "
                f"{entry.get('region', '?'):<8}  {entry.get('label', '?')}",
            )

    def open_history_item(self, _event: tk.Event) -> None:
        selection = self.history.curselection()
        if not selection or self.video_path is None:
            return
        video = str(self.video_path.resolve())
        relevant = [entry for entry in self.annotations if entry.get("video") == video]
        relevant = list(reversed(relevant[-250:]))
        entry = relevant[selection[0]]
        region = str(entry.get("region", ""))
        if region in self.regions:
            self.region_var.set(region)
        self.go_to_frame(int(entry["frame"]))

    def _on_close(self) -> None:
        self.temp_dir.cleanup()
        self.destroy()


def main() -> None:
    app = HudTrainingAnnotator()
    app.mainloop()


if __name__ == "__main__":
    main()
