"""Execute decoded low-level GUI policy actions in Vectorworks."""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def read_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def import_pyautogui() -> Any:
    try:
        import pyautogui  # type: ignore
    except ImportError as exc:
        raise SystemExit("Install pyautogui before live rollout: pip install pyautogui") from exc
    return pyautogui


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def resolve_model_point(point: list[float], calibration: dict[str, Any]) -> tuple[int, int]:
    canvas = calibration["canvas"]
    coordinate_mapping = canvas.get("coordinate_mapping", {})
    method = coordinate_mapping.get("method")
    if method != "centered_canvas_range_uniform":
        raise ValueError(f"Unsupported coordinate mapping method: {method}")

    center = coordinate_mapping["model_center_screen"]
    corners = coordinate_mapping["canvas_corners"]
    edge_margin_px = float(coordinate_mapping.get("edge_margin_px", 0))
    model_range = canvas["model_range_mm"]
    model_x = float(point[0])
    model_y = float(point[1])
    center_x = float(center["x"])
    center_y = float(center["y"])
    scale_multiplier = float(coordinate_mapping.get("scale_multiplier", 1.0))
    x_min = float(model_range["x_min"])
    x_max = float(model_range["x_max"])
    y_min = float(model_range["y_min"])
    y_max = float(model_range["y_max"])
    model_center_x = (x_min + x_max) / 2
    model_center_y = (y_min + y_max) / 2

    left_x = (float(corners["top_left"]["x"]) + float(corners["bottom_left"]["x"])) / 2 + edge_margin_px
    right_x = (float(corners["top_right"]["x"]) + float(corners["bottom_right"]["x"])) / 2 - edge_margin_px
    top_y = (float(corners["top_left"]["y"]) + float(corners["top_right"]["y"])) / 2 + edge_margin_px
    bottom_y = (float(corners["bottom_left"]["y"]) + float(corners["bottom_right"]["y"])) / 2 - edge_margin_px
    available_left = center_x - left_x
    available_right = right_x - center_x
    available_top = center_y - top_y
    available_bottom = bottom_y - center_y
    pixels_per_unit = min(
        available_left / abs(model_center_x - x_min),
        available_right / abs(x_max - model_center_x),
        available_top / abs(y_max - model_center_y),
        available_bottom / abs(model_center_y - y_min),
    ) * scale_multiplier

    x = center_x + (model_x - model_center_x) * pixels_per_unit
    y = center_y - (model_y - model_center_y) * pixels_per_unit
    return int(round(x)), int(round(y))


def clamp_to_canvas(x: int, y: int, calibration: dict[str, Any]) -> tuple[int, int]:
    rect = calibration.get("canvas", {}).get("rect")
    if not rect:
        return x, y
    left = int(rect["left"])
    top = int(rect["top"])
    right = left + int(rect["width"])
    bottom = top + int(rect["height"])
    return max(left, min(right, x)), max(top, min(bottom, y))


def screenshot_path(run_dir: Path, step_index: int) -> Path:
    screenshots_dir = run_dir / "screenshots"
    screenshots_dir.mkdir(parents=True, exist_ok=True)
    return screenshots_dir / f"policy_{step_index:05d}.png"


class VectorworksExecutor:
    def __init__(
        self,
        calibration_path: str | Path = "configs/vectorworks_grounding_template.json",
        dry_run: bool = False,
    ) -> None:
        self.calibration_path = Path(calibration_path)
        self.calibration = read_json(self.calibration_path)
        self.dry_run = dry_run
        self.pyautogui = None if dry_run else import_pyautogui()
        self.pause_after_action_s = float(self.calibration.get("timing", {}).get("pause_after_action_s", 0.15))

    def resolve_action_point(self, action: dict[str, Any]) -> tuple[int, int] | None:
        target = action.get("target") or {}
        model_point = target.get("model_point")
        if model_point is None:
            return None
        x, y = resolve_model_point(model_point, self.calibration)
        return clamp_to_canvas(x, y, self.calibration)

    def execute(self, action: dict[str, Any]) -> dict[str, Any]:
        action_type = action["action_type"]
        event: dict[str, Any] = {
            "timestamp_utc_before": utc_now_iso(),
            "action": action,
            "dry_run": self.dry_run,
        }
        point = self.resolve_action_point(action)
        if point is not None:
            event["resolved_screen_point"] = {"x": point[0], "y": point[1]}

        if self.dry_run:
            event["timestamp_utc_after"] = utc_now_iso()
            return event

        assert self.pyautogui is not None
        if action_type == "CLICK":
            if point is not None:
                self.pyautogui.moveTo(point[0], point[1], duration=0.08)
            self.pyautogui.click(button=action.get("button", "left"))
        elif action_type == "DOUBLE_CLICK":
            if point is not None:
                self.pyautogui.moveTo(point[0], point[1], duration=0.08)
            self.pyautogui.doubleClick(button=action.get("button", "left"))
        elif action_type == "HOTKEY":
            presses = max(int(action.get("repeat_count", 1)), 1)
            interval = max(float(action.get("key_interval_ms", 0.0)), 0.0) / 1000.0
            for press_index in range(presses):
                self.pyautogui.hotkey(*action["keys"])
                if press_index < presses - 1 and interval > 0:
                    time.sleep(interval)
        elif action_type == "PRESS_KEY":
            if point is not None:
                self.pyautogui.moveTo(point[0], point[1], duration=0.08)
            presses = max(int(action.get("repeat_count", 1)), 1)
            interval = max(float(action.get("key_interval_ms", 0.0)), 0.0) / 1000.0
            self.pyautogui.press(action["key"], presses=presses, interval=interval)
        else:
            raise ValueError(f"Unsupported action_type: {action_type}")

        time.sleep(self.pause_after_action_s)
        event["timestamp_utc_after"] = utc_now_iso()
        return event

    def capture_screenshot(self, run_dir: str | Path, step_index: int) -> str | None:
        if self.dry_run:
            return None
        assert self.pyautogui is not None
        path = screenshot_path(Path(run_dir), step_index)
        image = self.pyautogui.screenshot()
        image.save(path)
        return str(path).replace("\\", "/")
