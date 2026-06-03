# High-Level Action to GUI Action Mapping

This document summarizes the current action hierarchy used by the ResPlan to Vectorworks automation pipeline.

Source of truth:

- High-level to GUI action sequence: `scripts/generate_bim_sequences.py`
- GUI action to primitive execution: `scripts/build_primitive_plan.py`
- Current shortcut configuration: `configs/vectorworks_grounding_template.json`

## Overview

| High-level action | GUI action sequence |
|---|---|
| `CREATE_FILE` | `CREATE_NEW_FILE` -> `CONFIRM_CREATE_FILE` |
| `CREATE_EXTERIOR_WALL` | `ACTIVATE_WALL_TOOL` -> `DRAW_WALL_FROM_ENTITY_GEOMETRY` |
| `CREATE_SLAB` | `ACTIVATE_SLAB_TOOL` -> `SWITCH_SLAB_INNER_BOUNDARY_MODE` -> `GENERATE_SLAB_FROM_POINT` |
| `CREATE_INTERIOR_WALL` | `ACTIVATE_WALL_TOOL` -> `DRAW_WALL_FROM_ENTITY_GEOMETRY` |
| `CREATE_WINDOW` | `ACTIVATE_WINDOW_TOOL` -> `INSERT_WINDOW` |
| `CREATE_DOOR` | `ACTIVATE_DOOR_TOOL` -> `INSERT_DOOR` |
| `CREATE_ROOF` | `SELECT_ALL_ELEMENTS` -> `ACTIVATE_ROOF_TOOL` -> `CONFIRM_ROOF` |
| `FINAL_VIEW` | `CHANGE_VIEW` |

Current dataset count from generated GUI sequences under `outputs/bim_sequences`:

| High-level action | Count |
|---|---:|
| `CREATE_FILE` | 30 |
| `CREATE_EXTERIOR_WALL` | 362 |
| `CREATE_SLAB` | 30 |
| `CREATE_INTERIOR_WALL` | 367 |
| `CREATE_WINDOW` | 198 |
| `CREATE_DOOR` | 215 |
| `CREATE_ROOF` | 30 |
| `FINAL_VIEW` | 30 |

## Detailed Mapping

### `CREATE_FILE`

Purpose: create and initialize a new Vectorworks document.

GUI actions:

1. `CREATE_NEW_FILE`
   - Abstract grounding: keyboard shortcut `Ctrl+N`
   - Primitive execution: `HOTKEY ["ctrl", "n"]`
   - Current behavior: followed by `SLEEP 2000 ms`

2. `CONFIRM_CREATE_FILE`
   - Abstract grounding: keyboard key `Enter`
   - Primitive execution:
     - `PRESS_KEY "enter"`
     - `SLEEP 1000 ms`
     - `HOTKEY ["ctrl", "2"]`
     - `HOTKEY ["ctrl", "2"]`
   - The two `Ctrl+2` actions zoom out the drawing canvas before geometry creation.

### `CREATE_EXTERIOR_WALL`

Purpose: draw one exterior wall segment from its source geometry.

GUI actions:

1. `ACTIVATE_WALL_TOOL`
   - Abstract grounding in GUI sequence: keyboard shortcut `w`
   - Current primitive shortcut: Wall Tool = `["9"]`
   - Primitive execution: `HOTKEY ["9"]`, only when the active tool is not already Wall Tool.

2. `DRAW_WALL_FROM_ENTITY_GEOMETRY`
   - Abstract grounding: canvas point pair `wall_start_end_points`
   - Uses the wall entity geometry: `start_mm`, `end_mm`
   - Primitive execution, standard case:
     - `MOVE_TO start_mm`
     - `CLICK`
     - `MOVE_TO end_mm`
     - `DOUBLE_CLICK`
   - Short exterior wall chaining may replace the end `DOUBLE_CLICK` with `CLICK` and continue the next connected wall from the same point.

### `CREATE_SLAB`

Purpose: create a slab from an interior boundary point.

GUI actions:

1. `ACTIVATE_SLAB_TOOL`
   - Abstract grounding: keyboard shortcut `s`
   - Current primitive shortcut: Slab Tool = `["s"]`
   - Primitive execution: `HOTKEY ["s"]`

2. `SWITCH_SLAB_INNER_BOUNDARY_MODE`
   - Abstract grounding: screen point `slab_inner_boundary_mode_button`
   - Current screen point: `[292, 100]`
   - Primitive execution:
     - `MOVE_TO screen_point`
     - `CLICK`
   - Current slab `MOVE_TO` duration is `1000 ms`.

3. `GENERATE_SLAB_FROM_POINT`
   - Abstract grounding: canvas point `slab_generate_point`
   - Uses JSON field `slab_generate_point`
   - Primitive execution:
     - `MOVE_TO slab_generate_point`
     - `CLICK`
   - Current slab `MOVE_TO` duration is `1000 ms`.

### `CREATE_INTERIOR_WALL`

Purpose: draw one interior wall segment from its source geometry.

GUI actions:

1. `ACTIVATE_WALL_TOOL`
   - Abstract grounding in GUI sequence: keyboard shortcut `w`
   - Current primitive shortcut: Wall Tool = `["9"]`
   - Primitive execution: `HOTKEY ["9"]`, only when the active tool is not already Wall Tool.

2. `DRAW_WALL_FROM_ENTITY_GEOMETRY`
   - Abstract grounding: canvas point pair `wall_start_end_points`
   - Uses the wall entity geometry: `start_mm`, `end_mm`
   - Current primitive execution:
     - `MOVE_TO start_mm`
     - `CLICK`
     - `MOVE_TO end_mm`
     - `PRESS_KEY "enter"`
     - `PRESS_KEY "enter"`
   - Current interior end `MOVE_TO` duration is `500 ms`.
   - Current logic does not use `KEY_DOWN shift` or `KEY_UP shift` for interior walls.

### `CREATE_WINDOW`

Purpose: insert one window into its host wall.

GUI actions:

1. `ACTIVATE_WINDOW_TOOL`
   - Abstract grounding in GUI sequence: keyboard shortcut `Shift+W`
   - Current primitive shortcut: Window Tool = `["shift", "d"]`
   - Primitive execution: `HOTKEY ["shift", "d"]`, only when the active tool is not already Window Tool.

2. `INSERT_WINDOW`
   - Abstract grounding: canvas point `intersection_point`
   - Uses the opening `source_click_point`, derived from the JSON insertion point and current window wall-edge offset logic.
   - Primitive execution:
     - `MOVE_TO intersection_point`
     - `DOUBLE_CLICK`
   - Current opening `MOVE_TO` duration: `1000 ms`
   - Current double-click interval: `0.5 s`

### `CREATE_DOOR`

Purpose: insert one door or front door into its host wall.

GUI actions:

1. `ACTIVATE_DOOR_TOOL`
   - Abstract grounding in GUI sequence: keyboard shortcut `Shift+D`
   - Current primitive shortcut: Door Tool = `["alt", "shift", "d"]`
   - Primitive execution: `HOTKEY ["alt", "shift", "d"]`, only when the active tool is not already Door Tool.

2. `INSERT_DOOR`
   - Abstract grounding: canvas point `intersection_point`
   - Uses the opening `source_click_point`, derived from the JSON insertion point and current door wall-edge offset logic.
   - Primitive execution:
     - `MOVE_TO intersection_point`
     - `DOUBLE_CLICK`
   - Current opening `MOVE_TO` duration: `1000 ms`
   - Current double-click interval: `0.5 s`

### `CREATE_ROOF`

Purpose: select model elements and create/confirm a roof.

GUI actions:

1. `SELECT_ALL_ELEMENTS`
   - Abstract grounding: keyboard shortcut `Ctrl+A`
   - Current primitive shortcut: Select All = `["ctrl", "a"]`
   - Primitive execution: `HOTKEY ["ctrl", "a"]`

2. `ACTIVATE_ROOF_TOOL`
   - Abstract grounding: keyboard shortcut `Ctrl+Alt+1`
   - Current primitive shortcut: Roof Tool = `["ctrl", "alt", "1"]`
   - Primitive execution: `HOTKEY ["ctrl", "alt", "1"]`

3. `CONFIRM_ROOF`
   - Abstract grounding: keyboard key `Enter`
   - Current primitive execution:
     - `PRESS_KEY "enter"`
     - `PRESS_KEY "enter"`

### `FINAL_VIEW`

Purpose: high-level placeholder for final view state.

GUI actions:

1. `CHANGE_VIEW`
   - Abstract grounding: keyboard shortcut `1`
   - Current primitive execution: no primitive is emitted.
   - The primitive builder resets active tool state and returns.

## Notes

- GUI sequence grounding names are abstract and not always identical to the current primitive shortcuts. For execution, `configs/vectorworks_grounding_template.json` is authoritative.
- `VIEW_CHECK` was removed from the current generated sequence. Historical plans may still contain 3D view or drag-view actions, but current plans should not use them.
- `TYPE_TEXT`, `SCROLL`, `MOVE_REL`, and `DRAG_REL` are not part of the current primitive action space.
- The imitation training trajectory currently keeps key state-changing primitives (`CLICK`, `DOUBLE_CLICK`, `HOTKEY`, `PRESS_KEY`) and attaches relevant movement/screenshot context, while the full primitive log remains available for debugging.
