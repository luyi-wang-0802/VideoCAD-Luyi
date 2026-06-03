# Low-Level GUI Sequence Data Process

This folder contains the data processing step for the Vectorworks low-level GUI sequence dataset.

## Purpose

Convert collected rule-based Vectorworks runs into training-ready samples where the model predicts only state-changing GUI actions:

```text
CLICK
DOUBLE_CLICK
HOTKEY
PRESS_KEY
```

`MOVE_TO` and `SLEEP` are not policy actions. Click targets are attached to `CLICK` / `DOUBLE_CLICK`, and the interior-wall end-point move is attached to the first `PRESS_KEY enter`.

Screenshots captured after each key/mouse action are kept as visual observations. For each policy action:

```text
observation_before = previous policy action's after-screenshot
observation_after  = current policy action's after-screenshot
```

The first action has no recorded before screenshot because the collector starts capturing after the first executed primitive.

## Inputs

Run from the repository root. Default inputs are:

```text
resplan_to_JSON/
outputs/bim_sequences/
outputs/runs/
```

## Command

```powershell
python data_process\low_level_gui_sequence\prepare_low_level_gui_dataset.py
```

## Outputs

Default output folder:

```text
data_process/low_level_gui_sequence/results/
  samples/
    plan_00431.json
    plan_00496.json
    plan_01016.json
    ...
  dataset_index.json
  dataset_split.json
  action_vocab.json
  summary.json
```

`samples/*.json` contains one full training sample per plan. `dataset_index.json` is the loader-friendly index for training.

Each sample contains:

```text
compact_plan
high_level_sequence
gui_sequence
low_level_actions
```

Each item in `low_level_actions` contains the action label plus `observation_before` and `observation_after` screenshot paths.
