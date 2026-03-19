#!/usr/bin/env python3
"""
Mass evaluator for ViZDoom agents.

This script is intentionally very close to eval_models.py, but it evaluates
many model/scenario/weights combinations from a manifest file and writes a
single results table to disk instead of printing only summary statistics.

Accepted manifest formats
-------------------------
1) JSON
   Either a list of entries:
       [
         {
           "model_type": "ppo_cnn_gray",
           "scenario": "defend_the_center",
           "model_filename": "ppo_cnn_gray.pth"
         }
       ]

   or a dict with a top-level "models" list:
       {
         "models": [ ...entries... ]
       }

2) CSV
   Required columns:
       model_type,scenario,model_filename

3) Python literal text (.py/.txt/.cfg/.manifest)
   A Python literal containing either a list of dicts or a dict with a
   top-level "models" key. This is parsed with ast.literal_eval, not exec().

Manifest entry schema
---------------------
- model_type:      model key from model_registry.py
- scenario:        scenario name with or without ".cfg"
- model_filename:  local weights filename only, e.g. "ppo_cnn_gray.pth"
                   (no parent directories; the script resolves it to
                    ../models/<scenario_stem>/<model_filename>)

Output formats
--------------
- .csv  -> CSV output
- .xlsx -> Excel workbook output
- no extension -> ".csv" is appended

Output table layout
-------------------
The first column is "Episode".
Each remaining column header is the resolved model filepath, for example:
    ../models/defend_the_center/ppo_cnn_gray.pth

Rows 1..N contain per-episode rewards.
The final four rows are labeled in the Episode column as:
    mean
    std
    min
    max
"""

from __future__ import annotations

import argparse
import ast
import csv
import itertools as it
import json
import os
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Optional

import numpy as np
import torch
import vizdoom as vzd

from utils import *
import model_registry as MODELS

import q_rainbow_stacked as rainbow_lazy_mod
import ppo_late_fusion_rgb_corridor
import ppo_film_factorized_gray

try:
    import ppo_film_factorized_gray_cig
except ImportError:
    ppo_film_factorized_gray_cig = None


# ---------- GLOBALS (kept aligned with eval_models.py) ----------
learning_rate = 0.00025
discount_factor = 0.99
replay_memory_size = 10000
batch_size = 64
frame_repeat = 12


if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
    torch.backends.cudnn.benchmark = True
else:
    DEVICE = torch.device("cpu")


@dataclass(frozen=True)
class ManifestEntry:
    model_type: str
    scenario_cfg: str
    model_filename: str

    @property
    def scenario_stem(self) -> str:
        return Path(self.scenario_cfg).stem

    @property
    def model_path(self) -> Path:
        return Path("../models") / self.scenario_stem / self.model_filename

    @property
    def column_name(self) -> str:
        return str(self.model_path)


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------
def str2bool(v):
    if isinstance(v, bool):
        return v
    v = v.strip().lower()
    if v in {"true", "t", "1", "yes", "y"}:
        return True
    if v in {"false", "f", "0", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: '{v}'. Use True/False.")


def parse_cli():
    parser = argparse.ArgumentParser(
        description="Evaluate many ViZDoom agents from a manifest and save all episode rewards."
    )
    parser.add_argument(
        "manifest_path",
        type=str,
        help="Path to manifest file (.json, .csv, or Python literal text).",
    )
    parser.add_argument(
        "-n", "--episodes",
        type=int,
        default=1000,
        help="Number of evaluation episodes per model (default: 1000).",
    )
    parser.add_argument(
        "-sc", "--scenario",
        type=str,
        default="ALL",
        help="Scenario filter. Use ALL to evaluate every manifest entry, or a scenario such as defend_the_center (default: ALL).",
    )
    parser.add_argument(
        "-o", "--output",
        type=str,
        default="model_eval_data",
        help="Output filename. If no extension is given, .csv is appended (default: model_eval_data).",
    )
    parser.add_argument(
        "-s", "--show",
        type=str2bool,
        default=False,
        metavar="BOOL",
        help="Show the ViZDoom window while evaluating (default: False).",
    )
    return parser.parse_args()


# -----------------------------------------------------------------------------
# Manifest parsing
# -----------------------------------------------------------------------------
def normalize_scenario_name(name: str) -> str:
    text = str(name).strip()
    if not text:
        raise ValueError("Scenario cannot be empty.")
    return text if text.endswith(".cfg") else f"{text}.cfg"


def _first_present(mapping: dict[str, Any], keys: Iterable[str]) -> Any:
    for key in keys:
        if key in mapping:
            return mapping[key]
    raise KeyError(f"Missing required key. Expected one of: {', '.join(keys)}")


def _coerce_manifest_entries(raw: Any) -> list[ManifestEntry]:
    if isinstance(raw, dict):
        if "models" not in raw:
            raise ValueError("Manifest dict must contain a top-level 'models' key.")
        raw_entries = raw["models"]
    elif isinstance(raw, list):
        raw_entries = raw
    else:
        raise ValueError("Manifest must be either a list of entries or a dict with a 'models' key.")

    entries: list[ManifestEntry] = []
    for idx, item in enumerate(raw_entries, start=1):
        if not isinstance(item, dict):
            raise ValueError(f"Manifest entry #{idx} must be a dictionary.")

        try:
            model_type = str(_first_present(item, ("model_type", "type"))).strip()
            scenario = normalize_scenario_name(_first_present(item, ("scenario", "scenario_name")))
            model_filename = str(_first_present(item, ("model_filename", "filename", "model_file", "weights"))).strip()
        except Exception as exc:
            raise ValueError(f"Invalid manifest entry #{idx}: {exc}") from exc

        if model_type not in MODELS.AGENT_BY_MODEL:
            raise ValueError(
                f"Manifest entry #{idx} uses unknown model_type '{model_type}'. "
                f"Known values: {sorted(MODELS.AGENT_BY_MODEL.keys())}"
            )

        model_filename_path = Path(model_filename)
        if model_filename_path.name != model_filename or model_filename_path.parent != Path("."):
            raise ValueError(
                f"Manifest entry #{idx} has model_filename='{model_filename}', but model_filename must be a local filename only, not a path."
            )

        entries.append(
            ManifestEntry(
                model_type=model_type,
                scenario_cfg=scenario,
                model_filename=model_filename,
            )
        )

    return entries


def load_manifest(path: str | Path) -> list[ManifestEntry]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Manifest file not found: {path}")

    suffix = path.suffix.lower()
    if suffix == ".json":
        with path.open("r", encoding="utf-8") as f:
            raw = json.load(f)
        return _coerce_manifest_entries(raw)

    if suffix == ".csv":
        with path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            return _coerce_manifest_entries(list(reader))

    text = path.read_text(encoding="utf-8")
    try:
        raw = ast.literal_eval(text)
    except Exception as exc:
        raise ValueError(
            f"Unsupported manifest format for {path.name}. Use .json, .csv, or a Python literal text file."
        ) from exc
    return _coerce_manifest_entries(raw)


# -----------------------------------------------------------------------------
# Output helpers
# -----------------------------------------------------------------------------
def normalize_output_path(output: str | Path) -> Path:
    path = Path(output)
    if path.suffix.lower() not in {".csv", ".xlsx"}:
        path = path.with_suffix(".csv")
    return path


def build_output_matrix(results_by_column: OrderedDict[str, np.ndarray], episodes: int) -> list[list[Any]]:
    headers = ["Episode", *results_by_column.keys()]
    rows: list[list[Any]] = [headers]

    for ep_idx in range(episodes):
        row: list[Any] = [ep_idx + 1]
        for scores in results_by_column.values():
            row.append(float(scores[ep_idx]))
        rows.append(row)

    stats_labels = ["mean", "std", "min", "max"]
    for label in stats_labels:
        row = [label]
        for scores in results_by_column.values():
            if label == "mean":
                value = float(np.mean(scores))
            elif label == "std":
                value = float(np.std(scores))
            elif label == "min":
                value = float(np.min(scores))
            else:
                value = float(np.max(scores))
            row.append(value)
        rows.append(row)

    return rows


def write_output(rows: list[list[Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    suffix = output_path.suffix.lower()

    if suffix == ".csv":
        with output_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerows(rows)
        return

    if suffix == ".xlsx":
        from openpyxl import Workbook

        wb = Workbook()
        ws = wb.active
        ws.title = "model_eval_data"
        for row in rows:
            ws.append(row)
        wb.save(output_path)
        return

    raise ValueError(f"Unsupported output extension: {suffix}")


# -----------------------------------------------------------------------------
# Model/game construction helpers
# -----------------------------------------------------------------------------
def _load_weights_state_dict(path: Path, device: torch.device) -> dict[str, Any]:
    try:
        obj = torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        obj = torch.load(path, map_location=device)

    if isinstance(obj, dict):
        for key in ("state_dict", "network", "q_net", "model_state_dict"):
            value = obj.get(key)
            if isinstance(value, dict):
                return value
        return obj
    raise ValueError(f"Unexpected weights file format at {path}")


def _try_manual_weight_load(agent, model_path: Path, device: torch.device) -> None:
    state_dict = _load_weights_state_dict(model_path, device)

    if hasattr(agent, "network"):
        agent.network.load_state_dict(state_dict)
        return
    if hasattr(agent, "q_net"):
        agent.q_net.load_state_dict(state_dict)
        if hasattr(agent, "target_net"):
            try:
                agent.target_net.load_state_dict(state_dict)
            except Exception:
                pass
        return
    if hasattr(agent, "load_state_dict"):
        agent.load_state_dict(state_dict)
        return

    raise RuntimeError(
        f"Do not know how to load weights into agent of type {type(agent).__name__} from {model_path}"
    )


def _set_eval_mode(agent) -> None:
    if hasattr(agent, "set_eval_mode"):
        agent.set_eval_mode()
    if hasattr(agent, "eval"):
        try:
            agent.eval()
        except Exception:
            pass


def _coerce_action_index(action: Any) -> int:
    if isinstance(action, torch.Tensor):
        return int(action.item())
    if isinstance(action, np.ndarray):
        return int(np.asarray(action).item())
    return int(action)


def build_agent(model_type: str, agent_builder, model_path: Path, game: vzd.DoomGame, actions: list[list[int]]):
    factorized_mapper_by_model = getattr(MODELS, "FACTORIZED_ACTION_MAPPER", {})

    if model_type in factorized_mapper_by_model:
        mapper_cls = factorized_mapper_by_model[model_type]
        mapper = mapper_cls(game)

        constructor_attempts = (
            lambda: agent_builder(action_mapper=mapper, load_model_path=model_path),
            lambda: agent_builder(action_mapper=mapper, load_model_path=str(model_path)),
            lambda: agent_builder(action_mapper=mapper),
        )

        last_error: Optional[Exception] = None
        for attempt in constructor_attempts:
            try:
                agent = attempt()
                if not model_path.exists():
                    raise FileNotFoundError(f"Model weights not found: {model_path}")
                # If constructor did not consume load_model_path, manual load is still safe.
                try:
                    _try_manual_weight_load(agent, model_path, DEVICE)
                except Exception:
                    # Fine if constructor already loaded and a second load is not needed.
                    pass
                _set_eval_mode(agent)
                return agent
            except TypeError as exc:
                last_error = exc
                continue
        raise RuntimeError(
            f"Failed to construct factorized PPO agent for {model_type}. Last error: {last_error}"
        )

    if model_type in MODELS.PPO_MODELS:
        agent = agent_builder(action_size=len(actions), load_model_path=model_path)
        _set_eval_mode(agent)
        return agent

    if model_type == "q_rainbow_stacked":
        agent = agent_builder(
            action_size=len(actions),
            lr=learning_rate,
            discount_factor=discount_factor,
            memory_size=replay_memory_size,
            batch_size=batch_size,
        )
        state_dict = _load_weights_state_dict(model_path, DEVICE)
        agent.q_net.load_state_dict(state_dict)
        _set_eval_mode(agent)
        return agent

    agent = agent_builder(
        len(actions),
        lr=learning_rate,
        batch_size=batch_size,
        memory_size=replay_memory_size,
        discount_factor=discount_factor,
        load_model=True,
        model_weights=model_path,
    )
    _set_eval_mode(agent)
    return agent


@dataclass
class EvalContext:
    game: vzd.DoomGame
    agent: Any
    actions: list[list[int]]
    resolution: tuple[int, int]
    preprocess_fn: Any
    use_frame_stack: bool
    frame_stack: Any


def build_eval_context(entry: ManifestEntry, visible_window: bool) -> EvalContext:
    model_type = entry.model_type
    model_path = entry.model_path

    if not model_path.exists():
        raise FileNotFoundError(f"Weights file for {model_type} not found: {model_path}")

    agent_builder = MODELS.AGENT_BY_MODEL[model_type]
    config_file_path = os.path.join(SCENARIO_PATH, entry.scenario_cfg)

    game = vzd.DoomGame()
    game.load_config(config_file_path)
    game.set_window_visible(visible_window)
    game.set_mode(vzd.Mode.ASYNC_PLAYER if visible_window else vzd.Mode.PLAYER)
    game.set_screen_resolution(vzd.ScreenResolution.RES_640X480)

    color_mode = MODELS.COLOR_BY_MODEL[model_type]
    if color_mode == MODELS.RGB:
        game.set_screen_format(vzd.ScreenFormat.RGB24)
        preprocess_fn = preprocess_rgb
    elif color_mode == MODELS.GRAYSCALE:
        game.set_screen_format(vzd.ScreenFormat.GRAY8)
        preprocess_fn = preprocess
    elif color_mode == MODELS.AUTO and model_type == "q_rainbow_stacked":
        if rainbow_lazy_mod.USE_GRAYSCALE:
            game.set_screen_format(vzd.ScreenFormat.GRAY8)
        else:
            game.set_screen_format(vzd.ScreenFormat.RGB24)
        preprocess_fn = None
    else:
        raise ValueError(f"Invalid color format for model type {model_type}")

    if hasattr(game, "set_render_hud"):
        game.set_render_hud(True)

    resolution = MODELS.RESOLUTION_BY_MODEL[model_type]
    game.init()

    n = game.get_available_buttons_size()
    actions = [list(a) for a in it.product([0, 1], repeat=n)]
    if model_type == "ppo_late_fusion_rgb_corridor":
        actions = ppo_late_fusion_rgb_corridor.get_deadly_corridor_actions()

    agent = build_agent(model_type, agent_builder, model_path, game, actions)

    use_frame_stack = model_type in MODELS.FRAME_STACK_MODELS
    if use_frame_stack:
        if model_type in MODELS.LAZY_STACK_MODULE_BY_MODEL:
            frame_stack = rainbow_lazy_mod.FrameStack(
                rainbow_lazy_mod.FRAME_STACK_SIZE,
                rainbow_lazy_mod.FRAME_C,
                resolution,
            )
        else:
            frame_stack_cls = MODELS.FRAME_STACK_MODELS[model_type]
            frame_stack = frame_stack_cls(MODELS.FRAME_STACK_SIZE[model_type], resolution)
    else:
        frame_stack = None

    return EvalContext(
        game=game,
        agent=agent,
        actions=actions,
        resolution=resolution,
        preprocess_fn=preprocess_fn,
        use_frame_stack=use_frame_stack,
        frame_stack=frame_stack,
    )


# -----------------------------------------------------------------------------
# Evaluation logic
# -----------------------------------------------------------------------------
def preprocess_state_vars(model_type: str, game_variables, expected_num_vars: int, agent) -> np.ndarray:
    if model_type == "ppo_late_fusion_rgb_corridor":
        return ppo_late_fusion_rgb_corridor.preprocess_vars_corridor(game_variables)

    if model_type == "ppo_film_factorized_gray":
        return ppo_film_factorized_gray.preprocess_vars_safe_general(
            game_variables,
            ppo_film_factorized_gray.NUM_VARS,
            normalizer=getattr(agent, "vars_rms", None),
            update=False,
            clip=5.0,
        )

    if model_type == "ppo_film_factorized_gray_cig" and ppo_film_factorized_gray_cig is not None:
        return ppo_film_factorized_gray_cig.preprocess_vars_safe_general(
            game_variables,
            ppo_film_factorized_gray_cig.NUM_VARS,
            normalizer=getattr(agent, "vars_rms", None),
            update=False,
            clip=5.0,
        )

    return preprocess_vars_safe(game_variables, expected_num_vars)


@torch.no_grad()
def evaluate_model(entry: ManifestEntry, ctx: EvalContext, episodes: int, visible_window: bool) -> np.ndarray:
    model_type = entry.model_type
    game = ctx.game
    agent = ctx.agent
    actions = ctx.actions
    resolution = ctx.resolution
    preprocess_fn = ctx.preprocess_fn
    use_frame_stack = ctx.use_frame_stack
    frame_stack = ctx.frame_stack

    _set_eval_mode(agent)
    scores: list[float] = []
    expected_num_vars = infer_expected_num_vars(agent, game)

    for ep in range(episodes):
        game.new_episode()

        if use_frame_stack:
            if model_type == "q_rainbow_stacked":
                frame_stack.frames.clear()
                if hasattr(frame_stack, "_inited"):
                    frame_stack._inited = False
            else:
                frame_stack.reset()

        while not game.is_episode_finished():
            gs = game.get_state()
            if gs is None:
                break

            if model_type == "q_rainbow_stacked":
                frame_u8 = rainbow_lazy_mod.preprocess_frame_u8(gs.screen_buffer)
                if use_frame_stack:
                    if not getattr(frame_stack, "_inited", False):
                        frame_stack.reset(frame_u8)
                        frame_stack._inited = True
                    else:
                        frame_stack.append(frame_u8)
                    state_img = frame_stack.get()
                else:
                    state_img = frame_u8
            else:
                state_img = preprocess_fn(gs.screen_buffer, resolution)
                if use_frame_stack:
                    frame_stack.push(state_img)
                    state_img = frame_stack.get()

            if model_type in MODELS.PPO_MODELS:
                if model_type in MODELS.PPO_STATE_VAR_MODELS:
                    state_vars = preprocess_state_vars(model_type, gs.game_variables, expected_num_vars, agent)
                    action_idx = agent.get_action(state_img, state_vars, deterministic=True)
                else:
                    action_idx = agent.get_action(state_img, deterministic=True)
            else:
                state_vars = preprocess_state_vars(model_type, gs.game_variables, expected_num_vars, agent)
                try:
                    action_idx = agent.get_action(state_img, state_vars, eval_mode=True)
                except TypeError:
                    action_idx = agent.get_action(state_img, state_vars)

            game.make_action(actions[_coerce_action_index(action_idx)], frame_repeat)

        score = float(game.get_total_reward())
        scores.append(score)

        if (ep + 1) % 10 == 0 or ep == 0 or ep == episodes - 1:
            print(f"[{entry.column_name}] episode {ep + 1}/{episodes}: {score:.2f}")

    return np.asarray(scores, dtype=np.float32)


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main():
    args = parse_cli()

    manifest_entries = load_manifest(args.manifest_path)
    scenario_filter = str(args.scenario).strip()
    if scenario_filter.upper() != "ALL":
        scenario_filter = normalize_scenario_name(scenario_filter)
        manifest_entries = [e for e in manifest_entries if e.scenario_cfg == scenario_filter]

    if not manifest_entries:
        raise ValueError("No manifest entries matched the requested scenario filter.")

    seen_columns: set[str] = set()
    for entry in manifest_entries:
        if entry.column_name in seen_columns:
            raise ValueError(
                f"Duplicate output column detected for {entry.column_name}. "
                f"Each manifest entry must resolve to a unique model filepath."
            )
        seen_columns.add(entry.column_name)

    print(f"Loaded {len(manifest_entries)} manifest entries.")
    print(f"Episodes per model: {args.episodes}")
    print(f"Scenario filter    : {args.scenario}")
    print(f"Visible window     : {args.show}")

    results_by_column: OrderedDict[str, np.ndarray] = OrderedDict()

    for idx, entry in enumerate(manifest_entries, start=1):
        print("=" * 80)
        print(f"[{idx}/{len(manifest_entries)}] Evaluating {entry.model_type}")
        print(f"Scenario : {entry.scenario_cfg}")
        print(f"Weights  : {entry.model_path}")

        ctx: Optional[EvalContext] = None
        try:
            ctx = build_eval_context(entry, visible_window=args.show)
            scores = evaluate_model(entry, ctx, episodes=args.episodes, visible_window=args.show)
            results_by_column[entry.column_name] = scores

            print(
                "Summary : mean {:.2f} +/- {:.2f}, min {:.2f}, max {:.2f}".format(
                    float(scores.mean()),
                    float(scores.std()),
                    float(scores.min()),
                    float(scores.max()),
                )
            )
        finally:
            if ctx is not None:
                try:
                    ctx.game.close()
                except Exception:
                    pass

    output_path = normalize_output_path(args.output)
    rows = build_output_matrix(results_by_column, episodes=args.episodes)
    write_output(rows, output_path)

    print("=" * 80)
    print(f"Wrote results to: {output_path}")


if __name__ == "__main__":
    main()
