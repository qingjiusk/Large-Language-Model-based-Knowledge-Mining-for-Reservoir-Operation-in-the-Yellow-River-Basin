# Task 2 Report: OptimizationFormatter Enhancement

**Status:** DONE

## What was done

Added two new methods to `OptimizationFormatter` in `src/knowledge_graph/optimization_formatter.py`:

### `build_formulation_batch()`
- Consumes the `batch_data` dict from `GraphQuery.get_formulation_batch()` (Task 1)
- Iterates over all reservoirs in batch_data and calls the existing `build_formulation()` for each
- Supports `include_hydrology` and `include_rules` flags to control what data is included
- Returns `{"reservoirs": {id: formulation, ...}, "relations": [...]}`

### `build_solver_ready()`
- Consumes the same `batch_data` dict
- Produces a compact solver-ready format with indexed variables, merged constraints, time series, and objective hints
- Variables get index-suffixed symbols (e.g., `Z_1`, `P_2`) to avoid collisions across reservoirs
- Constraint expressions are rewritten to use indexed variable symbols
- Parameter-based constraints (dead_storage_level, flood_control_level) are automatically added as inequalities
- Supports optional `year` filtering for time series

## Verification results

All 5 test checks pass:
- `build_formulation_batch` with 2 reservoirs produces correct formulations
- `include_hydrology=False` and `include_rules=False` flags work correctly
- `build_solver_ready` produces indexed variables (`Z_1`, `P_2`)
- Merged constraints from both formulations and parameters (5 total: 3 from raw constraints + 2 from parameters)
- Meta correctly reports reservoir count, constraint count, and time_series_years

## Existing methods

No existing methods were modified. `build_formulation()` signature and behavior remain unchanged.

## Post-implementation fixes (2026-07-22)

### Issue 1 (Critical): Constraint expression symbol replacement didn't work

The original `build_solver_ready()` loop at lines 264-277 attempted to replace bare symbols
(e.g., "Z") with indexed ones (e.g., "Z_1") in constraint expressions via `str.replace()`.
But `_build_expression()` uses the Chinese `variable` field (e.g., "水位", "库容水位高程"),
not the symbols "Z", "V", "P". The replacement was a no-op — solver-ready output had
Chinese variable names in constraint expressions that didn't correspond to indexed variables.

**Fix**: Replaced the symbol-replacement loop with a direct expression builder using a
`CATEGORY_TO_SYMBOL` mapping derived from `VARIABLE_INFERENCE`:
- water_level -> Z
- storage -> V
- discharge -> Q_out
- power_output -> P
- power_generation -> E
- water_supply -> W_supply
- water_use -> W_use

New constraint expressions are built as e.g. `Z_1 <= 2594` instead of `水位 <= 2594`.
A fallback to `cons.get("expression")` is kept for constraints with unknown categories
or missing operator/value.

### Issue 2 (Important): Dead code — reservoir_map dict

Lines 239, 242 had `reservoir_map = {}` and `reservoir_map[idx] = rid` — never read.
Removed both lines.

### Minor docstring fixes

- Added `"meta"` key to the `build_solver_ready()` return contract in the docstring
- Changed `year` parameter docstring from "可选，过滤时间序列" to "可选，控制是否包含时间序列数据（非None时包含）"

### Verification

Ran verification script with mock data containing a dead-storage-level constraint — both
constraint expressions correctly use `Z_1` indexed symbols. Assertions pass.
