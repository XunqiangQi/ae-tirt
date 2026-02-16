"""Experiment result persistence and reporting."""

from __future__ import annotations

import csv
import os

from ae_tirt.utils.paths import ensure_dir, get_safe_path


def save_results(metrics, total_time, conditions, output_dir="TIRT_eval", repeat=None, summary_dir=None):
    """Write trait recovery metrics to CSV (overall + per-trait long + per-trait wide)."""
    if not os.path.isabs(output_dir):
        output_dir = get_safe_path(output_dir)
    ensure_dir(output_dir)
    if summary_dir is not None:
        if not os.path.isabs(summary_dir):
            summary_dir = get_safe_path(summary_dir)
        ensure_dir(summary_dir)

    output_file = get_safe_path("evaluation_results.csv", base_dir=output_dir)
    per_trait_file = get_safe_path("evaluation_results_per_trait.csv", base_dir=output_dir)
    per_trait_wide_file = get_safe_path("evaluation_results_per_trait_wide.csv", base_dir=output_dir)

    condition_id = conditions.get("condition_id", "")
    headers = [
        "condition_id",
        "repeat",
        "npersons",
        "ntraits",
        "npairs",
        "cor",
        "nitems_per_block",
        "weight_sign",
        "Trait_rmse",
        "Trait_bias",
        "Trait_cor",
        "Trait_rel",
        "Total_time(seconds)",
    ]
    row = [
        condition_id,
        repeat if repeat is not None else conditions.get("repeat", ""),
        conditions["npersons"],
        conditions["ntraits"],
        conditions["npairs"],
        conditions["cor"],
        conditions["nitems_per_block"],
        conditions.get("weight_sign", ""),
        metrics["traits"]["overall"]["rmse"],
        metrics["traits"]["overall"]["bias"],
        metrics["traits"]["overall"]["cor"],
        metrics["traits"]["overall"]["rel"],
        total_time,
    ]
    file_exists = os.path.isfile(output_file)
    with open(output_file, mode="a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(headers)
        writer.writerow(row)
    print(f"Evaluation results saved to: {output_file}")

    if summary_dir is not None:
        summary_file = get_safe_path("evaluation_results.csv", base_dir=summary_dir)
        summary_exists = os.path.isfile(summary_file)
        with open(summary_file, mode="a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            if not summary_exists:
                writer.writerow(headers)
            writer.writerow(row)

    per_trait_headers = [
        "condition_id",
        "repeat",
        "npersons",
        "ntraits",
        "npairs",
        "cor",
        "nitems_per_block",
        "weight_sign",
        "trait",
        "rmse",
        "bias",
        "cor",
        "rel",
    ]
    per_trait_exists = os.path.isfile(per_trait_file)
    repeat_val = repeat if repeat is not None else conditions.get("repeat", None)
    for trait_name, m in metrics["traits"]["per_trait"].items():
        trait_idx = trait_name.replace("trait_", "")
        per_trait_row = [
            condition_id,
            repeat_val,
            conditions["npersons"],
            conditions["ntraits"],
            conditions["npairs"],
            conditions["cor"],
            conditions["nitems_per_block"],
            conditions.get("weight_sign", ""),
            trait_idx,
            m["rmse"],
            m["bias"],
            m["cor"],
            m["rel"],
        ]
        with open(per_trait_file, mode="a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            if not per_trait_exists:
                writer.writerow(per_trait_headers)
                per_trait_exists = True
            writer.writerow(per_trait_row)
    print(f"Per-trait evaluation results saved to: {per_trait_file}")

    if summary_dir is not None:
        summary_per_trait_file = get_safe_path("evaluation_results_per_trait.csv", base_dir=summary_dir)
        summary_per_trait_exists = os.path.isfile(summary_per_trait_file)
        for trait_name, m in metrics["traits"]["per_trait"].items():
            trait_idx = trait_name.replace("trait_", "")
            per_trait_row = [
                condition_id,
                repeat_val,
                conditions["npersons"],
                conditions["ntraits"],
                conditions["npairs"],
                conditions["cor"],
                conditions["nitems_per_block"],
                conditions.get("weight_sign", ""),
                trait_idx,
                m["rmse"],
                m["bias"],
                m["cor"],
                m["rel"],
            ]
            with open(summary_per_trait_file, mode="a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                if not summary_per_trait_exists:
                    writer.writerow(per_trait_headers)
                    summary_per_trait_exists = True
                writer.writerow(per_trait_row)

    ntraits = conditions["ntraits"]
    wide_headers = ["condition_id", "repeat", "npersons", "ntraits", "npairs", "cor", "nitems_per_block", "weight_sign"]
    for k in range(1, ntraits + 1):
        wide_headers += [f"trait{k}_rmse", f"trait{k}_bias", f"trait{k}_cor", f"trait{k}_rel"]
    wide_exists = os.path.isfile(per_trait_wide_file)
    wide_row = [
        condition_id,
        repeat_val,
        conditions["npersons"],
        conditions["ntraits"],
        conditions["npairs"],
        conditions["cor"],
        conditions["nitems_per_block"],
        conditions.get("weight_sign", ""),
    ]
    for k in range(1, ntraits + 1):
        m = metrics["traits"]["per_trait"][f"trait_{k}"]
        wide_row += [m["rmse"], m["bias"], m["cor"], m["rel"]]
    with open(per_trait_wide_file, mode="a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not wide_exists:
            writer.writerow(wide_headers)
        writer.writerow(wide_row)
    print(f"Per-trait (wide) evaluation results saved to: {per_trait_wide_file}")

    if summary_dir is not None:
        summary_wide_file = get_safe_path("evaluation_results_per_trait_wide.csv", base_dir=summary_dir)
        summary_wide_exists = os.path.isfile(summary_wide_file)
        with open(summary_wide_file, mode="a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            if not summary_wide_exists:
                writer.writerow(wide_headers)
            writer.writerow(wide_row)


def save_item_parameter_results(item_metrics, conditions, output_dir="TIRT_eval", repeat=None, summary_dir=None):
    """Write item-parameter recovery metrics (w, b) to CSV."""
    if not os.path.isabs(output_dir):
        output_dir = get_safe_path(output_dir)
    ensure_dir(output_dir)
    if summary_dir is not None:
        if not os.path.isabs(summary_dir):
            summary_dir = get_safe_path(summary_dir)
        ensure_dir(summary_dir)

    tirt_output_file = get_safe_path("tirt_parameter_evaluation.csv", base_dir=output_dir)
    condition_id = conditions.get("condition_id", "")
    tirt_headers = [
        "condition_id",
        "repeat",
        "npersons",
        "ntraits",
        "npairs",
        "cor",
        "nitems_per_block",
        "weight_sign",
        "w_rmse",
        "w_bias",
        "w_cor",
        "b_rmse",
        "b_bias",
        "b_cor",
    ]

    tirt_row = [
        condition_id,
        repeat if repeat is not None else conditions.get("repeat", None),
        conditions["npersons"],
        conditions["ntraits"],
        conditions["npairs"],
        conditions["cor"],
        conditions["nitems_per_block"],
        conditions.get("weight_sign", ""),
    ]

    for param_name in ("w", "b"):
        if param_name in item_metrics["overall"]:
            metrics_data = item_metrics["overall"][param_name]
            tirt_row.extend([metrics_data["rmse"], metrics_data["bias"], metrics_data["cor"]])
        else:
            tirt_row.extend([None, None, None])

    file_exists = os.path.isfile(tirt_output_file)
    with open(tirt_output_file, mode="a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(tirt_headers)
        writer.writerow(tirt_row)
    print(f"TIRT parameter evaluation results saved to: {tirt_output_file}")

    if summary_dir is not None:
        summary_tirt_file = get_safe_path("tirt_parameter_evaluation.csv", base_dir=summary_dir)
        summary_tirt_exists = os.path.isfile(summary_tirt_file)
        with open(summary_tirt_file, mode="a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            if not summary_tirt_exists:
                writer.writerow(tirt_headers)
            writer.writerow(tirt_row)


def print_results(metrics, item_metrics, training_method):
    """Print trait and item-parameter recovery metrics."""
    print(f"\n===== Results for {training_method.replace('_', ' ').title()} Strategy =====")
    print("\n--- Trait Recovery Performance ---")
    print(f"Overall RMSE: {metrics['traits']['overall']['rmse']:.4f}")
    print(f"Overall Bias: {metrics['traits']['overall']['bias']:.4f}")
    print(f"Overall Correlation: {metrics['traits']['overall']['cor']:.4f}")
    print(f"Overall Reliability: {metrics['traits']['overall']['rel']:.4f}")

    print("\nTrait Detailed Metrics:")
    headers = ["Trait", "RMSE", "Bias", "Correlation", "Reliability"]
    row_fmt = "{:<10} {:<10} {:<10} {:<12} {:<12}"
    print(row_fmt.format(*headers))
    print("-" * 60)
    for trait_name, m in metrics["traits"]["per_trait"].items():
        print(row_fmt.format(trait_name, f"{m['rmse']:.4f}", f"{m['bias']:.4f}", f"{m['cor']:.4f}", f"{m['rel']:.4f}"))

    print("\n--- Item Parameter Recovery Performance ---")
    for param, label in (("w", "DM Weight (w)"), ("b", "DM Bias (b)")):
        if param in item_metrics["overall"]:
            m = item_metrics["overall"][param]
            print(f"{label}: RMSE={m['rmse']:.4f}, Bias={m['bias']:.4f}, Cor={m['cor']:.4f}")
    print("-" * 60)
