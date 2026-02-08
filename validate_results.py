import os
import json
from datetime import datetime

# CHANGE THIS if your matcher file name is not rule_based.py
from rule_based_expert_system import ForensicExpertSystem


def normalize_pred(raw_result):
    """
    Ground truth uses:
      - "original_00" (no extension) for real matches
      - None for random images (means should be rejected)

    Your matcher returns:
      - "original_00.jpg" for a match
      - "REJECTED" for no match
    """
    if raw_result is None:
        return None
    if raw_result == "REJECTED":
        return None
    # convert "original_02.jpg" -> "original_02"
    return os.path.splitext(os.path.basename(raw_result))[0]


def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))

    gt_path = os.path.join(base_dir, "ground_truth.json")
    originals_path = os.path.join(base_dir, "originals")

    if not os.path.exists(gt_path):
        print(f"❌ ground_truth.json not found at: {gt_path}")
        return
    if not os.path.exists(originals_path):
        print(f"❌ originals folder not found at: {originals_path}")
        return

    # Load ground truth
    with open(gt_path, "r", encoding="utf-8") as f:
        ground_truth = json.load(f)

    # Init expert system
    expert = ForensicExpertSystem(originals_path)

    # Where to save outputs
    out_predictions_path = os.path.join(base_dir, "predictions.json")
    out_comparison_path = os.path.join(base_dir, "comparison.json")

    predictions = {}   # file -> {expected, raw_pred, pred, reason, correct}
    results_list = []  # same data but as list (easier to inspect)
    summary = {
        "overall": {"total": 0, "correct": 0},
        "by_category": {}
    }

    def bump(cat, ok):
        if cat not in summary["by_category"]:
            summary["by_category"][cat] = {"total": 0, "correct": 0}
        summary["by_category"][cat]["total"] += 1
        summary["overall"]["total"] += 1
        if ok:
            summary["by_category"][cat]["correct"] += 1
            summary["overall"]["correct"] += 1

    # Evaluate every file in ground_truth.json
    for rel_path, expected in ground_truth.items():
        # category is first folder (hard/ modified_images/ random/)
        category = rel_path.split("/")[0] if "/" in rel_path else "unknown"

        # Build OS-safe full path
        full_path = os.path.join(base_dir, *rel_path.split("/"))

        raw_pred, reason = expert.analyze_image(full_path)
        pred = normalize_pred(raw_pred)

        # expected in ground_truth is "original_XX" or None
        correct = (pred == expected)

        bump(category, correct)

        record = {
            "file": rel_path,
            "category": category,
            "expected": expected,          # e.g. "original_03" or None
            "raw_predicted": raw_pred,     # e.g. "original_03.jpg" or "REJECTED"
            "predicted": pred,             # normalized: "original_03" or None
            "correct": correct,
            "reason": reason
        }

        predictions[rel_path] = record
        results_list.append(record)

        # Console output (✅ correct, ❌ wrong)
        icon = "✅" if correct else "❌"
        print(f"{icon} {rel_path} -> raw={raw_pred} | expected={expected}")

    # Add final accuracy % in summary
    def acc(correct, total):
        return 0.0 if total == 0 else (correct / total) * 100

    summary["overall"]["accuracy_percent"] = round(
        acc(summary["overall"]["correct"], summary["overall"]["total"]), 2
    )
    for cat, s in summary["by_category"].items():
        s["accuracy_percent"] = round(acc(s["correct"], s["total"]), 2)

    # Write predictions.json (all outputs)
    with open(out_predictions_path, "w", encoding="utf-8") as f:
        json.dump({
            "generated_at": datetime.now().isoformat(),
            "predictions": predictions
        }, f, indent=2)

    # Write comparison.json (expected vs predicted + summary)
    with open(out_comparison_path, "w", encoding="utf-8") as f:
        json.dump({
            "generated_at": datetime.now().isoformat(),
            "summary": summary,
            "results": results_list
        }, f, indent=2)

    print("\n--- SUMMARY ---")
    print(f"Overall: {summary['overall']['correct']}/{summary['overall']['total']} "
          f"({summary['overall']['accuracy_percent']}%)")
    for cat, s in summary["by_category"].items():
        print(f"{cat}: {s['correct']}/{s['total']} ({s['accuracy_percent']}%)")

    print("\n✅ Saved:")
    print(f" - {out_predictions_path}")
    print(f" - {out_comparison_path}")


if __name__ == "__main__":
    main()
