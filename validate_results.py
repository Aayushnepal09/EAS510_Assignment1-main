import os
import json
from datetime import datetime
from expert_system import ForensicSystem


def normalize_pred(raw_result):
    """
    Comparing results with ground truth 
    """
    if raw_result is None:
        return None
    if raw_result == "REJECTED":
        return None
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

    # Loading ground truth
    with open (gt_path, "r") as f:
        ground_truth = json.load(f)
        
    expert = ForensicSystem(originals_path)

    # saving output fro both result and compae values 
    out_predictions_path = os.path.join(base_dir, "predictions.json")
    out_comparison_path = os.path.join(base_dir, "comparison.json")

    predictions = {}  
    # same values as prediction but as list fro easy comparisiom values are :{expected, raw_pred, pred, reason, correct}
    results_list = [] 
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

    # evaluate every file in ground_truth.json
    for rel_path, expected in ground_truth.items():

        # category for folders (hard/modified_images/ random/)
        category = rel_path.split("/")[0] if "/" in rel_path else "unknown"

        #  OS-safe ful path
        full_path = os.path.join(base_dir, *rel_path.split("/"))

        raw_pred, reason = expert.analyze_image(full_path)
        pred = normalize_pred(raw_pred)

        # comparing 
        correct = (pred == expected)

        bump(category, correct)

        record = {
            "file": rel_path,
            "category": category,
            "expected": expected,          
            "raw_predicted": raw_pred,     
            "predicted": pred,            
            "correct": correct,
            "reason": reason
        }

        predictions[rel_path] = record
        results_list.append(record)

        # for console output
        icon = "✅" if correct else "❌"
        print(f"{icon} {rel_path} -> raw={raw_pred} | expected={expected}")

    # adding overall final results 
    def acc(correct, total):
        return 0.0 if total == 0 else (correct / total) * 100

    summary["overall"]["accuracy_percent"] = round(
        acc(summary["overall"]["correct"], summary["overall"]["total"]), 2
    )
    for cat, s in summary["by_category"].items():
        s["accuracy_percent"] = round(acc(s["correct"], s["total"]), 2)

    # saving result of values 
    with open(out_predictions_path, "w", encoding="utf-8") as f:
        json.dump({
            "generated_at": datetime.now().isoformat(),
            "predictions": predictions
        }, f, indent=2)

    # saving results and compared values 
    with open(out_comparison_path, "w", encoding="utf-8") as f:
        json.dump({
            "generated_at": datetime.now().isoformat(),
            "summary": summary,
            "results": results_list
        }, f, indent=2)

    #showing summary in cnsole 
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
