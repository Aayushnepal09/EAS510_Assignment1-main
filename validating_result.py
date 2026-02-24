import json
import os
from collections import defaultdict

def compute_accuracy(results_file, ground_truth):
    if not os.path.exists(results_file):
        print(f"File not found: {results_file}")
        return


    
    with open(results_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    total_images = 0
    correct_predictions = 0

    category_correct = defaultdict(int)
    category_total = defaultdict(int)

    current_image = None
    


    for line in lines:
        line = line.strip()
        if line.startswith("Processing:"):
            current_image = line.split("Processing: ")[1].strip()
        

        elif line.startswith("Final Score:") and current_image:
            total_images += 1
            prediction_part = line.split("->")[1].strip()
            
            # Find the true actual label and original path key for category
            actual_label = None
            original_path = ""
            for k, v in ground_truth.items():
                if os.path.basename(k) == current_image:
                    actual_label = v
                    original_path = k
                    break

            category = original_path.split('/')[0] if '/' in original_path else "unknown"
            category_total[category] += 1

            raw_prediction = "REJECTED"
            predicted_label = None

            if prediction_part == "REJECTED":
                raw_prediction = "REJECTED"
                predicted_label = None
            elif "MATCH to" in prediction_part:
                raw_prediction = prediction_part.replace("MATCH to ", "").strip()
                predicted_label = raw_prediction.replace(".jpg", "")

            if predicted_label == actual_label:
                correct_predictions += 1
                category_correct[category] += 1
                print(f"✅ {original_path} -> raw={raw_prediction} | expected={actual_label}")
            else:
                print(f"❌ {original_path} -> raw={raw_prediction} | expected={actual_label}")
            
            current_image = None 
    print("\n--- SUMMARY ---")
    print(f"Overall: {total_images}/{total_images} (100.0%)" if correct_predictions == total_images else f"Overall: {correct_predictions}/{total_images} ({(correct_predictions/total_images)*100:.1f}%)")
    print(f"Total Correct: {correct_predictions}")
    print(f"Total Images: {total_images}")
    
    for cat in sorted(category_total.keys()):
        c = category_correct[cat]
        t = category_total[cat]
        pct = (c/t)*100 if t > 0 else 0
        print(f"{cat}: {c}/{t} ({pct:.1f}%)")
    print("-" * 50)
    

def main():
    try:
        with open('ground_truth.json', 'r', encoding='utf-8') as f:
            gt = json.load(f)
    except Exception as e:
        print("Could not load ground truth JSON:", e)
        return

    print("--- Evaluating System Accuracy vs Ground Truth ---")
    print(f"\nEvaluating: results_v1.txt")
    compute_accuracy('results_v1.txt', gt)
    print(f"\nEvaluating: results_v1_hard.txt")
    compute_accuracy('results_v1_hard.txt', gt)
    print(f"\nEvaluating: results_v2.txt")
    compute_accuracy('results_v2.txt', gt)

if __name__ == "__main__":
    main()
