import os
import glob
from forensics_detective import SimpleDetective
import rules
import rules_v2

def run_test(detective, folder_path, output_file, use_v2=False):
    rule_module = rules_v2 if use_v2 else rules
    
    images = glob.glob(os.path.join(folder_path, "*"))
    images.sort()
    
    results = []
    correct = 0
    total = 0
    
    with open(output_file, 'w') as f:
        for img_path in images:
            filename = os.path.basename(img_path)
            f.write(f"Processing: {filename}\n")
            
            match, score, evidence = detective.find_best_match(img_path, use_v2=use_v2, rule_module=rule_module)
            
            total_possible = sum(e[3] for e in evidence)
            
            for name, pts, ev, possible in evidence:
                status = "FIRED" if pts > 0 else "NO MATCH"
                f.write(f"Rule {evidence.index((name, pts, ev, possible))+1} ({name}): {status} - {ev} -> {pts}/{possible} points\n")
            
            match_str = match if match == "REJECTED" else f"MATCH to {match}"
            f.write(f"Final Score: {score}/{total_possible} -> {match_str}\n\n")
            
            # Simple accuracy check for console output (optional but helpful)
            # Assuming ground truth based on filename as per README
            if "random" not in folder_path:
                total += 1
                # modified_03... -> original_03.jpg
                expected = filename.split('_')[1] # e.g. "03"
                if match != "REJECTED" and expected in match:
                    correct += 1
            elif match == "REJECTED":
                correct += 1
                total += 1
            else:
                total += 1

    accuracy = (correct / total) * 100 if total > 0 else 0
    return accuracy

def main():
    detective = SimpleDetective()
    detective.register_targets("originals")
    
    # Phase 1: Easy + Random
    print("Running Phase 1 (Easy Cases)...")
    acc_easy = run_test(detective, "modified_images", "results_v1_easy.txt", use_v2=False)
    print("Running Phase 1 (Random Cases)...")
    acc_random = run_test(detective, "random", "results_v1_random.txt", use_v2=False)
    
    # Merge for results_v1.txt
    with open("results_v1.txt", "w") as outfile:
        for f in ["results_v1_easy.txt", "results_v1_random.txt"]:
            with open(f, "r") as infile:
                outfile.write(infile.read())
    
    # Clean up temp files
    os.remove("results_v1_easy.txt")
    os.remove("results_v1_random.txt")

    # Phase 1: Hard
    print("Running Phase 1 (Hard Cases)...")
    run_test(detective, "hard", "results_v1_hard.txt", use_v2=False)
    
    # Phase 2: All
    print("Running Phase 2 (All Cases)...")
    run_test(detective, "modified_images", "results_v2_easy.txt", use_v2=True)
    run_test(detective, "hard", "results_v2_hard.txt", use_v2=True)
    run_test(detective, "random", "results_v2_random.txt", use_v2=True)
    
    with open("results_v2.txt", "w") as outfile:
        for f in ["results_v2_easy.txt", "results_v2_hard.txt", "results_v2_random.txt"]:
            with open(f, "r") as infile:
                outfile.write(infile.read())

    # Clean up temp files
    os.remove("results_v2_easy.txt")
    os.remove("results_v2_hard.txt")
    os.remove("results_v2_random.txt")
    
    print("\nProcessing complete. Results saved to results_v1.txt, results_v1_hard.txt, and results_v2.txt")

if __name__ == "__main__":
    main()
