import json
import glob
from collections import defaultdict, Counter
from pathlib import Path
import statistics

def load_results(path=None):
    """Load the most recent evaluation results file if no path specified."""
    if path is None:
        # Find most recent evaluation results file
        results_files = glob.glob("evaluation_results_*.json")
        if not results_files:
            print("No evaluation results found!")
            return None
        path = max(results_files, key=lambda x: Path(x).stat().st_mtime)
        print(f"Loading: {path}\n")
    
    with open(path, "r") as f:
        return json.load(f)

def summarize_system(system_name, data):
    """Summarize metrics for a single system."""
    results = data.get("results", [])
    total = len(results)
    correct = sum(1 for r in results if r.get("correct", False))

    # Extract additional metrics - handle various formats
    confidences_correct = []
    confidences_incorrect = []
    
    for r in results:
        conf = r.get("confidence", 0)
        # Handle both numeric and string confidence values
        if isinstance(conf, str):
            # Try to convert string confidence to numeric
            conf_map = {"low": 0.33, "medium": 0.66, "high": 0.9}
            try:
                conf_value = float(conf)
            except (ValueError, TypeError):
                conf_value = conf_map.get(conf.lower(), 0.5)
        else:
            conf_value = float(conf) if conf else 0.0
        
        if r.get("correct", False):
            confidences_correct.append(conf_value)
        else:
            confidences_incorrect.append(conf_value)
    
    latencies = [r.get("latency_seconds", 0) for r in results if "latency_seconds" in r]
    tool_calls = [len(r.get("tool_calls", [])) for r in results]
    
    # Accuracy by expected label
    label_totals = Counter(r.get("expected", "unknown") for r in results)
    label_correct = Counter(r.get("expected", "unknown") for r in results if r.get("correct", False))

    # Build confusion matrix - handle both formats
    confusion_matrix = {}
    
    # Try to use the pre-computed confusion matrix from the data if available
    if "confusion_matrix" in data and data["confusion_matrix"]:
        existing_cm = data["confusion_matrix"]
        # Convert from predicted_X_actual_Y format to X->Y format
        label_map = {"bs": "bullshit", "not": "not_bullshit", "amb": "contextually_ambiguous"}
        for key, val in existing_cm.items():
            if key.startswith("predicted_"):
                # Parse: predicted_bs_actual_not -> bullshit->not_bullshit
                parts = key.replace("predicted_", "").split("_actual_")
                if len(parts) == 2:
                    pred_short, actual_short = parts
                    pred_full = label_map.get(pred_short, pred_short)
                    actual_full = label_map.get(actual_short, actual_short)
                    confusion_matrix[f"{actual_full}->{pred_full}"] = val
    
    # If no pre-computed matrix, build from results
    if not confusion_matrix:
        for expected in ["bullshit", "not_bullshit", "contextually_ambiguous"]:
            for predicted in ["bullshit", "not_bullshit", "contextually_ambiguous"]:
                count = sum(1 for r in results 
                           if r.get("expected") == expected and r.get("predicted") == predicted)
                confusion_matrix[f"{expected}->{predicted}"] = count

    # Misclassification patterns (only incorrect predictions)
    misclassifications = Counter(
        (r.get("expected", "unknown"), r.get("predicted", "unknown"))
        for r in results
        if not r.get("correct", False)
    )

    return {
        "system": system_name,
        "accuracy": correct / total if total > 0 else 0,
        "total": total,
        "correct": correct,
        "accuracy_by_label": {
            label: label_correct[label] / label_totals[label] if label_totals[label] > 0 else 0
            for label in label_totals if label != "unknown"
        },
        "confusion_matrix": confusion_matrix,
        "misclassifications": misclassifications,
        "avg_confidence_correct": statistics.mean(confidences_correct) if confidences_correct else 0,
        "avg_confidence_incorrect": statistics.mean(confidences_incorrect) if confidences_incorrect else 0,
        "confidence_calibration": (statistics.mean(confidences_correct) - statistics.mean(confidences_incorrect)) if confidences_correct and confidences_incorrect else 0,
        "avg_latency": statistics.mean(latencies) if latencies else 0,
        "avg_tool_calls": statistics.mean(tool_calls) if tool_calls else 0,
        "label_distribution": {k: v for k, v in dict(label_totals).items() if k != "unknown"},
    }

def print_confusion_matrix(cm, indent="  "):
    """Pretty print confusion matrix."""
    labels = ["bullshit", "not_bullshit", "contextually_ambiguous"]
    short_labels = ["BS", "Not BS", "Ambig"]
    
    print(f"{indent}              Predicted:")
    print(f"{indent}              {'BS':<8} {'Not BS':<8} {'Ambig':<8}")
    print(f"{indent}    Actual:")
    
    for i, actual in enumerate(labels):
        row = [cm.get(f"{actual}->{pred}", 0) for pred in labels]
        print(f"{indent}    {short_labels[i]:<8}  {row[0]:<8} {row[1]:<8} {row[2]:<8}")

def main(path=None):
    data = load_results(path)
    if data is None:
        return

    print("="*80)
    print("EVALUATION RESULTS SUMMARY")
    print("="*80)

    overall_total = 0
    overall_correct = 0
    user_system_summaries = {}
    
    # System-level aggregates across all users
    system_aggregates = defaultdict(lambda: {
        "total": 0,
        "correct": 0,
        "confidences_correct": [],
        "confidences_incorrect": [],
        "latencies": [],
        "tool_calls": [],
        "confusion_matrix": Counter(),
        "label_totals": Counter(),
        "label_correct": Counter(),
    })

    # Process each user
    for user, systems in data.items():
        print(f"\n{'='*80}")
        print(f"USER: {user}")
        print(f"{'='*80}")
        
        user_total = 0
        user_correct = 0

        for system_name, system_data in systems.items():
            summary = summarize_system(system_name, system_data)
            user_system_summaries[(user, system_name)] = summary

            # Print system-level summary for this user
            print(f"\n{'-'*80}")
            print(f"System: {system_name}")
            print(f"{'-'*80}")
            print(f"Accuracy: {summary['accuracy']:.1%} ({summary['correct']}/{summary['total']})")
            print(f"Avg Latency: {summary['avg_latency']:.2f}s")
            print(f"Avg Tool Calls: {summary['avg_tool_calls']:.1f}")
            
            print(f"\nConfidence Analysis:")
            print(f"  Correct predictions:   {summary['avg_confidence_correct']:.3f}")
            print(f"  Incorrect predictions: {summary['avg_confidence_incorrect']:.3f}")
            print(f"  Calibration gap:       {summary['confidence_calibration']:.3f}")
            
            print(f"\nAccuracy by Label:")
            for label, acc in summary["accuracy_by_label"].items():
                total = summary["label_distribution"][label]
                correct = int(acc * total)
                print(f"  {label:<25} {acc:>6.1%}  ({correct}/{total})")
            
            print(f"\nConfusion Matrix:")
            print_confusion_matrix(summary["confusion_matrix"])
            
            if summary["misclassifications"]:
                print(f"\nTop Misclassifications:")
                for (expected, predicted), count in summary["misclassifications"].most_common(5):
                    print(f"  {expected} -> {predicted}: {count}")

            # Accumulate statistics
            overall_total += summary["total"]
            overall_correct += summary["correct"]
            user_total += summary["total"]
            user_correct += summary["correct"]
            
            # Aggregate by system across users
            agg = system_aggregates[system_name]
            agg["total"] += summary["total"]
            agg["correct"] += summary["correct"]
            if summary["avg_confidence_correct"] > 0:
                agg["confidences_correct"].append(summary["avg_confidence_correct"])
            if summary["avg_confidence_incorrect"] > 0:
                agg["confidences_incorrect"].append(summary["avg_confidence_incorrect"])
            if summary["avg_latency"] > 0:
                agg["latencies"].append(summary["avg_latency"])
            agg["tool_calls"].append(summary["avg_tool_calls"])
            
            # Aggregate confusion matrix
            for key, val in summary["confusion_matrix"].items():
                agg["confusion_matrix"][key] += val
            
            # Aggregate label stats
            for label, total in summary["label_distribution"].items():
                agg["label_totals"][label] += total
                correct_count = int(summary["accuracy_by_label"].get(label, 0) * total)
                agg["label_correct"][label] += correct_count

        print(f"\n{'-'*80}")
        print(f"User Overall: {user_correct/user_total:.1%} ({user_correct}/{user_total})")
        print(f"{'-'*80}")

    # OVERALL SUMMARY ACROSS ALL USERS
    print(f"\n\n{'='*80}")
    print("OVERALL SUMMARY ACROSS ALL USERS")
    print(f"{'='*80}")
    print(f"\nTotal Accuracy: {overall_correct/overall_total:.1%} ({overall_correct}/{overall_total})")

    # System comparison
    print(f"\n{'-'*80}")
    print("SYSTEM COMPARISON")
    print(f"{'-'*80}")
    
    system_comparison = []
    for system_name, agg in system_aggregates.items():
        accuracy = agg["correct"] / agg["total"] if agg["total"] > 0 else 0
        avg_latency = statistics.mean(agg["latencies"]) if agg["latencies"] else 0
        avg_tools = statistics.mean(agg["tool_calls"]) if agg["tool_calls"] else 0
        avg_conf_correct = statistics.mean(agg["confidences_correct"]) if agg["confidences_correct"] else 0
        avg_conf_incorrect = statistics.mean(agg["confidences_incorrect"]) if agg["confidences_incorrect"] else 0
        
        # Per-label accuracy
        label_accs = {}
        for label in ["bullshit", "not_bullshit", "contextually_ambiguous"]:
            if agg["label_totals"][label] > 0:
                label_accs[label] = agg["label_correct"][label] / agg["label_totals"][label]
            else:
                label_accs[label] = 0
        
        system_comparison.append({
            "name": system_name,
            "accuracy": accuracy,
            "latency": avg_latency,
            "tools": avg_tools,
            "conf_correct": avg_conf_correct,
            "conf_incorrect": avg_conf_incorrect,
            "label_accs": label_accs,
            "confusion_matrix": dict(agg["confusion_matrix"]),
        })
    
    # Sort by accuracy
    system_comparison.sort(key=lambda x: x["accuracy"], reverse=True)
    
    print(f"\n{'System':<35} {'Accuracy':<12} {'Latency':<12} {'Tools':<10} {'Conf(C)':<10} {'Conf(I)'}")
    print("-" * 100)
    for sys in system_comparison:
        print(f"{sys['name']:<35} {sys['accuracy']:>10.1%}  {sys['latency']:>9.2f}s  {sys['tools']:>8.1f}  {sys['conf_correct']:>8.3f}  {sys['conf_incorrect']:>8.3f}")
    
    # Per-label accuracy comparison
    print(f"\n{'-'*80}")
    print("PER-LABEL ACCURACY COMPARISON")
    print(f"{'-'*80}")
    print(f"\n{'System':<35} {'Bullshit':<12} {'Not BS':<12} {'Ambiguous'}")
    print("-" * 80)
    for sys in system_comparison:
        print(f"{sys['name']:<35} {sys['label_accs']['bullshit']:>10.1%}  {sys['label_accs']['not_bullshit']:>10.1%}  {sys['label_accs']['contextually_ambiguous']:>10.1%}")
    
    # Confusion matrices for each system
    print(f"\n{'-'*80}")
    print("CONFUSION MATRICES BY SYSTEM")
    print(f"{'-'*80}")
    for sys in system_comparison:
        print(f"\n{sys['name']}:")
        print_confusion_matrix(sys['confusion_matrix'])
    
    # INSIGHTS
    print(f"\n\n{'='*80}")
    print("KEY INSIGHTS")
    print(f"{'='*80}")
    
    best_system = system_comparison[0]
    worst_system = system_comparison[-1]
    
    print(f"\n[BEST] Best System: {best_system['name']}")
    print(f"   Accuracy: {best_system['accuracy']:.1%}")
    print(f"   Avg Latency: {best_system['latency']:.2f}s")
    print(f"   Avg Tools: {best_system['tools']:.1f}")
    
    print(f"\n[WORST] Worst System: {worst_system['name']}")
    print(f"   Accuracy: {worst_system['accuracy']:.1%}")
    print(f"   Avg Latency: {worst_system['latency']:.2f}s")
    print(f"   Avg Tools: {worst_system['tools']:.1f}")
    
    print(f"\n[GAP] Performance Gap: {(best_system['accuracy'] - worst_system['accuracy'])*100:.1f} percentage points")
    
    # Identify which system is best at each label
    print(f"\n[LABELS] Best System Per Label:")
    for label in ["bullshit", "not_bullshit", "contextually_ambiguous"]:
        best_for_label = max(system_comparison, key=lambda x: x['label_accs'][label])
        print(f"   {label:<25} {best_for_label['name']:<35} ({best_for_label['label_accs'][label]:.1%})")
    
    # Calibration analysis
    print(f"\n[CALIBRATION] Confidence Calibration (higher is better):")
    for sys in system_comparison:
        calibration = sys['conf_correct'] - sys['conf_incorrect']
        print(f"   {sys['name']:<35} {calibration:>6.3f}")
    
    # Efficiency analysis
    print(f"\n[EFFICIENCY] Efficiency (Accuracy per Tool Call):")
    for sys in system_comparison:
        efficiency = sys['accuracy'] / sys['tools'] if sys['tools'] > 0 else 0
        print(f"   {sys['name']:<35} {efficiency:>6.3f}")
    
    print(f"\n{'='*80}\n")


if __name__ == "__main__":
    import sys
    import io
    
    # Force UTF-8 encoding for output (fixes Windows redirection issues)
    if sys.stdout.encoding != 'utf-8':
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    
    path = sys.argv[1] if len(sys.argv) > 1 else None
    main(path)