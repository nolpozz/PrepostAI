import json
from collections import defaultdict, Counter

def load_results(path="evaluation_results.json"):
    with open(path, "r") as f:
        return json.load(f)

def summarize_system(system_name, data):
    results = data["results"]
    total = len(results)
    correct = sum(1 for r in results if r["correct"])

    # accuracy by expected label
    label_totals = Counter(r["expected"] for r in results)
    label_correct = Counter(r["expected"] for r in results if r["correct"])

    # misclassification mapping: expected → predicted
    confusion = Counter(
        (r["expected"], r["predicted"])
        for r in results
        if not r["correct"]
    )

    return {
        "system": system_name,
        "accuracy": correct / total if total > 0 else 0,
        "total": total,
        "correct": correct,
        "accuracy_by_label": {
            label: label_correct[label] / label_totals[label]
            for label in label_totals
        },
        "confusion": confusion
    }

def main(path="evaluation_results.json"):
    data = load_results(path)

    overall_total = 0
    overall_correct = 0

    user_system_summaries = {}

    # summary across all users + systems
    grand_confusion = Counter()

    for user, systems in data.items():
        print(f"\n===== USER: {user} =====")
        user_total = 0
        user_correct = 0
        user_confusion = Counter()

        for system, results in systems.items():
            summary = summarize_system(system, results)
            user_system_summaries[(user, system)] = summary

            # print system-level summary
            print(f"\n--- System: {system} ---")
            print(f"Accuracy: {summary['accuracy']:.2f} ({summary['correct']}/{summary['total']})")
            print("Accuracy by label:")
            for label, acc in summary["accuracy_by_label"].items():
                print(f"  {label}: {acc:.2f}")
            
            print("\nMisclassifications:")
            for (expected, predicted), count in summary["confusion"].items():
                print(f"  expected={expected}, predicted={predicted}: {count}")

            # accumulate overall / per-user statistics
            overall_total += summary["total"]
            overall_correct += summary["correct"]
            user_total += summary["total"]
            user_correct += summary["correct"]
            user_confusion.update(summary["confusion"])
            grand_confusion.update(summary["confusion"])

        print(f"\nUser-level accuracy: {user_correct/user_total:.2f} ({user_correct}/{user_total})")
        print("User-level misclassification breakdown:")
        for (exp, pred), count in user_confusion.items():
            print(f"  expected={exp}, predicted={pred}: {count}")

    # global summary
    print("\n===== OVERALL SUMMARY =====")
    print(f"Overall accuracy: {overall_correct/overall_total:.2f} ({overall_correct}/{overall_total})")

    print("\nOverall misclassification patterns:")
    for (expected, predicted), count in grand_confusion.items():
        print(f"  expected={expected}, predicted={pred}: {count}")

    # natural-language summary
    print("\n===== NATURAL LANGUAGE SUMMARY =====")

    # find best/worst systems
    system_acc = [
        (system, summary["accuracy"])
        for (user, system), summary in user_system_summaries.items()
    ]
    system_acc_sorted = sorted(system_acc, key=lambda x: x[1], reverse=True)

    print(f"Best performing system: {system_acc_sorted[0][0]} ({system_acc_sorted[0][1]:.2f})")
    print(f"Worst performing system: {system_acc_sorted[-1][0]} ({system_acc_sorted[-1][1]:.2f})")

    print("\nMost common misclassification types:")
    for (expected, predicted), count in grand_confusion.most_common():
        print(f"  {expected} → {predicted}: {count}")


if __name__ == "__main__":
    main()
