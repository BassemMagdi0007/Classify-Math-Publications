import json
import argparse

# Parse command-line arguments
parser = argparse.ArgumentParser(description="Create training data by merging test data with ground truth labels.")
parser.add_argument("test_data_path", type=str, help="Path to the example-test-data.jsonl file")
parser.add_argument("test_results_path", type=str, help="Path to the example-test-results.json file")
parser.add_argument("--output", type=str, default="training_data.jsonl", help="Path to save the output training data file")
args = parser.parse_args()

# Load test data
with open(args.test_data_path, "r") as f:
    papers = [json.loads(line) for line in f]

# Load ground truth labels
with open(args.test_results_path, "r") as f:
    labels = json.load(f)

# Merge into training data
training_data = []
for paper in papers:
    paper_id = paper["id"]
    training_data.append({
        "id": paper_id,
        "formulas": paper["formulas"],
        "classification": labels[paper_id]
    })

# Save as new file
with open(args.output, "w") as f:
    for item in training_data:
        f.write(json.dumps(item) + "\n")

print(f"Training data saved to {args.output}")