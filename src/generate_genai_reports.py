import os
import json
import time
from dotenv import load_dotenv
from google import genai

# --------------------------------------------------
# 1. Load API key
# --------------------------------------------------

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise ValueError("GEMINI_API_KEY not found in .env")

client = genai.Client(api_key=api_key)


# --------------------------------------------------
# 2. Load evidence
# --------------------------------------------------

input_file = "results/consensus/genai_evidence.json"
output_file = "results/consensus/genai_reports.json"

with open(input_file, "r") as f:
    evidence_data = json.load(f)

print(f"Loaded {len(evidence_data)} evidence records.")


# --------------------------------------------------
# 3. Generate one explanation
# --------------------------------------------------

def generate_explanation(record):

    # IMPORTANT:
    # We deliberately do NOT send defect_type to Gemini.
    evidence_for_gemini = {
        "image": {
            "path": record["image"]["path"],
            "category": record["image"]["category"]
        },
        "models": record["models"],
        "evidence": record["evidence"],
        "consensus": record["consensus"]
    }

    prompt = f"""
You are an AI assistant for an industrial visual inspection system.

Your task is to explain the evidence produced by two anomaly
detection models: PatchCore and PaDiM.

Use ONLY the evidence provided below.

IMPORTANT RULES:

1. Do not invent visual details.
2. Do not invent physical causes of an anomaly.
3. Do not use or infer any ground-truth defect label.
4. Do not interpret anomaly scores as probabilities.
5. Do not interpret the consensus score as a probability.
6. The uncertainty score is an evidence-consistency measure,
   not a probability of being wrong.
7. If both models predict the same result, state that they agree.
8. If their predictions differ, clearly state that they disagree.
9. Discuss heatmap IoU as the degree of overlap between the
   two models' anomaly localization maps.
10. Keep the explanation concise and professional.
11. Refer to the result as an evidence-based visual assessment.

Return the following sections:

1. Inspection Result
2. Model Agreement
3. Evidence Summary
4. Consensus Assessment
5. Evidence Consistency
6. Overall Explanation

Evidence:

{json.dumps(evidence_for_gemini, indent=2)}
"""

    response = client.interactions.create(
        model="gemini-3.8-flash",
        input=prompt
    )

    return response.output_text


# --------------------------------------------------
# 4. Process all records
# --------------------------------------------------

reports = []

for i, record in enumerate(evidence_data[:3]):

    print(f"\nProcessing {i + 1}/{len(evidence_data)}")
    print(f"Image: {record['image']['path']}")

    try:

        explanation = generate_explanation(record)

        report = {
            "image": record["image"],
            "models": record["models"],
            "evidence": record["evidence"],
            "consensus": record["consensus"],
            "genai_explanation": explanation
        }

        reports.append(report)

        print("✓ Generated successfully")

    except Exception as e:

        print(f"✗ Error: {e}")

        report = {
            "image": record["image"],
            "models": record["models"],
            "evidence": record["evidence"],
            "consensus": record["consensus"],
            "genai_explanation": None,
            "error": str(e)
        }

        reports.append(report)

    # Small delay between requests
    time.sleep(1)


# --------------------------------------------------
# 5. Save results
# --------------------------------------------------

with open(output_file, "w") as f:
    json.dump(reports, f, indent=4)

print("\n==========================================")
print("GENAI PROCESSING COMPLETE")
print("==========================================")
print(f"Total records: {len(reports)}")
print(f"Output saved to: {output_file}")