import os
import json
from dotenv import load_dotenv
from google import genai

# Load API key
load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

# Connect to Gemini
client = genai.Client(api_key=api_key)

# Load our evidence file
with open("results/consensus/genai_evidence.json", "r") as f:
    evidence = json.load(f)

# Take ONLY the first image for testing
record = evidence[0]

# Remove ground-truth information before sending to Gemini
record_for_gemini = {
    "image": {
        "path": record["image"]["path"],
        "category": record["image"]["category"]
    },
    "models": record["models"],
    "evidence": record["evidence"],
    "consensus": record["consensus"]
}

prompt = f"""
You are an industrial visual inspection assistant.

Your job is to explain the evidence produced by two anomaly
detection models: PatchCore and PaDiM.

IMPORTANT RULES:
- Use ONLY the evidence provided below.
- Do not invent physical causes of defects.
- Do not invent visual details that are not provided.
- Do not interpret anomaly scores as probabilities.
- Do not interpret the consensus score as a probability.
- Do not mention any ground-truth defect label.
- If the models disagree, clearly say that they disagree.
- The explanation should be simple and professional.

Provide:

1. Inspection result
2. Model agreement
3. Evidence summary
4. Consensus assessment
5. Uncertainty assessment
6. A short overall explanation

Evidence:

{json.dumps(record_for_gemini, indent=2)}
"""

interaction = client.interactions.create(
    model="gemini-3.8-flash",
    input=prompt
)

print("\n================ GEMINI RESULT ================\n")
print(interaction.output_text)