import os
import json
import hashlib
import pandas as pd
from datasets import load_dataset

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MANIFEST_PATH = os.path.join(BASE_DIR, "training_manifest.json")
OUTPUT_PATH = os.path.join(BASE_DIR, "retrieval_data.csv")


def fingerprint(question, answer, url):
    content = json.dumps(
        [question, answer, url],
        ensure_ascii=False
    )
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def main():
    with open(MANIFEST_PATH, "r", encoding="utf-8") as file:
        manifest = json.load(file)

    expected = manifest["ordered_fingerprints"]

    print("Downloading MedQuAD...")
    dataset = load_dataset("lavita/MedQuAD", split="train")

    source = dataset.to_pandas()
    lookup = {}

    for question, answer, url in source[
        ["question", "answer", "document_url"]
    ].itertuples(index=False, name=None):

        original = (question, answer, url)

        lookup[fingerprint(*original)] = original

        if isinstance(answer, str):
            cleaned_answer = answer.rstrip()

            cleaned = (question, cleaned_answer, url)

            lookup[fingerprint(*cleaned)] = cleaned

    reconstructed = []
    missing = []

    for position, record_hash in enumerate(expected):
        if record_hash not in lookup:
            missing.append(position)
        else:
            reconstructed.append(lookup[record_hash])

    if missing:
        raise RuntimeError(
            f"Reconstruction failed: {len(missing)} records missing. "
            f"First missing positions: {missing[:10]}"
        )

    result = pd.DataFrame(
        reconstructed,
        columns=["question", "answer", "document_url"]
    )

    # Verify exact record order and contents
    actual_hashes = [
        fingerprint(*row)
        for row in result.itertuples(index=False, name=None)
    ]

    if actual_hashes != expected:
        raise RuntimeError("Fingerprint verification failed.")

    result.to_csv(OUTPUT_PATH, index=False)

    print("\nHEALTHASSIST AI — DATASET RECONSTRUCTION")
    print("=" * 50)
    print("Records:", len(result))
    print("Fingerprint verification: PASS")
    print("Row order verification: PASS")
    print("Saved:", OUTPUT_PATH)


if __name__ == "__main__":
    main()
