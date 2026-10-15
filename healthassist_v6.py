def previous_extract_medical_condition(question):
    text = str(question).lower().strip()
    text = re.sub('\\s+', ' ', text)
    text = text.rstrip(' ?.!')
    patterns = ['^what are the symptoms of (.+)$', '^what are the signs of (.+)$', '^how is (.+) diagnosed$', '^how is (.+) treated$', '^how can (.+) be prevented$', '^what are the treatments for (.+)$', '^what is the treatment for (.+)$', '^what causes (.+)$', '^what (?:is|are)(?: \\([^)]+\\))? (.+)$']
    for pattern in patterns:
        match = re.match(pattern, text)
        if match:
            condition = match.group(1).strip()
            return condition if condition else None
    return None

def previous_extract_condition(question):
    text = re.sub('\\s+', ' ', str(question).lower().strip())
    text = text.rstrip(' ?.!')
    match = re.match('^how to diagnose (.+)$', text)
    if match:
        return match.group(1).strip()
    return previous_extract_medical_condition(question)

def previous_detect_question_intent(question):
    text = str(question).lower().strip()
    patterns = {'symptoms': ['\\bsymptoms?\\b', '\\bsigns?\\b'], 'diagnosis': ['\\bdiagnos(?:e|ed|is|ing|tic|tics)\\b', '\\btests?\\b', '\\bdetected\\b'], 'treatment': ['\\btreat(?:ment|ments|ed|ing)?\\b', '\\btherapy\\b', '\\bcure\\b'], 'causes': ['\\bcauses?\\b', '\\bwhy does\\b', '\\brisk factors?\\b'], 'prevention': ['\\bprevent(?:ion|ed|ing)?\\b', '\\bavoid\\b'], 'definition': ['\\bwhat is\\b', '\\bwhat are\\b', '\\bdefine\\b']}
    for intent, regex_list in patterns.items():
        if any((re.search(regex, text) for regex in regex_list)):
            return intent
    return 'other'

"""HealthAssist AI v6 prototype functions.
Research use only. Not validated for clinical use.
Retrieval data must be loaded separately.
"""

import re
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def medical_safety_gate(question):
    text = str(question).lower().strip()

    emergency_patterns = [
        r"\b(chest pain|chest pressure)\b",
        r"\b(can't breathe|cannot breathe|difficulty breathing)\b",
        r"\b(stroke symptoms|signs of a stroke)\b",
        r"\b(severe bleeding|unconscious)\b"
    ]

    diagnosis_patterns = [
        r"\bwhat disease do i have\b",
        r"\bdiagnose me\b",
        r"\bwhat is wrong with me\b",
        r"\bdo i have\b",
        r"\bwhat illness do i have\b"
    ]

    medication_patterns = [
        r"\bwhat medicine should i take\b",
        r"\bwhat medication should i take\b",
        r"\bwhich drug should i take\b",
        r"\bprescribe\b"
    ]

    if any(re.search(pattern, text) for pattern in emergency_patterns):
        return {
            "allowed": False,
            "category": "potential_emergency",
            "message": (
                "I cannot assess urgent symptoms or recommend treatment. "
                "If you are experiencing chest pain, serious breathing difficulty, "
                "or other severe symptoms, seek emergency medical care immediately."
            )
        }

    if any(re.search(pattern, text) for pattern in diagnosis_patterns):
        return {
            "allowed": False,
            "category": "personal_diagnosis",
            "message": (
                "I cannot diagnose medical conditions. "
                "A qualified healthcare professional can evaluate your symptoms."
            )
        }

    if any(re.search(pattern, text) for pattern in medication_patterns):
        return {
            "allowed": False,
            "category": "personal_medication",
            "message": (
                "I cannot recommend medication for your personal symptoms. "
                "Please consult a qualified healthcare professional."
            )
        }

    if re.search(r"\bxyz\d+\b", text):
        return {
            "allowed": False,
            "category": "unsupported_condition",
            "message": (
                "I could not verify the condition mentioned in your question. "
                "Please check the condition name."
            )
        }

    return {
        "allowed": True,
        "category": "general_information",
        "message": "Question may proceed to retrieval."
    }


def extract_medical_condition(question):
    text = re.sub(
        r"\s+", " ", str(question).lower().strip()
    )
    text = text.rstrip(" ?.!")

    match = re.match(
        r"^how to prevent (.+)$",
        text
    )

    if match:
        return match.group(1).strip()

    return previous_extract_condition(question)


def detect_question_intent(question):
    text = re.sub(r"\s+", " ", str(question).lower().strip())

    if re.match(r"^how to diagnose\b", text):
        return "diagnosis"

    return previous_detect_question_intent(question)


def check_answer_quality(answer):
    text = str(answer).strip()
    lower_text = text.lower()

    issues = []

    website_phrases = [
        "the human phenotype ontology provides",
        "you can use the medlineplus medical dictionary",
        "the table below includes",
        "click here for more information"
    ]

    for phrase in website_phrases:
        if phrase in lower_text:
            issues.append("Contains website-specific instructions")
            break

    if len(text.split()) < 15:
        issues.append("Answer contains very little information")

    if len(text) < 50:
        issues.append("Answer is unusually short")

    return {
        "passed": len(issues) == 0,
        "issues": issues
    }


def format_medical_answer(answer):
    text = str(answer).strip()

    # Normalize unusual spaces
    text = text.replace("\u00a0", " ")

    # Normalize repeated whitespace
    text = re.sub(r"\s+", " ", text)

    # Add a space after punctuation when clearly missing
    text = re.sub(r"([.!?])([A-Z])", r"\1 \2", text)

    return text


def format_answer_for_display(answer):
    text = format_medical_answer(answer)

    # Separate sentences into paragraphs without rewriting them
    sentences = re.split(r"(?<=[.!?])\s+(?=[A-Z])", text)

    paragraphs = []

    for start in range(0, len(sentences), 3):
        paragraph = " ".join(sentences[start:start + 3])
        paragraphs.append(paragraph)

    return "\n\n".join(paragraphs)


def score_definition_answer(answer, condition):
    text = format_medical_answer(answer).lower()
    condition = condition.lower()

    score = 0

    # Prefer answers beginning with a direct definition
    opening = text[:250]

    if re.search(
        rf"\b{re.escape(condition)}\b\s+(?:is|means|refers to)\b",
        opening
    ):
        score += 5

    # Reward early mention of the requested condition
    if condition in opening:
        score += 2

    # Prefer concise explanatory answers
    word_count = len(text.split())

    if 40 <= word_count <= 300:
        score += 2

    # Penalize answers primarily beginning with other topics
    unrelated_openings = [
        "people with",
        "studies have shown",
        "heart disease and stroke",
        "prediabetes means"
    ]

    if any(opening.startswith(phrase) for phrase in unrelated_openings):
        score -= 3

    return score


def calculate_formatting_penalty(answer):
    text = str(answer).lower()
    penalty = 0
    issues = []

    # Known extraction errors observed in our dataset
    broken_patterns = [
        "fromgenesor",
        "therenin",
        "systemactivity",
        "bodys",
        "weightat"
    ]

    for pattern in broken_patterns:
        if pattern in text:
            penalty += 2
            issues.append(f"Possible broken text: {pattern}")

    # Detect immediately repeated sequences of 4 to 12 words
    words = re.findall(r"\b\w+\b", text)

    for length in range(4, 13):
        for i in range(len(words) - 2 * length + 1):
            first = words[i:i + length]
            second = words[i + length:i + 2 * length]

            if first == second:
                penalty += 1
                issues.append("Repeated word sequence detected")
                break

        if "Repeated word sequence detected" in issues:
            break

    return {
        "penalty": penalty,
        "issues": issues
    }


def extract_explicit_symptom_sentences(answer):
    text = format_medical_answer(answer)

    sentences = re.split(r"(?<=[.!?])\s+", text)

    symptom_patterns = [
        r"\bsymptoms include\b",
        r"\bsigns and symptoms include\b",
        r"\bcommon symptoms are\b",
        r"\bcommon signs include\b"
    ]

    matches = []

    for sentence in sentences:
        if any(
            re.search(pattern, sentence, flags=re.IGNORECASE)
            for pattern in symptom_patterns
        ):
            matches.append(sentence.strip())

    return matches


def extract_condition_symptoms(answer, condition):
    if not condition:
        return []

    sentences = extract_explicit_symptom_sentences(answer)

    condition_pattern = (
        r"(?<!\w)" + re.escape(condition) + r"(?!\w)"
    )

    return [
        sentence
        for sentence in sentences
        if re.search(
            condition_pattern,
            sentence,
            flags=re.IGNORECASE
        )
    ]


def retrieve_healthassist_v4(question, min_similarity=0.3):
    condition = extract_medical_condition(question)
    intent = detect_question_intent(question)

    not_found = {
        "status": "not_found",
        "matched_question": None,
        "answer": "I could not find a sufficiently relevant answer to this question.",
        "similarity": None,
        "source": None
    }

    if condition is None or intent == "other":
        return not_found.copy()

    candidate_indices = [
        i for i, (c, candidate_intent) in enumerate(
            zip(train_conditions, train_intents)
        )
        if c == condition and candidate_intent == intent
    ]

    if not candidate_indices:
        return not_found.copy()

    query_vector = tfidf_vectorizer_v2.transform([question])

    similarities = cosine_similarity(
        query_vector,
        train_tfidf_v2[candidate_indices]
    )[0]

    candidates = []

    for position, index in enumerate(candidate_indices):
        similarity = float(similarities[position])

        if similarity < min_similarity:
            continue

        row = train_df.iloc[index]

        if not check_answer_quality(row["answer"])["passed"]:
            continue

        relevance = (
            score_definition_answer(row["answer"], condition)
            if intent == "definition"
            else 0
        )

        formatting = calculate_formatting_penalty(row["answer"])

        candidates.append({
            "index": index,
            "similarity": similarity,
            "relevance": relevance,
            "formatting_penalty": formatting["penalty"]
        })

    if not candidates:
        return not_found.copy()

    # Definition questions: prioritize relevance.
    # Other questions: prioritize similarity.
    # Use formatting quality to break ties.
    if intent == "definition":
        candidates.sort(
            key=lambda item: (
                -item["relevance"],
                -round(item["similarity"], 4),
                item["formatting_penalty"]
            )
        )
    else:
        candidates.sort(
            key=lambda item: (
                -round(item["similarity"], 4),
                item["formatting_penalty"]
            )
        )

    best = candidates[0]
    row = train_df.iloc[best["index"]]

    return {
        "status": "found",
        "matched_question": row["question"],
        "answer": format_medical_answer(row["answer"]),
        "similarity": best["similarity"],
        "source": row.get("document_url", "Not available"),
        "formatting_penalty": best["formatting_penalty"]
    }


def retrieve_symptom_fallback(question):
    condition = extract_medical_condition(question)
    intent = detect_question_intent(question)

    if not condition or intent != "symptoms":
        return {"status": "not_found"}

    for index, (dataset_condition, dataset_intent) in enumerate(
        zip(train_conditions, train_intents)
    ):
        if dataset_condition != condition:
            continue

        if dataset_intent != "definition":
            continue

        row = train_df.iloc[index]

        if not check_answer_quality(row["answer"])["passed"]:
            continue

        sentences = extract_condition_symptoms(
            row["answer"],
            condition
        )

        if sentences:
            return {
                "status": "found",
                "condition": condition,
                "answer": sentences[0],
                "matched_question": row["question"],
                "source": row.get("document_url", "Not available"),
                "method": "definition_symptom_extraction"
            }

    return {"status": "not_found"}


def retrieve_healthassist_v5(question, min_similarity=0.3):
    # Step 1: Try primary retrieval
    primary_result = retrieve_healthassist_v4(
        question,
        min_similarity=min_similarity
    )

    if primary_result["status"] == "found":
        return {
            **primary_result,
            "method": "primary_retrieval"
        }

    # Step 2: Try restricted symptom fallback
    fallback_result = retrieve_symptom_fallback(question)

    if fallback_result["status"] == "found":
        return fallback_result

    # Step 3: No suitable answer found
    return {
        "status": "not_found",
        "method": "no_suitable_answer"
    }


def extract_malaria_diagnosis():
    results = []

    for index in malaria_indices:
        row = train_df.iloc[index]

        if not check_answer_quality(row["answer"])["passed"]:
            continue

        answer = format_medical_answer(row["answer"])
        sentences = re.split(r"(?<=[.!?])\s+", answer)

        # Narrow rule for the diagnosis statement
        for sentence in sentences:
            if re.search(
                r"\ba blood test can diagnose it\b",
                sentence,
                flags=re.IGNORECASE
            ):
                results.append({
                    "condition": "malaria",
                    "sentence": sentence.strip(),
                    "source": row.get(
                        "document_url", "Not available"
                    ),
                    "original_question": row["question"]
                })

    return results


def experimental_malaria_diagnosis_response():
    results = extract_malaria_diagnosis()

    if not results:
        return {
            "status": "not_found"
        }

    evidence = results[0]

    return {
        "status": "limited_information",
        "condition": "malaria",
        "answer": (
            "The available source states that a blood test "
            "can be used to diagnose malaria. "
            "However, this information is incomplete and "
            "does not explain the diagnostic procedure. "
            "A qualified healthcare professional should "
            "determine the appropriate testing and "
            "interpret the results."
        ),
        "source_statement": evidence["sentence"],
        "source": evidence["source"],
        "method": "restricted_diagnosis_extraction"
    }


def retrieve_healthassist_v6(question, min_similarity=0.3):
    # First, try the existing v5 system
    result = retrieve_healthassist_v5(
        question,
        min_similarity=min_similarity
    )

    if result["status"] == "found":
        return result

    # Restricted fallback: malaria diagnosis only
    condition = extract_medical_condition(question)
    intent = detect_question_intent(question)

    if condition == "malaria" and intent == "diagnosis":
        return experimental_malaria_diagnosis_response()

    return {
        "status": "not_found",
        "method": "no_suitable_answer"
    }


def healthassist_answer(question):
    question = str(question).strip()

    if not question:
        return {
            "status": "invalid",
            "answer": "Please enter a medical question.",
            "source": None,
            "matched_question": None
        }

    # Always check safety before retrieval
    safety = medical_safety_gate(question)

    if not safety["allowed"]:
        return {
            "status": "blocked",
            "answer": safety["message"],
            "source": None,
            "matched_question": None
        }

    # Use the updated retrieval system
    result = retrieve_healthassist_v6(question)

    if result["status"] == "found":
        return {
            "status": "found",
            "answer": format_medical_answer(result["answer"]),
            "source": result.get("source"),
            "matched_question": result.get("matched_question")
        }

    if result["status"] == "limited_information":
        return {
            "status": "limited_information",
            "answer": result["answer"],
            "source": result.get("source"),
            "matched_question": result.get("matched_question")
        }

    return {
        "status": "not_found",
        "answer": (
            "I could not find a sufficiently relevant answer "
            "in my available medical information sources. "
            "Please consult a qualified healthcare professional "
            "for medical advice."
        ),
        "source": None,
        "matched_question": None
    }


def gradio_healthassist(question):
    result = healthassist_answer(question)
    status = result["status"]

    if status == "found":
        answer = format_answer_for_display(result["answer"])

        details = (
            f"Status: Retrieved information\n"
            f"Matched question: {result.get('matched_question') or 'Not available'}\n"
            f"Source: {result.get('source') or 'Not available'}"
        )

    elif status == "limited_information":
        answer = (
            "LIMITED INFORMATION — NOT A COMPLETE MEDICAL ANSWER\n\n"
            + result["answer"]
        )

        details = (
            f"Status: Limited information\n"
            f"Source: {result.get('source') or 'Not available'}"
        )

    else:
        answer = result["answer"]
        details = f"Status: {status}\nNo source available."

    return answer, details

