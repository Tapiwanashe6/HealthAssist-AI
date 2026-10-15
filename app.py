
import os
import importlib.util
import joblib
import pandas as pd
import gradio as gr

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Load HealthAssist AI functions
module_path = os.path.join(BASE_DIR, "healthassist_v6.py")
spec = importlib.util.spec_from_file_location("healthassist_v6", module_path)
healthassist = importlib.util.module_from_spec(spec)
spec.loader.exec_module(healthassist)

# Load saved retrieval components
healthassist.tfidf_vectorizer_v2 = joblib.load(
    os.path.join(BASE_DIR, "tfidf_vectorizer.joblib")
)
healthassist.train_tfidf_v2 = joblib.load(
    os.path.join(BASE_DIR, "train_tfidf_matrix.joblib")
)
healthassist.train_conditions = joblib.load(
    os.path.join(BASE_DIR, "train_conditions.joblib")
)
healthassist.train_intents = joblib.load(
    os.path.join(BASE_DIR, "train_intents.joblib")
)
healthassist.train_df = pd.read_csv(
    os.path.join(BASE_DIR, "retrieval_data.csv")
)

healthassist.malaria_indices = [
    i for i, (condition, intent) in enumerate(
        zip(healthassist.train_conditions, healthassist.train_intents)
    )
    if condition == "malaria" and intent == "definition"
]

# Build Gradio application
with gr.Blocks(title="HealthAssist AI v6") as demo:
    gr.Markdown("# HealthAssist AI")
    gr.Markdown(
        "### What can you ask?\\n"
        "HealthAssist AI provides general educational information "
        "about medical conditions when suitable sources are available.\\n\\n"
        "**Supported question types:**\\n"
        "- What is a medical condition?\\n"
        "- What are its symptoms?\\n"
        "- How is it diagnosed?\\n"
        "- What are its general treatments or prevention methods?\\n\\n"
        "**Examples:** What is diabetes? | "
        "What are the symptoms of asthma? | "
        "How is malaria diagnosed?\\n\\n"
        "**Not supported:** Entering isolated symptoms such as "
        "'fever' or 'stomach ache' to obtain a diagnosis, "
        "personal treatment recommendations, or emergency assessment.\\n\\n"
        "**Note:** Not every condition or question has a suitable "
        "answer in the available dataset."
    )

    question = gr.Textbox(
        label="Your Medical Question",
        placeholder="Example: What is diabetes?",
        lines=2
    )

    submit = gr.Button("Ask HealthAssist")

    answer = gr.Textbox(
        label="HealthAssist AI Answer",
        lines=8,
        interactive=False
    )

    source = gr.Textbox(
        label="Answer Status and Source",
        lines=4,
        interactive=False
    )

    submit.click(
        fn=healthassist.gradio_healthassist,
        inputs=question,
        outputs=[answer, source]
    )

    gr.Examples(
        examples=[
            ["What is diabetes?"],
            ["What are the symptoms of asthma?"],
            ["How is pneumonia diagnosed?"],
            ["How is malaria diagnosed?"]
        ],
        inputs=question
    )

    gr.Markdown(
        "**Medical disclaimer:** HealthAssist AI is an educational "
        "research prototype. Information may be incomplete, outdated, "
        "or incorrect. It cannot diagnose medical conditions, recommend "
        "personal treatment, or reliably identify emergencies. "
        "Consult a qualified healthcare professional for medical "
        "concerns. For emergencies, seek urgent medical care."
    )

if __name__ == "__main__":
    demo.launch()
