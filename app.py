import gradio as gr
from transformers import pipeline

# Load the summarization model
summarizer = pipeline("summarization", model="AventIQ-AI/t5-text-summarizer")

# Define the summarization function
def summarize_text(input_text):
    summary = summarizer(input_text, max_length=150, min_length=30, do_sample=False)
    return summary[0]['summary_text']

# Create the Gradio UI
iface = gr.Interface(
    fn=summarize_text,
    inputs=gr.Textbox(lines=5, placeholder="Enter text to summarize..."),
    outputs="text",
    title="T5 Text Summarizer",
    description="Enter a passage, and the T5 model will generate a concise summary."
)

# Launch the app
iface.launch()