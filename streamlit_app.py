import streamlit as st
from transformers import pipeline

# --- Set up the Streamlit page layout ---
st.set_page_config(
    page_title="Standalone AI Summarizer",
    layout="wide"
)

# --- Use Streamlit's caching to load the model only once ---
# This is crucial for performance!
@st.cache_resource
def load_summarizer_model():
    """Loads the pre-trained summarization model from Hugging Face."""
    st.info("Please wait while the model is being loaded. This happens only once.", icon="⏳")
    # Using a smaller, faster model for this demo
    model_name = "sshleifer/distilbart-cnn-12-6"
    summarizer = pipeline("summarization", model=model_name, framework="pt")
    st.success("Model loaded successfully!", icon="✅")
    return summarizer

# Load the model and hide the progress bar after loading
summarizer = load_summarizer_model()

# --- Create the main UI ---
st.title("📄 AI-Powered Document Summarizer")
st.markdown("Enter any text below and get a concise, AI-generated summary instantly.")

# User input text area
input_text = st.text_area(
    "Paste your text here:",
    height=300,
    placeholder="The sun rose over the rolling hills, casting a golden light...",
    help="Try a news article, a long email, or a research paper abstract."
)

# Button to trigger the summarization
if st.button("Generate Summary", type="primary"):
    if not input_text.strip():
        st.warning("Please enter some text to summarize.")
    else:
        # Check if the text is long enough for the model
        if len(input_text.split()) < 50:
            st.warning("Input text is too short. A good summary requires at least 50 words.")
        else:
            with st.spinner("Generating summary..."):
                try:
                    # Use the loaded model to generate the summary
                    summary_list = summarizer(input_text, max_length=150, min_length=30, do_sample=False)
                    summary = summary_list[0]['summary_text']

                    # Display the output
                    st.success("Summary Generated!")
                    st.markdown(f"**Summary:**\n{summary}")

                    # Optional: Show original text in an expander
                    with st.expander("Show Original Text"):
                        st.write(input_text)
                except Exception as e:
                    st.error(f"An error occurred during summarization: {e}")



