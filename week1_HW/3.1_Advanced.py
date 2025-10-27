import gradio as gr
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama.llms import OllamaLLM

# --- LangChain + Ollama setup ---
template = """Question: What is the capital of {country}?
Answer: Only output the name of the capital city."""

prompt = ChatPromptTemplate.from_template(template)
model = OllamaLLM(model="llama2")

# Build the chain (prompt → model)
chain = prompt | model


# --- Define a function for Gradio ---
def get_capital(country):
    """Takes a country name, runs the chain, and returns the answer."""
    result = chain.invoke({"country": country})
    return result


# --- Create the Gradio interface ---
iface = gr.Interface(
    fn=get_capital,
    inputs=gr.Textbox(label="Enter a country name", placeholder="e.g., Uzbekistan"),
    outputs=gr.Textbox(label="Answer"),
    title="Ollama + LangChain Demo",
    description="Ask about the capital of any country. Uses Ollama with a LangChain prompt.",
    allow_flagging="never"
)

# --- Launch the app ---
if __name__ == "__main__":
    iface.launch()
