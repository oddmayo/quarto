from shiny import App, ui, reactive, render
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# LlamaIndex imports
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import Settings, StorageContext, load_index_from_storage
from llama_index.llms.gemini import Gemini
from llama_index.core.prompts import PromptTemplate

# Configure embedding model
embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
Settings.embed_model = embed_model

# Configure LLM
llm = Gemini(
    model="models/gemini-2.5-flash-lite", 
    api_key=os.getenv("GOOGLE_API_KEY")
)
Settings.llm = llm

# Load persisted index
storage_context = StorageContext.from_defaults(persist_dir="data/index_store")
index = load_index_from_storage(storage_context)

# Create query engine with custom prompt
query_engine = index.as_query_engine()

qa_prompt_tmpl_str = (
    "Context information is below.\n"
    "---------------------\n"
    "{context_str}\n"
    "---------------------\n"
    "Given the context above, answer the query in a step-by-step manner. "
    "Include code snippets when relevant. If you don't know the answer, say 'I don't know!'.\n"
    "Query: {query_str}\n"
    "Answer: "
)
qa_prompt_tmpl = PromptTemplate(qa_prompt_tmpl_str)
query_engine.update_prompts({"response_synthesizer:text_qa_template": qa_prompt_tmpl})

# Sample questions
SAMPLE_QUESTIONS = [
    "What is Crawl4AI?",
    "How do I add a proxy to my crawler?",
    "How to build a simple web crawler?",
    "How does the JsonCssExtractionStrategy work?",
    "How can I crawl multiple URLs?",
]

# UI
app_ui = ui.page_fluid(
    ui.tags.head(
        ui.tags.style("""
            body { background-color: #ffffff; color: #000000; font-family: 'Segoe UI', sans-serif; }
            .card { background-color: #ffffff; border: 2px solid #000000; border-radius: 12px; padding: 24px; margin-bottom: 20px; }
            .btn-primary { background-color: #2563eb; border: none; font-weight: 600; color: #ffffff; }
            .btn-primary:hover { background-color: #1d4ed8; }
            .btn-outline-secondary { border: 1px solid #374151; color: #000000; background-color: #f3f4f6; }
            .btn-outline-secondary:hover { background-color: #e5e7eb; color: #000000; }
            textarea, input { background-color: #ffffff !important; border: 2px solid #374151 !important; color: #000000 !important; font-size: 16px !important; }
            .response-box { background-color: #ffffff; border: 2px solid #374151; border-radius: 8px; padding: 20px; white-space: pre-wrap; font-family: 'Consolas', 'Monaco', monospace; font-size: 15px; line-height: 1.7; color: #000000; }
            h2 { color: #1e40af; }
            h4 { color: #000000; }
            .subtitle { color: #374151; font-size: 1em; margin-bottom: 20px; }
            label { color: #000000; font-weight: 600; font-size: 16px; }
            p { color: #000000; }
        """)
    ),
    ui.div(
        {"class": "container", "style": "max-width: 800px; margin-top: 40px;"},
        ui.div(
            {"class": "card"},
            ui.h2("🤖 Crawl4AI Assistant"),
            ui.div(
                {"class": "subtitle"},
                "Ask questions about the Crawl4AI library — powered by Google Gemini + LlamaIndex"
            ),
            ui.input_text_area(
                "question", 
                "Your question:", 
                placeholder="e.g., How do I extract data using CSS selectors?",
                width="100%",
                rows=3
            ),
            ui.input_action_button("ask", "Ask", class_="btn-primary mt-2"),
            ui.hr(),
            ui.p("💡 Try a sample question:", style="color: #aaa; font-size: 0.85em;"),
            ui.div(
                {"class": "d-flex flex-wrap gap-2"},
                *[ui.input_action_button(f"sample_{i}", q[:40] + "..." if len(q) > 40 else q, class_="btn-outline-secondary btn-sm") 
                  for i, q in enumerate(SAMPLE_QUESTIONS)]
            ),
        ),
        ui.div(
            {"class": "card"},
            ui.h4("📝 Response"),
            ui.output_ui("response"),
        ),
    ),
)

# Server
def server(input, output, session):
    response_text = reactive.Value("")
    
    @reactive.effect
    @reactive.event(input.ask)
    def handle_ask():
        if input.question():
            response_text.set("🔄 Thinking...")
            try:
                response = query_engine.query(input.question())
                response_text.set(str(response))
            except Exception as e:
                response_text.set(f"❌ Error: {str(e)}")
    
    # Handle sample question clicks
    for i, q in enumerate(SAMPLE_QUESTIONS):
        def make_handler(question):
            @reactive.effect
            @reactive.event(getattr(input, f"sample_{SAMPLE_QUESTIONS.index(question)}"))
            def _():
                ui.update_text_area("question", value=question)
        make_handler(q)
    
    @output
    @render.ui
    def response():
        text = response_text()
        if not text:
            return ui.div("Ask a question to get started!", style="color: #666;")
        return ui.div({"class": "response-box"}, text)

app = App(app_ui, server)
