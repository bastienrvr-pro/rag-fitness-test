"""
RAG Fitness Chatbot - Gradio Interface with Feedback System

Features:
- Chat interface with conversation history
- Thumbs up/down feedback buttons
- Source citations display
- Feedback export (JSON)
- Settings (Hybrid search toggle, top_k)
"""

import gradio as gr
import sys
from pathlib import Path
import json
from datetime import datetime
from typing import List, Dict, Tuple

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from chatbot import RAGChatbot
import config


# ==============================================================================
# GLOBAL STATE
# ==============================================================================

# Initialize chatbot
print("\n" + "="*80)
print("🚀 INITIALIZING RAG FITNESS CHATBOT")
print("="*80 + "\n")

chatbot = RAGChatbot()

# Feedback storage
FEEDBACK_FILE = Path(__file__).parent / "data" / "feedback.json"
feedback_data = []

# Load existing feedback
if FEEDBACK_FILE.exists():
    with open(FEEDBACK_FILE, 'r', encoding='utf-8') as f:
        feedback_data = json.load(f)
    print(f"📊 Loaded {len(feedback_data)} previous feedback entries\n")


# ==============================================================================
# FEEDBACK FUNCTIONS
# ==============================================================================

def save_feedback(
    question: str,
    answer: str,
    sources: List[Dict],
    feedback_type: str,  # "positive" or "negative"
    comment: str = ""
):
    """Save feedback to JSON file"""
    
    feedback_entry = {
        "timestamp": datetime.now().isoformat(),
        "question": question,
        "answer": answer,
        "sources": [
            {
                "source": s['source'],
                "page": s['page'],
                "score": s['score']
            }
            for s in sources
        ],
        "feedback": feedback_type,
        "comment": comment
    }
    
    feedback_data.append(feedback_entry)
    
    # Save to file
    FEEDBACK_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(FEEDBACK_FILE, 'w', encoding='utf-8') as f:
        json.dump(feedback_data, f, indent=2, ensure_ascii=False)
    
    return f"✅ Feedback saved! Total: {len(feedback_data)}"


def get_feedback_stats():
    """Get feedback statistics"""
    
    if not feedback_data:
        return "No feedback yet"
    
    positive = sum(1 for f in feedback_data if f['feedback'] == 'positive')
    negative = sum(1 for f in feedback_data if f['feedback'] == 'negative')
    total = len(feedback_data)
    
    stats = f"""📊 Feedback Statistics:
    
Total feedback: {total}
👍 Positive: {positive} ({positive/total*100:.1f}%)
👎 Negative: {negative} ({negative/total*100:.1f}%)

Recent feedback:
"""
    
    # Show last 3
    for entry in feedback_data[-3:]:
        emoji = "👍" if entry['feedback'] == 'positive' else "👎"
        stats += f"\n{emoji} {entry['timestamp'][:19]}\n"
        stats += f"   Q: {entry['question'][:60]}...\n"
    
    return stats


# ==============================================================================
# CHAT FUNCTIONS
# ==============================================================================

def format_sources_html(sources: List[Dict]) -> str:
    """Format sources as HTML for better display"""
    
    html = '<div style="margin-top: 20px; padding: 15px; background-color: #1f2937; border-radius: 8px; border: 1px solid #374151;">'
    html += '<h3 style="margin-top: 0; color: #f3f4f6;">📚 Sources Used</h3>'
    
    for i, source in enumerate(sources, 1):
        html += f'<div style="margin: 10px 0; padding: 10px; background-color: #374151; border-radius: 5px; border: 1px solid #4b5563;">'
        html += f'<strong style="color: #f3f4f6;">{i}. {source["source"]}</strong><br>'
        html += f'<span style="color: #9ca3af;">Page {source["page"]} | Score: {source["score"]:.3f}</span><br>'
        html += f'<span style="font-size: 0.9em; color: #6b7280;">{source["authors"]} ({source["year"]})</span>'
        html += '</div>'
    
    html += '</div>'
    
    return html


def chat_function(
    message: str,
    history: List,
    use_hybrid: bool,
    top_k: int
) -> Tuple[List, str, Dict]:
    """
    Main chat function
    
    Returns:
        - Updated history (list of dicts with role/content)
        - Sources HTML
        - Result dict (for feedback)
    """
    
    if not message.strip():
        return history, "", {}
    
    # Generate answer
    result = chatbot.answer(
        question=message,
        doc_type="scientific_paper",
        top_k=top_k,
        use_hybrid=use_hybrid
    )
    
    answer = result['answer']
    sources = result['sources']
    
    # Update history - Gradio new format (dict with role/content)
    if history is None:
        history = []
    
    history.append({"role": "user", "content": message})
    history.append({"role": "assistant", "content": answer})
    
    # Format sources
    sources_html = format_sources_html(sources)
    
    return history, sources_html, result


def handle_feedback(
    feedback_type: str,
    history: List,
    last_result: Dict,
    comment: str
):
    """Handle thumbs up/down feedback"""
    
    if not history or not last_result:
        return "⚠️ No message to provide feedback on"
    
    # Get last Q&A from new format (dicts with role/content)
    # Last user message
    user_messages = [msg for msg in history if msg.get("role") == "user"]
    assistant_messages = [msg for msg in history if msg.get("role") == "assistant"]
    
    if not user_messages or not assistant_messages:
        return "⚠️ No complete Q&A to provide feedback on"
    
    last_question = user_messages[-1]["content"]
    last_answer = assistant_messages[-1]["content"]
    
    # Save feedback
    message = save_feedback(
        question=last_question,
        answer=last_answer,
        sources=last_result.get('sources', []),
        feedback_type=feedback_type,
        comment=comment
    )
    
    return message


# ==============================================================================
# GRADIO INTERFACE
# ==============================================================================

# Custom CSS for better appearance
custom_css = """
#chatbot {
    height: 500px;
}
.feedback-buttons {
    display: flex;
    gap: 10px;
    margin-top: 10px;
}
"""

with gr.Blocks(css=custom_css, title="RAG Fitness Chatbot") as demo:
    
    # Header
    gr.Markdown("""
    # 🏋️ RAG Fitness Chatbot
    
    **Your AI fitness & nutrition advisor powered by scientific research**
    
    Ask questions about:
    - 🥩 Protein intake & nutrition
    - 💪 Training variables (volume, frequency, intensity)
    - 📏 Range of motion effects
    - 💊 Supplementation
    
    *Powered by: Semantic Chunking + Hybrid Search (BM25 + Dense + Cross-Encoder)*
    """)
    
    # Store last result for feedback
    last_result = gr.State({})
    
    with gr.Row():
        with gr.Column(scale=2):
            # Chat interface
            chatbot_ui = gr.Chatbot(
                label="Conversation",
                elem_id="chatbot",
                height=500
            )
            
            with gr.Row():
                message_input = gr.Textbox(
                    placeholder="Ask about fitness, nutrition, or training...",
                    label="Your Question",
                    scale=4
                )
                submit_btn = gr.Button("Send 🚀", scale=1, variant="primary")
            
            # Feedback section
            gr.Markdown("### 💬 Feedback on Last Answer")
            
            with gr.Row():
                thumbs_up_btn = gr.Button("👍 Helpful", scale=1)
                thumbs_down_btn = gr.Button("👎 Not Helpful", scale=1)
                clear_btn = gr.Button("🗑️ Clear Chat", scale=1)
            
            feedback_comment = gr.Textbox(
                placeholder="Optional: Tell us more about your feedback...",
                label="Feedback Comment (Optional)",
                lines=2
            )
            
            feedback_status = gr.Textbox(
                label="Feedback Status",
                interactive=False
            )
        
        with gr.Column(scale=1):
            # Settings
            gr.Markdown("### ⚙️ Settings")
            
            use_hybrid = gr.Checkbox(
                label="Use Hybrid Search (BM25 + Dense + Reranking)",
                value=True,
                info="Recommended for best results"
            )
            
            top_k = gr.Slider(
                minimum=3,
                maximum=10,
                value=5,
                step=1,
                label="Number of sources to retrieve",
                info="More sources = more context but slower"
            )
            
            # Sources display
            gr.Markdown("### 📚 Sources")
            sources_display = gr.HTML(
                label="Sources Used",
                value="<p style='color: #9ca3af; padding: 15px; background-color: #1f2937; border-radius: 8px; border: 1px solid #374151;'>Sources will appear here after asking a question</p>"
            )
            
            # Feedback stats
            gr.Markdown("### 📊 Feedback Stats")
            stats_display = gr.Textbox(
                label="Statistics",
                value=get_feedback_stats(),
                interactive=False,
                lines=12
            )
            
            refresh_stats_btn = gr.Button("🔄 Refresh Stats")
    
    # Footer
    gr.Markdown("""
    ---
    **System Info:**
    - Embedding Model: BGE-Large (1024 dim)
    - LLM: Llama 3.2 3B (Ollama)
    - Chunking: Semantic (similarity-based)
    - Search: Hybrid (BM25 + Dense + Cross-Encoder)
    - Knowledge Base: 4 scientific papers, ~1,000 chunks
    - Recall@5: 88.2% ✅
    """)
    
    # ===========================================================================
    # EVENT HANDLERS
    # ===========================================================================
    
    # Send message
    submit_btn.click(
        fn=chat_function,
        inputs=[message_input, chatbot_ui, use_hybrid, top_k],
        outputs=[chatbot_ui, sources_display, last_result]
    ).then(
        fn=lambda: "",
        outputs=message_input
    )
    
    # Enter key to send
    message_input.submit(
        fn=chat_function,
        inputs=[message_input, chatbot_ui, use_hybrid, top_k],
        outputs=[chatbot_ui, sources_display, last_result]
    ).then(
        fn=lambda: "",
        outputs=message_input
    )
    
    # Thumbs up
    thumbs_up_btn.click(
        fn=lambda h, r, c: handle_feedback("positive", h, r, c),
        inputs=[chatbot_ui, last_result, feedback_comment],
        outputs=feedback_status
    ).then(
        fn=lambda: "",
        outputs=feedback_comment
    )
    
    # Thumbs down
    thumbs_down_btn.click(
        fn=lambda h, r, c: handle_feedback("negative", h, r, c),
        inputs=[chatbot_ui, last_result, feedback_comment],
        outputs=feedback_status
    ).then(
        fn=lambda: "",
        outputs=feedback_comment
    )
    
    # Clear chat
    clear_btn.click(
        fn=lambda: ([], "<p style='color: #9ca3af; padding: 15px; background-color: #1f2937; border-radius: 8px; border: 1px solid #374151;'>Sources will appear here after asking a question</p>", {}),
        outputs=[chatbot_ui, sources_display, last_result]
    )
    
    # Refresh stats
    refresh_stats_btn.click(
        fn=get_feedback_stats,
        outputs=stats_display
    )


# ==============================================================================
# LAUNCH
# ==============================================================================

if __name__ == "__main__":
    print("\n" + "="*80)
    print("🚀 LAUNCHING GRADIO INTERFACE")
    print("="*80 + "\n")
    
    demo.launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=False,
        show_error=True
    )
