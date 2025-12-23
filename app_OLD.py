"""
Application Gradio - Interface Chatbot RAG Fitness (Version Minimaliste)

Compatible avec toutes versions de Gradio
Lancer avec : python app.py
"""

import gradio as gr
from src.chatbot import RAGChatbot


# ============================================================================
# INITIALISATION
# ============================================================================

print("\n🚀 Démarrage application...")

# Initialiser chatbot
chatbot = RAGChatbot()


# ============================================================================
# FONCTION PRINCIPALE
# ============================================================================

def respond(message, history):
    """
    Répond à un message utilisateur
    
    Args:
        message: Question utilisateur
        history: Historique conversation
    
    Returns:
        Réponse formatée
    """
    
    # Obtenir réponse du chatbot
    result = chatbot.answer(
        question=message,
        doc_type="scientific_paper",
        min_year=None,
        top_k=5
    )
    
    # Formater avec sources
    response = chatbot.format_answer_with_sources(result)
    
    return response


# ============================================================================
# INTERFACE GRADIO (VERSION MINIMALE)
# ============================================================================

# Questions exemples
examples = [
    "Combien de protéines par jour pour l'hypertrophie ?",
    "Quel est le volume d'entraînement optimal par semaine ?",
    "Le full ROM est-il meilleur que le partial ROM ?",
    "Composition nutritionnelle du poulet ?",
    "La créatine est-elle efficace pour la musculation ?"
]

# Interface simple
demo = gr.ChatInterface(
    fn=respond,
    title="🏋️ RAG Fitness Assistant",
    description="""
Posez vos questions sur :
- 💪 Nutrition pour l'hypertrophie
- 🏋️ Volume et fréquence d'entraînement  
- 📊 Composition nutritionnelle des aliments
- 🔬 Suppléments basés sur la science

**Sources** : Articles scientifiques (Helms, ISSN, Schoenfeld, etc.) + Base CIQUAL
""",
    examples=examples
)


# ============================================================================
# LANCEMENT
# ============================================================================

if __name__ == "__main__":
    
    print("\n" + "="*80)
    print("✅ APPLICATION PRÊTE")
    print("="*80)
    print(f"\n🌐 Ouvre ton navigateur : http://localhost:7860")
    print(f"💡 Questions exemples disponibles en bas")
    print(f"⚡ Appuie sur Ctrl+C pour arrêter\n")
    
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False
    )
