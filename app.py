import streamlit as st
from rag_engine import RAGEngine
import os

# Page configuration
st.set_page_config(
    page_title="Company Policy Chatbot",
    page_icon="📚",
    layout="wide"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stTextInput > div > div > input {
        background-color: #f0f2f6;
    }
    .source-box {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
        border-left: 4px solid #4CAF50;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize RAG engine in session state (cached)
@st.cache_resource
def initialize_rag():
    """Initialize and cache RAG engine"""
    rag = RAGEngine()
    rag.initialize()
    return rag

# Initialize session state for messages
if "messages" not in st.session_state:
    st.session_state.messages = []

# Initialize RAG engine
if "rag_engine" not in st.session_state:
    with st.spinner("🔧 Initializing chatbot... This may take a moment."):
        st.session_state.rag_engine = initialize_rag()

# Sidebar
with st.sidebar:
    st.title("📚 Policy Chatbot")
    st.markdown("---")
    
    st.markdown("""
    ### About
    This chatbot answers questions about company policies using 
    **Retrieval-Augmented Generation (RAG)**.
    
    ### Features
    ✓ Answers policy questions  
    ✓ Cites sources  
    ✓ Conversation memory  
    ✓ No hallucinations  
    """)
    
    st.markdown("---")
    
    # Clear conversation button
    if st.button("🔄 Clear Conversation", use_container_width=True):
        st.session_state.messages = []
        st.session_state.rag_engine.reset_memory()
        st.success("Conversation cleared!")
        st.rerun()
    
    st.markdown("---")
    
    # Example questions
    st.markdown("### 💡 Example Questions")
    example_questions = [
        "What is the leave policy?",
        "How can employees request remote work?",
        "What are the IT security requirements?",
        "What is the dress code policy?"
    ]
    
    for question in example_questions:
        if st.button(question, key=f"ex_{question}", use_container_width=True):
            # Add to messages and rerun to display
            st.session_state.messages.append({"role": "user", "content": question})
            st.rerun()

# Main chat interface
st.title("💬 Company Policy Assistant")
st.markdown("Ask me anything about company policies!")
st.markdown("---")

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        
        # Display sources if available (for assistant messages)
        if message["role"] == "assistant" and "sources" in message:
            with st.expander("📄 View Sources"):
                for i, source in enumerate(message["sources"], 1):
                    st.markdown(f"""
                    <div class="source-box">
                        <strong>Source {i}:</strong> {source['source']} (Page {source['page']})<br>
                        <em>{source['content']}</em>
                    </div>
                    """, unsafe_allow_html=True)

# Chat input
if prompt := st.chat_input("Ask a question about company policies..."):
    # Add user message to chat
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Get response from RAG engine
    with st.chat_message("assistant"):
        with st.spinner("🔍 Searching company policies..."):
            try:
                result = st.session_state.rag_engine.query(prompt)
                
                # Display answer
                st.markdown(result["answer"])
                
                # Display sources
                if result["sources"]:
                    with st.expander("📄 View Sources"):
                        for i, source in enumerate(result["sources"], 1):
                            st.markdown(f"""
                            <div class="source-box">
                                <strong>Source {i}:</strong> {source['source']} (Page {source['page']})<br>
                                <em>{source['content']}</em>
                            </div>
                            """, unsafe_allow_html=True)
                
                # Add assistant response to chat history
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": result["answer"],
                    "sources": result["sources"]
                })
                
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
                st.info("💡 Please make sure you have PDF documents in the 'data' folder and your OpenAI API key is set correctly.")

# Footer
st.markdown("---")
st.markdown("""
    <div style='text-align: center; color: #666; padding: 1rem;'>
        Built with LangChain, Chroma, and OpenAI | RAG Architecture
    </div>
""", unsafe_allow_html=True)
