"""
Chat handler for AI-powered chatbot using Groq API with Llama-3
"""
import os
import json
from typing import Dict, List, Optional
from groq import Groq

# Load environment variables (should already be loaded by app.py, but safe to load here too)
from dotenv import load_dotenv
load_dotenv()

# Get Groq API key from environment
GROQ_API_KEY = os.getenv('GROQ_API_KEY')

# Initialize Groq client (will be None if key is missing)
client = None

def _ensure_client():
    """Ensure Groq client is initialized. Raises error if API key is missing."""
    global client
    
    if not GROQ_API_KEY:
        raise RuntimeError(
            "GROQ_API_KEY environment variable is not set. "
            "Please set it in your .env file or environment variables. "
            "See .env.example for reference."
        )
    
    if client is None:
        try:
            client = Groq(api_key=GROQ_API_KEY)
            print("✅ Groq API client initialized successfully")
        except Exception as e:
            raise RuntimeError(
                f"Failed to initialize Groq client: {str(e)}. "
                "Please check that your GROQ_API_KEY is valid."
            )
    
    return client

# Try to initialize client at module load (but don't fail if key is missing)
if GROQ_API_KEY:
    try:
        client = Groq(api_key=GROQ_API_KEY)
        print("✅ Groq API client initialized successfully")
    except Exception as e:
        print(f"⚠️  Warning: Could not initialize Groq client: {e}")
        print("⚠️  Chat feature will not be available until GROQ_API_KEY is set correctly.")

# Model configuration
MODEL_8B = "llama-3.1-8b-instant"
MODEL_70B = "llama-3.1-70b-versatile"
DEFAULT_MODEL = MODEL_8B  # Use 8B for faster responses, switch to 70B for better quality

# System prompt for the chatbot
SYSTEM_PROMPT = """You are a helpful AI assistant for the World Bank Loan Prediction Dashboard. 
You help users understand:
- How to use the prediction tool
- Loan outcome predictions and their meanings
- Model explanations and feature importance
- World Bank loan data and statistics

Be concise, accurate, and helpful. If you don't know something, say so."""


def chat_with_llm(
    user_message: str,
    conversation_history: Optional[List[Dict]] = None,
    model: str = DEFAULT_MODEL,
    use_rag_context: Optional[str] = None,
    use_explanation_context: Optional[str] = None
) -> Dict:
    """
    Send a message to the LLM and get a response
    
    Args:
        user_message: User's message
        conversation_history: Previous conversation messages
        model: Model to use (8B or 70B)
        use_rag_context: Optional RAG context to include
        use_explanation_context: Optional model explanation context
        
    Returns:
        Dict with 'response', 'model_used', and 'success' keys
    """
    try:
        groq_client = _ensure_client()
    except RuntimeError as e:
        return {
            'response': str(e),
            'model_used': None,
            'success': False,
            'error': str(e)
        }
    
    if not groq_client:
        return {
            'response': 'Sorry, the AI chat service is not available. Please check that GROQ_API_KEY is set in your environment.',
            'model_used': None,
            'success': False,
            'error': 'Groq client not initialized'
        }
    
    try:
        # Build messages
        messages = []
        
        # Add system prompt with context
        system_content = SYSTEM_PROMPT
        if use_rag_context:
            system_content += f"\n\nAdditional context from documents:\n{use_rag_context}"
        if use_explanation_context:
            system_content += f"\n\nModel explanation context:\n{use_explanation_context}"
        
        messages.append({
            "role": "system",
            "content": system_content
        })
        
        # Add conversation history
        if conversation_history:
            messages.extend(conversation_history)
        
        # Add current user message
        messages.append({
            "role": "user",
            "content": user_message
        })
        
        # Call Groq API
        response = groq_client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.7,
            max_tokens=1024,
            stream=False
        )
        
        assistant_message = response.choices[0].message.content
        
        return {
            'response': assistant_message,
            'model_used': model,
            'success': True
        }
        
    except Exception as e:
        return {
            'response': f'Sorry, an error occurred: {str(e)}',
            'model_used': None,
            'success': False,
            'error': str(e)
        }


def detect_query_type(user_message: str) -> str:
    """
    Detect what type of query the user is asking
    
    Returns:
        'general', 'rag', 'explain', or 'auto'
    """
    message_lower = user_message.lower()
    
    # Check for explanation requests
    explain_keywords = ['explain', 'why', 'how', 'feature', 'importance', 'shap', 'prediction', 'model']
    if any(keyword in message_lower for keyword in explain_keywords):
        if 'feature' in message_lower or 'importance' in message_lower or 'shap' in message_lower:
            return 'explain'
    
    # Check for data/document queries
    rag_keywords = ['data', 'dataset', 'csv', 'statistics', 'what is', 'tell me about', 'information about']
    if any(keyword in message_lower for keyword in rag_keywords):
        return 'rag'
    
    # Default to general
    return 'general'


def format_conversation_history(messages: List[Dict]) -> List[Dict]:
    """
    Format conversation history for the API
    
    Args:
        messages: List of message dicts with 'role' and 'content'
        
    Returns:
        Formatted messages list
    """
    formatted = []
    for msg in messages:
        if 'role' in msg and 'content' in msg:
            formatted.append({
                'role': msg['role'],
                'content': msg['content']
            })
    return formatted

