import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from flask import Flask, render_template, request, jsonify, send_from_directory
import pickle
import os

# Load environment variables from .env file
from dotenv import load_dotenv
load_dotenv()

# Import new modules
from chat import chat_with_llm, detect_query_type, format_conversation_history
from rag_engine import query_rag, index_folder
from explain import explain_prediction, format_explanation_for_llm, initialize_explainer

app = Flask(__name__)

# Global variables
model = None
device = None
scaler = None
cat_vocab = None
class_names = None
cat_cols = None
num_cols = None
model_hparams = None

# =============================================
# TabTransformer Model Architecture
# =============================================
class TabTransformer(nn.Module):
    def __init__(
        self,
        cat_cardinalities,
        num_dim,
        n_classes,
        d_model=64,
        n_heads=4,
        n_layers=2,
        mlp_hidden=128,
        dropout=0.1,
    ):
        super().__init__()
        self.n_cat = len(cat_cardinalities)
        self.num_dim = num_dim

        self.cat_embeds = nn.ModuleList([nn.Embedding(card, d_model) for card in cat_cardinalities])

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=4 * d_model,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        self.num_proj = nn.Sequential(
            nn.LayerNorm(num_dim) if num_dim > 0 else nn.Identity(),
            nn.Linear(num_dim, d_model) if num_dim > 0 else nn.Identity(),
            nn.GELU(),
        )

        in_dim = d_model + (d_model if num_dim > 0 else 0)
        self.head = nn.Sequential(
            nn.Linear(in_dim, mlp_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, n_classes),
        )

    def forward(self, x_cat, x_num):
        if self.n_cat > 0:
            tokens = [emb(x_cat[:, i]) for i, emb in enumerate(self.cat_embeds)]
            x_tok = torch.stack(tokens, dim=1)         # (B, n_cat, d_model)
            x_ctx = self.transformer(x_tok)            # (B, n_cat, d_model)
            x_cat_pooled = x_ctx.mean(dim=1)           # (B, d_model)
        else:
            x_cat_pooled = None

        x_num_proj = self.num_proj(x_num) if (x_num is not None and x_num.shape[1] > 0) else None

        if x_cat_pooled is not None and x_num_proj is not None:
            x_all = torch.cat([x_cat_pooled, x_num_proj], dim=1)
        else:
            x_all = x_cat_pooled if x_cat_pooled is not None else x_num_proj

        return self.head(x_all)


def load_model():
    """Load the trained model and artifacts"""
    global model, device, scaler, cat_vocab, class_names, cat_cols, num_cols, model_hparams
    
    model_file = 'models/tabtransformer_model.pt'
    artifacts_file = 'models/tabtransformer_artifacts.pkl'
    
    if not os.path.exists(model_file):
        print("⚠️  Model file not found. Please train the model first.")
        return False
    
    if not os.path.exists(artifacts_file):
        print("⚠️  Artifacts file not found. Please train the model first.")
        return False
    
    try:
        # Load artifacts
        with open(artifacts_file, 'rb') as f:
            artifacts = pickle.load(f)
        
        scaler = artifacts['scaler']
        cat_vocab = artifacts['cat_vocab']
        class_names = artifacts['class_names']
        cat_cols = artifacts['cat_cols']
        num_cols = artifacts['num_cols']
        model_hparams = artifacts['model_hparams']
        n_classes = artifacts['n_classes']
        
        # Initialize model architecture
        cat_cardinalities = [len(cat_vocab[c]) for c in cat_cols]
        num_dim = len(num_cols)
        
        model = TabTransformer(
            cat_cardinalities=cat_cardinalities,
            num_dim=num_dim,
            n_classes=n_classes,
            **model_hparams
        )
        
        # Load model weights
        state_dict = torch.load(model_file, map_location='cpu')
        if isinstance(state_dict, dict) and 'model' in state_dict:
            # If checkpoint format
            model.load_state_dict(state_dict['model'])
        else:
            # If direct state dict
            model.load_state_dict(state_dict)
        
        # Set to eval mode
        device = torch.device("cpu")
        model.to(device)
        model.eval()
        
        print("✅ Model loaded successfully")
        print(f"   - Classes: {class_names}")
        print(f"   - Categorical features: {cat_cols}")
        print(f"   - Numerical features: {num_cols}")
        
        # Initialize explainer (would need background data for full SHAP)
        # For now, just store the model reference
        try:
            initialize_explainer(model, None, cat_cols, num_cols)
        except Exception as e:
            print(f"⚠️  Could not initialize explainer: {e}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error loading model: {str(e)}")
        return False


def encode_categorical(data_dict, cat_vocab):
    """Encode categorical features using vocabularies"""
    encoded = []
    for col in cat_cols:
        value = data_dict.get(col, '')
        # Handle unknown values
        if value not in cat_vocab[col]:
            # Use the first vocab entry as default
            encoded.append(0)
        else:
            encoded.append(cat_vocab[col][value])
    return encoded


def preprocess_input(data_dict):
    """Preprocess input data for prediction"""
    # Encode categorical features
    x_cat = encode_categorical(data_dict, cat_vocab)
    
    # Prepare numerical features
    x_num = []
    for col in num_cols:
        # Handle log-prefixed columns
        if col.startswith('log_'):
            original_col = col.replace('log_', '')
            value = data_dict.get(original_col, 0)
            # Apply log transformation
            x_num.append(np.log1p(max(0, value)))
        else:
            x_num.append(data_dict.get(col, 0))
    
    # Scale numerical features
    x_num = scaler.transform([x_num])[0]
    
    # Convert to tensors
    x_cat_tensor = torch.tensor([x_cat], dtype=torch.long)
    x_num_tensor = torch.tensor([x_num], dtype=torch.float32)
    
    return x_cat_tensor, x_num_tensor


@app.route('/')
def index():
    """Render the main prediction page"""
    # Get unique values for dropdowns from vocabularies
    loan_types = sorted(cat_vocab['loan_type'].keys()) if cat_vocab else []
    regions = sorted(cat_vocab['region'].keys()) if cat_vocab else []
    
    return render_template('index.html', 
                         loan_types=loan_types,
                         regions=regions)


@app.route('/analytics')
def analytics():
    """Render the analytics page"""
    return render_template('analytics.html')


@app.route('/model')
def model():
    """Render the model information page"""
    return render_template('model.html')


@app.route('/insights')
def insights():
    """Render the insights page"""
    return render_template('insights.html')


@app.route('/about')
def about():
    """Render the about page"""
    return render_template('about.html')


@app.route('/predict', methods=['POST'])
def predict():
    """Handle prediction requests with optional explanations"""
    try:
        if model is None:
            return jsonify({'error': 'Model not loaded'}), 500
        
        # Get form data
        data = request.json
        
        # Check if explanation is requested
        include_explanation = request.json.get('include_explanation', False)
        
        # Validate required fields (user provides non-log versions)
        required_fields_user = ['loan_type', 'region', 'interest_rate', 
                               'original_principal_amount', 'gdp_total', 'gdp_per_capita']
        for field in required_fields_user:
            if field not in data:
                return jsonify({'error': f'Missing required field: {field}'}), 400
        
        # Preprocess input
        x_cat, x_num = preprocess_input(data)
        
        # Make prediction
        with torch.no_grad():
            x_cat = x_cat.to(device)
            x_num = x_num.to(device)
            logits = model(x_cat, x_num)
            probabilities = torch.softmax(logits, dim=1)[0]
            predicted_class_idx = torch.argmax(logits, dim=1)[0].item()
        
        # Format results
        predicted_class = class_names[predicted_class_idx]
        prob_dict = {class_names[i]: float(prob) for i, prob in enumerate(probabilities)}
        confidence_score = float(probabilities[predicted_class_idx])
        
        response = {
            'prediction': predicted_class,
            'probabilities': prob_dict,
            'confidence_score': confidence_score
        }
        
        # Add explanation if requested
        if include_explanation:
            try:
                explanation = explain_prediction(
                    model=model,
                    input_data=data,
                    cat_cols=cat_cols,
                    num_cols=num_cols,
                    scaler=scaler,
                    cat_vocab=cat_vocab,
                    preprocess_input_func=preprocess_input
                )
                response['explanation'] = explanation
            except Exception as e:
                response['explanation_error'] = str(e)
        
        return jsonify(response)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/chat', methods=['POST'])
def chat():
    """Handle chat requests with LLM"""
    try:
        data = request.json
        user_message = data.get('message', '')
        conversation_history = data.get('history', [])
        query_type = data.get('query_type', 'auto')  # 'general', 'rag', 'explain', 'auto'
        
        if not user_message:
            return jsonify({'error': 'Message is required'}), 400
        
        # Auto-detect query type if not specified
        if query_type == 'auto':
            query_type = detect_query_type(user_message)
        
        # Get RAG context if needed
        rag_context = None
        if query_type in ['rag', 'auto']:
            rag_result = query_rag(user_message, n_results=3)
            if rag_result.get('success') and rag_result.get('context'):
                rag_context = rag_result['context']
        
        # Get explanation context if needed
        explanation_context = None
        if query_type == 'explain':
            # If there's a prediction in the conversation, use it
            # For now, we'll just respond with general explanation info
            explanation_context = "Model explanation context would be included here for explain queries."
        
        # Format conversation history
        formatted_history = format_conversation_history(conversation_history)
        
        # Get LLM response
        response = chat_with_llm(
            user_message=user_message,
            conversation_history=formatted_history,
            use_rag_context=rag_context,
            use_explanation_context=explanation_context
        )
        
        return jsonify({
            'response': response.get('response', 'Sorry, I could not generate a response.'),
            'success': response.get('success', False),
            'query_type': query_type,
            'model_used': response.get('model_used')
        })
        
    except Exception as e:
        return jsonify({
            'error': str(e),
            'success': False
        }), 500


@app.route('/rag-query', methods=['POST'])
def rag_query():
    """Handle RAG-only queries"""
    try:
        data = request.json
        query = data.get('query', '')
        
        if not query:
            return jsonify({'error': 'Query is required'}), 400
        
        result = query_rag(query, n_results=5)
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({
            'error': str(e),
            'success': False
        }), 500


@app.route('/explain', methods=['POST'])
def explain():
    """Handle model explanation requests"""
    try:
        if model is None:
            return jsonify({'error': 'Model not loaded'}), 500
        
        data = request.json
        
        # Validate required fields
        required_fields_user = ['loan_type', 'region', 'interest_rate', 
                               'original_principal_amount', 'gdp_total', 'gdp_per_capita']
        for field in required_fields_user:
            if field not in data:
                return jsonify({'error': f'Missing required field: {field}'}), 400
        
        explanation = explain_prediction(
            model=model,
            input_data=data,
            cat_cols=cat_cols,
            num_cols=num_cols,
            scaler=scaler,
            cat_vocab=cat_vocab,
            preprocess_input_func=preprocess_input
        )
        
        return jsonify(explanation)
        
    except Exception as e:
        return jsonify({
            'error': str(e),
            'success': False
        }), 500


@app.route('/health')
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None
    })


if __name__ == '__main__':
    print("🚀 Starting Flask app...")
    
    # Index RAG files on startup
    print("📚 Indexing RAG files...")
    rag_result = index_folder("./rag_files")
    print(f"   - Indexed {rag_result.get('files_indexed', 0)} files")
    
    if load_model():
        print("✅ Ready to serve predictions!")
    else:
        print("⚠️  Running without pre-trained model")
    
    app.run(debug=True, host='0.0.0.0', port=9000)
