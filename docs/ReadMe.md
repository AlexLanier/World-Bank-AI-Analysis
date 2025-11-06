# World Bank Loan Prediction Dashboard - AI-Powered

A comprehensive Flask web application for predicting World Bank loan cancellation and disbursement outcomes using deep learning, with an integrated AI chatbot assistant.

## 🆕 New Features

### 🤖 AI-Powered Chatbot
- **Side-panel chat interface** with collapsible design
- **Groq API integration** using Llama-3 (8B or 70B models)
- **Three types of answers**:
  - **General AI responses**: LLM-powered answers to general questions
  - **Dataset-aware answers**: RAG (Retrieval-Augmented Generation) from indexed documents
  - **Model explanation answers**: Feature importance and SHAP-based explanations

### 📚 RAG (Retrieval-Augmented Generation)
- Automatic indexing of files in `/rag_files` directory
- Supports CSV, TXT, and Markdown files
- ChromaDB vector storage for fast semantic search
- Context-aware responses based on your data

### 🔍 Model Explainability
- Feature importance visualization
- SHAP-like explanation values
- "Explain Prediction" button for detailed insights
- Top contributing features display

### 🎨 Enhanced UI
- Modern, responsive design
- Collapsible AI chat panel
- Improved form controls with tooltips and sliders
- Better visual hierarchy and spacing
- Organized static files (CSS/JS separation)

## Features

- 🌍 Predict loan outcomes (Fully Disbursed, Major Cancellation, Minor Cancellation)
- 📊 Beautiful, modern UI with gradient design
- 🤖 TabTransformer deep learning model (PyTorch)
- 🧠 AI chatbot assistant with multiple capabilities
- 📈 Displays prediction probabilities with confidence scores
- 🔍 Model explanation and feature importance
- 🎨 Responsive design
- 📄 PDF export functionality

## Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Set Up API Key

The application requires a Groq API key for the AI chatbot feature. 

#### Using .env file (Recommended)

1. Copy the example file:
   ```bash
   cp .env.example .env
   ```

2. Edit `.env` and add your API key:
   ```bash
   GROQ_API_KEY=your_actual_api_key_here
   ```

3. Get your API key from: https://console.groq.com/

**Note**: The `.env` file is gitignored and will not be committed. The application will raise a clear error if the key is missing.

### 3. Add RAG Files (Optional)

Place any text, CSV, or markdown files in the `rag_files/` directory. These will be automatically indexed on startup.

### 4. Run the Flask App

```bash
python app.py
```

The app will start on `http://localhost:9000`

## Usage

### Making Predictions

1. Open your web browser and navigate to `http://localhost:9000`
2. Fill in the loan details:
   - Loan Type
   - Region
   - Interest Rate
   - Original Principal Amount
   - GDP Total
   - GDP Per Capita
3. Click "🔮 Predict Loan Outcome"
4. View the prediction with confidence scores
5. Click "🔍 Explain Prediction" to see feature importance

### Using the AI Chatbot

1. **Open the chat panel** on the right side of the screen
2. **Ask questions** about:
   - How the model works
   - What data is available
   - Loan outcome explanations
   - Feature importance
   - General questions about World Bank loans

#### Example Chat Prompts

- **General Questions**:
  - "How does the loan prediction model work?"
  - "What are the different loan types?"
  - "Explain the prediction outcomes"

- **Data Questions** (RAG-enabled):
  - "What data is available in the dataset?"
  - "Tell me about World Bank loan statistics"
  - "What are the loan cancellation rates?"

- **Model Explanation**:
  - "Explain why this loan was predicted as Fully Disbursed"
  - "What features are most important for predictions?"
  - "How does interest rate affect loan outcomes?"

## API Endpoints

### POST /predict
Make a prediction for loan outcome.

**Request Body:**
```json
{
  "loan_type": "SCP USD",
  "region": "EAST ASIA AND PACIFIC",
  "interest_rate": 2.5,
  "original_principal_amount": 50000000,
  "gdp_total": 5000000000,
  "gdp_per_capita": 5000,
  "include_explanation": false
}
```

**Response:**
```json
{
  "prediction": "Fully Disbursed",
  "confidence_score": 0.95,
  "probabilities": {
    "Fully Disbursed": 0.95,
    "Major Cancellation": 0.03,
    "Minor Cancellation": 0.02
  }
}
```

### POST /chat
Chat with the AI assistant.

**Request Body:**
```json
{
  "message": "How does the model work?",
  "history": [],
  "query_type": "auto"
}
```

**Response:**
```json
{
  "response": "The model uses a TabTransformer architecture...",
  "success": true,
  "query_type": "general",
  "model_used": "llama-3.1-8b-instant"
}
```

### POST /explain
Get model explanation for a prediction.

**Request Body:** (Same as /predict)

**Response:**
```json
{
  "success": true,
  "predicted_class": 0,
  "top_features": [
    {
      "feature": "interest_rate",
      "importance": 0.25,
      "contribution": 0.25
    }
  ]
}
```

### POST /rag-query
Query the RAG system directly.

**Request Body:**
```json
{
  "query": "What data is available?"
}
```

### GET /health
Check if the app and model are loaded correctly.

## Features Used by the Model

### Categorical Features
- `loan_type`: Type of loan (e.g., SCP USD, CPL, FSL)
- `region`: Geographic region

### Numerical Features (log-transformed)
- `interest_rate`: Loan interest rate percentage
- `log_original_principal_amount`: Log of principal amount
- `log_gdp_total`: Log of total GDP
- `log_gdp_per_capita`: Log of GDP per capita

## Model Architecture

- **Type**: TabTransformer (PyTorch)
- **Architecture**: Transformer-based encoder for categorical features + MLP for numerical features
- **Accuracy**: ~87% on test set
- **Classes**: 3 (Fully Disbursed, Major Cancellation, Minor Cancellation)

## Directory Structure

```
.
├── app.py                 # Flask application
├── chat.py               # AI chatbot handler
├── rag_engine.py         # RAG system for document retrieval
├── explain.py            # Model explanation module
├── templates/
│   ├── index.html        # Main prediction UI
│   ├── model.html        # Model information page
│   ├── insights.html     # Data insights page
│   └── about.html        # About page
├── static/
│   ├── css/
│   │   ├── main.css      # Main styles
│   │   └── chat.css      # Chat panel styles
│   └── js/
│       └── chat.js       # Chat UI component
├── rag_files/            # Files to index for RAG
├── chroma_db/            # ChromaDB vector database (auto-created)
├── requirements.txt      # Python dependencies
├── tabtransformer_model.pt      # Trained PyTorch model
├── tabtransformer_artifacts.pkl # Model artifacts
└── README.md             # This file
```

## Backend Architecture

### chat.py
- Handles LLM requests via Groq API
- Detects query type (general, RAG, explain)
- Formats conversation history
- Integrates RAG and explanation contexts

### rag_engine.py
- Indexes documents from `/rag_files`
- Uses ChromaDB for vector storage
- Performs semantic search on queries
- Returns relevant document chunks

### explain.py
- Computes feature importance
- Generates SHAP-like explanations
- Formats explanations for LLM context
- Provides top contributing features

## Configuration

### Chat Model Selection
Edit `chat.py` to change the model:
- `MODEL_8B = "llama-3.1-8b-instant"` (faster, default)
- `MODEL_70B = "llama-3.1-70b-versatile"` (better quality)

### RAG Settings
- Default results: 3 documents per query
- Chunk size: 500 words
- Chunk overlap: 50 words

## Troubleshooting

### Chatbot not responding
- Check that `GROQ_API_KEY` is set correctly
- Verify internet connection for API calls
- Check browser console for errors

### RAG not working
- Ensure files are in `rag_files/` directory
- Check that ChromaDB initialized successfully
- Look for indexing errors in server logs

### Model explanation errors
- Ensure model is loaded correctly
- Check that input data is valid
- Verify feature names match training data

## Notes

- The app expects a trained TabTransformer model saved as `tabtransformer_model.pt`
- All numerical features are automatically log-transformed (log1p) before prediction
- The RAG system automatically indexes files on startup
- Chat history is maintained in the frontend (browser memory)

## License

MIT License

## Credits

- **Model**: TabTransformer architecture
- **LLM**: Groq API with Llama-3
- **Vector DB**: ChromaDB
- **Embeddings**: Sentence Transformers
