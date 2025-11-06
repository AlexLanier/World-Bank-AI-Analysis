# Setup Guide

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Set Up API Key

The application requires a Groq API key for the AI chatbot feature. Follow these steps:

#### Option A: Using .env file (Recommended for local development)

1. Copy the example environment file:
   ```bash
   cp .env.example .env
   ```

2. Edit the `.env` file and replace `your_key_here` with your actual Groq API key:
   ```bash
   GROQ_API_KEY=your_actual_api_key_here
   ```

3. Get your API key from: https://console.groq.com/

#### Option B: Using environment variables

Export the key in your shell:
```bash
export GROQ_API_KEY="your_groq_api_key_here"
```

**Important**: 
- The `.env` file is automatically ignored by git (see `.gitignore`)
- Never commit your actual API key to version control
- The application will raise a clear error if the key is missing

### 3. Run the Application

```bash
python app.py
```

The app will start on `http://localhost:9000`

## Project Structure

After cleanup, the project is organized as:

```
WorldBank AI Analysis/
├── app.py                 # Main Flask app
├── chat.py               # AI chatbot
├── rag_engine.py         # RAG system
├── explain.py            # Model explanations
├── models/               # Model files
├── data/                 # Data files
│   ├── raw/             # Raw data
│   └── processed/       # Processed data
├── templates/            # HTML templates
├── static/              # CSS/JS files
├── rag_files/           # RAG documents
├── scripts/             # Utility scripts
└── docs/                # Documentation
```

## Testing

### Health Check
```bash
curl http://localhost:9000/health
```

### Make a Prediction
```bash
curl -X POST http://localhost:9000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "loan_type": "SCP USD",
    "region": "EAST ASIA AND PACIFIC",
    "interest_rate": 2.5,
    "original_principal_amount": 50000000,
    "gdp_total": 5000000000,
    "gdp_per_capita": 5000
  }'
```

### Chat with AI (requires GROQ_API_KEY)
```bash
curl -X POST http://localhost:9000/chat \
  -H "Content-Type: application/json" \
  -d '{
    "message": "How does the model work?",
    "history": []
  }'
```

## Troubleshooting

### Model not loading
- Check that `models/tabtransformer_model.pt` exists
- Verify `models/tabtransformer_artifacts.pkl` exists

### Chat not working
- Set `GROQ_API_KEY` environment variable
- Check internet connection for API calls

### RAG not working
- Ensure files are in `rag_files/` directory
- Check ChromaDB initialization in logs

### Import errors
- Install all dependencies: `pip install -r requirements.txt`
- Check Python version (3.8+ required)

## Notes

- The app automatically indexes files in `rag_files/` on startup
- ChromaDB database is created in `chroma_db/` automatically
- All model paths have been updated to use `models/` directory
- Data files are organized in `data/raw/` and `data/processed/`

