# Folder Structure

```
WorldBank AI Analysis/
├── app.py                      # Main Flask application
├── chat.py                     # AI chatbot handler (Groq API)
├── rag_engine.py               # RAG system for document retrieval
├── explain.py                  # Model explanation module
├── requirements.txt            # Python dependencies
├── .env.example                # Environment variables template
├── .gitignore                  # Git ignore file
│
├── templates/                  # HTML templates
│   ├── index.html             # Main prediction page
│   ├── model.html             # Model information page
│   ├── insights.html          # Data insights page
│   ├── analytics.html         # Analytics dashboard
│   └── about.html             # About page
│
├── static/                     # Static assets
│   ├── css/
│   │   ├── main.css           # Main application styles
│   │   └── chat.css           # Chat panel styles
│   └── js/
│       └── chat.js            # Chat UI component
│
├── rag_files/                  # RAG document storage
│   ├── README.md              # RAG files documentation
│   └── worldbank_info.txt     # Sample information file
│
├── chroma_db/                  # ChromaDB vector database (auto-created)
│
├── Model Files/
│   ├── tabtransformer_model.pt         # Trained PyTorch model
│   ├── tabtransformer_artifacts.pkl    # Model artifacts (scaler, vocab, etc.)
│   ├── tabtransformer_checkpoint.pt    # Training checkpoint
│   └── tabtransformer_checkpoint.pkl   # Checkpoint artifacts
│
├── Data Files/                 # CSV and data files
│   ├── gdp_*.csv
│   ├── worldbank_loans*.json
│   └── ...
│
└── Documentation/
    ├── ReadMe.md              # Main README
    ├── FOLDER_STRUCTURE.md    # This file
    └── ...
```

## Key Directories

### `/templates`
HTML templates for the Flask application. Uses Jinja2 templating.

### `/static`
Static assets (CSS, JS, images) served by Flask.

### `/rag_files`
Files automatically indexed for RAG system. Supports:
- `.txt` - Text files
- `.csv` - CSV data files
- `.md` - Markdown files

### `/chroma_db`
Vector database storage (auto-created by ChromaDB). Contains embedded document vectors.

## Backend Modules

- **app.py**: Main Flask app with routes and model loading
- **chat.py**: Groq API integration for LLM chat
- **rag_engine.py**: Document indexing and retrieval
- **explain.py**: Model explanation and feature importance

## Frontend Components

- **static/js/chat.js**: Chatbot UI component
- **static/css/chat.css**: Chat panel styles
- **static/css/main.css**: Main application styles

