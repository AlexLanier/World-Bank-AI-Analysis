# Project Structure

## Clean Organization

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
├── models/                     # Model files
│   ├── tabtransformer_model.pt
│   ├── tabtransformer_artifacts.pkl
│   ├── tabtransformer_checkpoint.pt
│   ├── tabtransformer_checkpoint.pkl
│   ├── trained_model.cbm
│   └── label_encoder.pkl
│
├── data/                       # Data files
│   ├── raw/                    # Raw data files
│   │   ├── *.csv
│   │   ├── *.json
│   │   ├── *.parquet
│   │   └── *.txt
│   └── processed/              # Processed data
│       └── catboost_info/
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
├── scripts/                    # Utility scripts
│   ├── train_model.py
│   └── test_api.py
│
└── docs/                       # Documentation
    ├── ReadMe.md              # Main README
    ├── PROJECT_STRUCTURE.md   # This file
    ├── SAMPLE_PROMPTS.md      # Chatbot prompt examples
    ├── FOLDER_STRUCTURE.md    # Legacy structure doc
    ├── *.ipynb                # Jupyter notebooks
    └── ui-screenshots/        # UI screenshots
```

## Key Directories

### `/models`
All trained model files, artifacts, and checkpoints.

### `/data/raw`
Raw data files (CSV, JSON, Parquet, TXT).

### `/data/processed`
Processed data and intermediate files.

### `/templates`
HTML templates for the Flask application.

### `/static`
Static assets (CSS, JS) served by Flask.

### `/rag_files`
Files automatically indexed for RAG system.

### `/chroma_db`
Vector database storage (auto-created by ChromaDB).

### `/scripts`
Utility scripts for training and testing.

### `/docs`
All documentation and notebooks.

## Path Updates

After reorganization, the following paths were updated:

- Model paths: `models/tabtransformer_model.pt` (was: `tabtransformer_model.pt`)
- Data files: Moved to `data/raw/`
- Documentation: Moved to `docs/`
- Scripts: Moved to `scripts/`

## Notes

- The app.py file has been updated to use the new model paths
- RAG files remain in `rag_files/` at the root level for easy access
- ChromaDB database is auto-created in `chroma_db/` at root
- All other references should work with the new structure

