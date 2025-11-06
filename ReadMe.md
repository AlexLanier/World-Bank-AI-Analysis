# World Bank Loan Prediction Dashboard

A comprehensive AI-powered Flask web application for predicting World Bank loan cancellation and disbursement outcomes using deep learning, with an integrated AI chatbot assistant.

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Set Up API Key

The application requires a Groq API key for the AI chatbot feature. 

**Recommended: Using .env file**

1. Copy the example environment file:
   ```bash
   cp .env.example .env
   ```

2. Edit the `.env` file and replace `your_key_here` with your actual Groq API key:
   ```bash
   GROQ_API_KEY=your_actual_api_key_here
   ```

3. Get your API key from: https://console.groq.com/

**Alternative: Environment Variables**

You can also export the key in your shell:
```bash
export GROQ_API_KEY="your_groq_api_key_here"
```

**Important Security Notes:**
- The `.env` file is automatically ignored by git (see `.gitignore`)
- Never commit your actual API key to version control
- The application will show a clear error message if the key is missing when you try to use the chat feature

### 3. Run the Application

```bash
python app.py
```

The app will start on `http://localhost:9000`

## 📚 Documentation

- **Full Documentation**: See `docs/ReadMe.md`
- **Setup Guide**: See `docs/SETUP_GUIDE.md`
- **Sample Prompts**: See `docs/SAMPLE_PROMPTS.md`
- **Project Structure**: See `docs/PROJECT_STRUCTURE.md`

## 🔒 Security

- API keys are loaded from environment variables only
- No hardcoded secrets in the codebase
- `.env` file is gitignored
- All secret files are excluded from version control

## Features

- 🤖 AI-powered chatbot with Groq API
- 📊 Loan outcome prediction
- 🔍 Model explainability
- 📚 RAG (Retrieval-Augmented Generation)
- 📄 PDF export
- 🎨 Modern, responsive UI

## License

MIT License

