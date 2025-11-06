# RAG Files Directory

This directory contains files that are automatically indexed for the RAG (Retrieval-Augmented Generation) system.

## Supported File Types

- **CSV files** (`.csv`): Data files with tabular information
- **Text files** (`.txt`, `.text`): Plain text documents
- **Markdown files** (`.md`, `.markdown`): Markdown formatted documents

## How It Works

When the application starts, all files in this directory are automatically:
1. Read and chunked into smaller pieces
2. Embedded into vectors
3. Stored in ChromaDB for fast retrieval

## Example Files

You can add any relevant World Bank data or documentation files here. They will be automatically indexed and made searchable through the AI chatbot.

## Usage

Ask the chatbot questions like:
- "What data is available in the dataset?"
- "Tell me about World Bank loans"
- "What are the statistics for loan outcomes?"

