# Document RAG Chatbot

A command-line Retrieval-Augmented Generation (RAG) application that
answers questions using your own documents.

Built with Python, OpenAI, and FAISS, the application loads TXT, PDF,
and DOCX files, retrieves relevant text, and uses that context to
generate answers with source filenames.

## Features

- Read documents from TXT, PDF, and DOCX files
- Search nested folders inside the data directory
- Split documents into overlapping text chunks
- Generate embeddings using OpenAI
- Retrieve relevant chunks using FAISS
- Generate answers using retrieved document context
- Display retrieved source filenames
- Save and reuse the vector index between runs
- Interactive command-line interface

## Tech Stack

- Python
- OpenAI API
- FAISS
- NumPy
- python-dotenv
- pypdf
- python-docx

## Models

| Purpose | Model |
|---------|-------|
| Text embeddings | `text-embedding-3-small` |
| Answer generation | `gpt-4.1-mini` |

The model names are configured directly in `rag.py`.

## Project Structure

```text
document-rag-chatbot/
├── data/
│   ├── notes.txt
│   ├── document.pdf
│   └── report.docx
├── vector_store/          # Generated automatically
│   ├── faiss.index
│   └── chunks.json
├── .env                  # Create locally; do not commit
├── .env.example
├── .gitignore
├── rag.py
└── README.md
```

The files shown inside `data/` are examples. Replace them with your
own documents.

## How It Works

```text
Documents: TXT / PDF / DOCX
              |
              v
       Extract document text
              |
              v
  Split into overlapping chunks
              |
              v
     Generate OpenAI embeddings
              |
              v
       Save vectors in FAISS
              |
              v
         User question
              |
              v
       Embed the question
              |
              v
 Retrieve the most similar chunks
              |
              v
 Send question + context to OpenAI
              |
              v
   Answer + retrieved source filenames
```

The document index is reused on later runs. Each question still
requires an embedding request and an answer-generation request.

## Installation

### 1. Get the project

Clone this repository using its GitHub URL, or download and extract
the ZIP file.

Open a terminal in the project folder.

### 2. Install dependencies

Run this command in your Python environment:

```bash
pip install openai python-dotenv numpy faiss-cpu pypdf python-docx
```

### 3. Configure your API key

Create a `.env` file beside `rag.py`:

```env
OPENAI_API_KEY=your_openai_api_key_here
```

Replace the placeholder with your own OpenAI API key.

Do not commit your real `.env` file to GitHub.

You can include a `.env.example` file containing the placeholder
above so others know which variable to configure.

### 4. Add your documents

Create a `data` folder beside `rag.py`, then add supported files:

```text
data/
├── notes.txt
├── handbook.pdf
└── report.docx
```

Files inside subfolders are also processed.

### 5. Run the application

```bash
python rag.py
```

On the first run, the application:

1. Loads your documents.
2. Splits the extracted text into chunks.
3. Creates embeddings.
4. Builds the FAISS index.
5. Saves the index and chunk metadata in `vector_store/`.

On subsequent runs, it loads the saved vector store when available
and consistent with the saved chunk count.

## Example Questions

Ask questions that relate to the documents you have added:

```text
You: What are the main topics covered in these documents?
```

```text
You: Summarize the deployment steps described in the handbook.
```

```text
You: What does the report say about the project requirements?
```

Type either of the following to exit:

```text
exit
```

```text
quit
```

## Configuration

The following settings are defined in `rag.py`:

| Setting | Default | Purpose |
|---------|---------|---------|
| `CHUNK_SIZE` | `1000` | Maximum characters per chunk |
| `CHUNK_OVERLAP` | `150` | Characters repeated between adjacent chunks |
| `TOP_K` | `4` | Maximum chunks retrieved per question |
| `EMBEDDING_BATCH_SIZE` | `100` | Texts sent per embedding batch |
| `TEMPERATURE` | `0.2` | Answer-generation sampling setting |
| `TOP_P` | `1.0` | Nucleus sampling setting |
| `MAX_TOKENS` | `700` | Output token limit passed to the chat API |

Chunking is character-based, not token-based.

Embeddings are normalized before being stored in an
`IndexFlatIP` FAISS index so inner-product search corresponds
to cosine similarity.

## Updating the Document Index

Changes to documents are not detected automatically.

After adding, editing, or deleting documents:

1. Stop the application.
2. Delete the `vector_store` folder.
3. Run the application again.

```bash
python rag.py
```

Also rebuild the index if you change the embedding model or
chunking configuration.

Rebuilding sends the document chunks to OpenAI again to generate
new embeddings.

## Limitations

- Requires an internet connection and access to the configured OpenAI models.
- OpenAI API usage may incur charges.
- Document text and user questions are sent to OpenAI.
- Scanned or image-only PDFs require OCR before this app can read their text.
- DOCX extraction reads paragraphs; it does not explicitly extract tables.
- Unsupported formats such as images, spreadsheets, and `.doc` are skipped.
- The index does not automatically refresh when documents change.
- Retrieval returns the nearest chunks without a minimum similarity threshold.
- Displayed sources identify retrieved files, not verified evidence for
  every sentence in the answer.
- Answers can still contain mistakes and should be checked against
  the original documents.
- Previous questions and answers are not passed to later requests:
  this version does not implement conversation memory.

## Security and Privacy

- Never commit API keys or your real `.env` file.
- Only index documents you are authorized to send to the API.
- Do not upload private documents or generated vector-store files
  to a public repository.
- `chunks.json` contains extracted document text.
- Only load vector-store files from trusted sources.

## Suggested .gitignore

```gitignore
.env
.venv/
venv/
__pycache__/
*.pyc
vector_store/

# Keep local documents out of the repository.
data/*
!data/.gitkeep
```

If you want the empty `data` directory to appear on GitHub,
add an empty `data/.gitkeep` file.

## Project Purpose

This project demonstrates a basic RAG pipeline: document loading,
chunking, embedding generation, vector search, and context-based
answer generation.
