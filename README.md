# RAG Workbench

RAG Workbench is a personal, self-hosted search engine that allows you to "chat" with your documents. It uses a Retrieval-Augmented Generation (RAG) pipeline to provide accurate, context-aware answers to your questions based on the content of your uploaded files.

The application is built with a Python/Flask backend and a clean, responsive interface using Tailwind CSS.

![Workbench Screenshot](https://raw.githubusercontent.com/dcarley24/rag-workbench/main/screenshot.jpg)

## Key Features

- **Document Upload:** Supports various file types including `.pdf`, `.docx`, `.txt`, and archives (`.zip`, `.tar.gz`).
- **Vector-Based Knowledge Base:** Automatically parses, chunks, and indexes your documents into a FAISS vector store for fast and semantic searching.
- **Interactive Q&A:** Ask questions in natural language through a clean, modern workbench interface.
- **Streamed Responses:** Answers from the AI model are streamed in real-time for an interactive, "live" experience.
- **Source Verification:** Easily view the exact source chunks from your documents that were used to generate an answer.
- **Two-Step Reasoning:** Uses a robust process that first attempts a direct answer and then falls back to a more advanced reasoning model for complex or inferential questions.
- **Admin Panel:** A simple interface to monitor the status of the knowledge base, download backups, or reset the environment.

## Technology Stack

- **Backend:** Python, Flask
- **Frontend:** HTML, Tailwind CSS, Vanilla JavaScript
- **AI/ML:**
    - OpenAI API for embeddings (`text-embedding-3-small`) and chat completions (`gpt-4-turbo`).
    - Facebook AI Similarity Search (FAISS) for efficient vector searching.
- **WSGI Server:** Can be run with Flask's built-in development server or a production-grade server like Waitress.

## Setup and Installation

Follow these steps to get the RAG Workbench running on your local machine.

### 1. Prerequisites

- Python 3.8+
- An OpenAI API Key

### 2. Clone the Repository

```bash
git clone [https://github.com/dcarley24/rag-workbench.git](https://github.com/dcarley24/rag-workbench.git)
cd rag-workbench
```

### 3. Create a Virtual Environment

It's highly recommended to use a virtual environment to manage dependencies.

```bash
# Create the virtual environment
python3 -m venv venv

# Activate it (on macOS/Linux)
source venv/bin/activate

# On Windows, use:
# venv\Scripts\activate
```

### 4. Install Dependencies

First, create a `requirements.txt` file in the root of your project directory with the following content:

```
# requirements.txt
flask
openai
faiss-cpu
numpy
werkzeug
tiktoken
markdown-it-py
python-dotenv
```
*Note: `faiss-cpu` is recommended for general compatibility. If you have a CUDA-enabled GPU, you can use `faiss-gpu` instead.*

Now, install these packages using pip:

```bash
pip install -r requirements.txt
```

### 5. Configure Environment Variables

Create a file named `.env` in the root of your project directory. This file will hold your secret API key.

Add your OpenAI API key to the `.env` file:

```
OPENAI_API_KEY="sk-YourSecretKeyGoesHere"
```
The application will automatically load this key at startup.

## Running the Application

Once the setup is complete, you can start the Flask server.

```bash
python app.py
```

You should see output indicating that the server is running. You can now access the RAG Workbench in your web browser, typically at:

**[http://127.0.0.1:5013](http://127.0.0.1:5013)**

## How to Use

1.  **Upload Documents:** In the left-hand panel, click on "Upload & Process" to expand the form. Select one or more documents or a single archive file and click "Upload and Process".
2.  **Wait for Processing:** A progress bar will appear, showing the status of the parsing and embedding process. For large files, this may take a few moments. The page will reload automatically when finished.
3.  **Ask Questions:** Once your documents are indexed, type a question into the text area and click "Ask".
4.  **View Results:** The AI-generated answer will appear in the right-hand panel. Below the answer, you can click "View Sources" to see the retrieved text chunks that provided the context.
