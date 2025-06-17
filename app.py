import os
import json
import hashlib
import time
import numpy as np
import faiss
import openai
import tarfile
import zipfile
import threading
from flask import Flask, render_template, request, redirect, url_for, jsonify, current_app
from werkzeug.utils import secure_filename
from werkzeug.exceptions import RequestEntityTooLarge
from smart_loader import smart_load

# --- The New KnowledgeBase Class ---
class KnowledgeBase:
    def __init__(self, vector_file='chunks_with_vectors.json', index_file='faiss_index.faiss'):
        self.vector_file = vector_file
        self.index_file = index_file
        self.dim = 1536  # Default for text-embedding-3-small

        self.chunks = []
        self.faiss_index = faiss.IndexFlatL2(self.dim)
        self.existing_ids = set()

        self.load_from_disk()

    def load_from_disk(self):
        if os.path.exists(self.vector_file) and os.path.exists(self.index_file):
            print("Loading existing data from disk...")
            with open(self.vector_file, 'r', encoding='utf-8') as f:
                self.chunks = json.load(f)
            self.faiss_index = faiss.read_index(self.index_file)
            self.existing_ids = {c['id'] for c in self.chunks}
            print(f"Loaded {len(self.chunks)} chunks and a FAISS index with {self.faiss_index.ntotal} vectors.")
        else:
            print("No existing data found. Starting with a fresh state.")

    def persist_to_disk(self):
        print("Persisting data to disk...")
        with open(self.vector_file, 'w', encoding='utf-8') as f:
            json.dump(self.chunks, f, indent=2)
        faiss.write_index(self.faiss_index, self.index_file)
        print("Data persisted successfully.")

    def add_documents(self, file_paths, progress_status, task_id):
        client = openai.OpenAI()
        all_raw_chunks = []
        for file_path in file_paths:
            if not os.path.isfile(file_path): continue
            filename = os.path.basename(file_path)
            progress_status[task_id] = {"status": "processing", "filename": f"Parsing {filename}...", "total": 0, "current": 0}
            parsed_content = smart_load(file_path)
            for text_chunk in parsed_content:
                chunk_id = hashlib.sha256(text_chunk.encode('utf-8')).hexdigest()
                if chunk_id not in self.existing_ids:
                    all_raw_chunks.append({"id": chunk_id, "text": text_chunk, "source_file": filename})

        total_chunks = len(all_raw_chunks)
        progress_status[task_id]["total"] = total_chunks
        if total_chunks == 0:
            progress_status[task_id] = {"status": "done", "added": 0}
            return

        for i in range(0, total_chunks, 50):
            batch_raw_chunks = all_raw_chunks[i:i + 50]
            batch_texts = [chunk["text"] for chunk in batch_raw_chunks]
            progress_status[task_id]["filename"] = f"Embedding batch {i // 50 + 1}..."
            progress_status[task_id]["current"] = i
            try:
                response = client.embeddings.create(model='text-embedding-3-small', input=batch_texts)
                batch_embeddings = [item.embedding for item in response.data]
                self.faiss_index.add(np.array(batch_embeddings, dtype='float32'))
                for j, raw_chunk in enumerate(batch_raw_chunks):
                    self.chunks.append({
                        "id": raw_chunk['id'], "content": raw_chunk['text'],
                        "source_file": raw_chunk['source_file'], "embedding": batch_embeddings[j]
                    })
                    self.existing_ids.add(raw_chunk['id'])
            except Exception as e:
                print(f"Error embedding batch: {e}")
            time.sleep(1)

        progress_status[task_id]["current"] = total_chunks
        self.persist_to_disk()
        progress_status[task_id] = {"status": "done", "added": total_chunks}

    def reset(self):
        print("--- Resetting Knowledge Base ---")
        if os.path.exists(self.vector_file): os.remove(self.vector_file)
        if os.path.exists(self.index_file): os.remove(self.index_file)
        self.chunks.clear()
        self.existing_ids.clear()
        self.faiss_index.reset()
        print("--- Environment has been reset ---")

    def get_status(self):
        def get_file_info(filepath):
            if os.path.exists(filepath):
                size_in_kb = round(os.path.getsize(filepath) / 1024, 2)
                return "Exists", f"{size_in_kb} KB"
            return "Missing", "N/A"

        json_status, json_size = get_file_info(self.vector_file)
        faiss_status, faiss_size = get_file_info(self.index_file)

        return {
            "json_file_status": json_status, "json_file_size": json_size,
            "faiss_file_status": faiss_status, "faiss_file_size": faiss_size,
            "chunks_in_memory": len(self.chunks), "vectors_in_index": self.faiss_index.ntotal,
        }

# --- App Factory ---
def create_app():
    app = Flask(__name__)
    app.config['UPLOAD_FOLDER'] = 'uploads'
    app.config['BACKUP_FOLDER'] = 'backups'
    # NEW: Set max upload size to 100MB
    app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    os.makedirs(app.config['BACKUP_FOLDER'], exist_ok=True)

    # Create and attach the single KnowledgeBase instance
    app.kb = KnowledgeBase()

    # Import and register blueprints
    from ask_route import ask_bp
    from admin_route import admin_bp
    app.register_blueprint(ask_bp)
    app.register_blueprint(admin_bp)

    return app

app = create_app()

# --- Background Task Runner ---
progress_status = {}
def background_task_wrapper(app_context, file_paths, task_id):
    with app_context:
        current_app.kb.add_documents(file_paths, progress_status, task_id)

# --- Routes ---
@app.route('/')
def index():
    source_files = sorted(list(set(c.get('source_file', 'unknown') for c in current_app.kb.chunks)))
    return render_template('workbench.html', source_files=source_files)

@app.route('/process', methods=['POST'])
def process_files():
    uploaded_files = request.files.getlist('files')
    if not uploaded_files or uploaded_files[0].filename == '':
        return jsonify({"error": "No files selected"}), 400

    task_id = hashlib.md5(str(time.time()).encode()).hexdigest()
    saved_paths = []
    for file in uploaded_files:
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        saved_paths.append(file_path)

    thread = threading.Thread(target=background_task_wrapper, args=(app.app_context(), saved_paths, task_id))
    thread.start()

    return jsonify({"task_id": task_id})

@app.route('/progress_status/<task_id>')
def progress_status_endpoint(task_id):
    return jsonify(progress_status.get(task_id, {}))

# NEW: Custom error handler for oversized files
@app.errorhandler(413)
@app.errorhandler(RequestEntityTooLarge)
def handle_request_entity_too_large(e):
    max_size_mb = app.config.get('MAX_CONTENT_LENGTH', 0) / (1024 * 1024)
    return jsonify(error=f"File upload failed. The file is larger than the configured limit of {max_size_mb:.0f} MB."), 413


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5013, debug=False)
