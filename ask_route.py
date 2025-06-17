from flask import Blueprint, render_template, request, current_app, Response
from openai import OpenAI
import numpy as np
import tiktoken
import json
import time
from markdown_it import MarkdownIt

ask_bp = Blueprint('ask', __name__)

client = OpenAI()
md = MarkdownIt()

# Helper functions that don't depend on request context
def build_strict_prompt(query, results):
    context = ""
    for r in results:
        context += f"--- Source: {r.get('source_file', 'unknown')} ---\n{r['content']}\n\n"
    prompt = (
        "You are an expert synthesizer... If the context does not contain the information... state clearly: 'Based on the provided context, I cannot answer this question.'\n\n"
        f"CONTEXT:\n{context}\n\nQUERY:\n{query}"
        "IMPORTANT: Do not include any ethical warnings, disclaimers, or your own modern commentary. Stick strictly to the information in the provided context."
    )
    return prompt

def build_reasoning_prompt(query, results):
    context = ""
    for r in results:
        context += f"--- Source: {r.get('source_file', 'unknown')} ---\n{r['content']}\n\n"
    prompt = (
        "You are an expert AI synthesizer and analyst... please **infer and explain** the answer to the user's query.\n\n"
        f"CONTEXT:\n{context}\n\nQUERY:\n{query}"
        "IMPORTANT: Do not include any ethical warnings, disclaimers, or your own modern commentary. Stick strictly to the information in the provided context."
    )
    return prompt

def is_unhelpful_answer(answer):
    unhelpful_phrases = ["cannot answer", "can't answer", "does not contain"]
    return any(phrase in answer.lower() for phrase in unhelpful_phrases)

@ask_bp.route('/ask-stream')
def ask_stream():
    kb = current_app.kb
    query = request.args.get('query', '')
    model = request.args.get('model', 'gpt-4-turbo')

    # --- MODIFICATION: Define embed_query locally within the route ---
    def embed_query(text):
        response = client.embeddings.create(input=[text], model='text-embedding-3-small')
        return response.data[0].embedding

    def stream_response():
        if not query or kb.faiss_index.ntotal == 0:
            yield f"data: {json.dumps({'token': 'Knowledge base is empty or no query provided.'})}\n\n"
            yield f"data: {json.dumps({'end_of_stream': True})}\n\n"
            return

        try:
            # ATTEMPT 1: Standard RAG
            query_vector = embed_query(query)
            D, I = kb.faiss_index.search(np.array([query_vector]), 5)
            results = [kb.chunks[i] for i in I[0] if i != -1]
            prompt = build_strict_prompt(query, results)

            response = client.chat.completions.create(model=model, messages=[{"role": "user", "content": prompt}], temperature=0.1)
            initial_answer = response.choices[0].message.content

            if not is_unhelpful_answer(initial_answer):
                final_answer = initial_answer
                for char in final_answer:
                    yield f"data: {json.dumps({'token': char})}\n\n"
                    time.sleep(0.005)
            else:
                # FALLBACK: Reasoning Agent
                yield f"data: {json.dumps({'status': 'Preparing a detailed response for you...'})}\n\n"
                time.sleep(1)
                reasoning_prompt = build_reasoning_prompt(query, results)
                stream = client.chat.completions.create(model=model, messages=[{"role": "user", "content": reasoning_prompt}], temperature=0.4, stream=True)

                streamed_response = ""
                for chunk in stream:
                    content = chunk.choices[0].delta.content or ""
                    streamed_response += content
                    yield f"data: {json.dumps({'token': content})}\n\n"
                    time.sleep(0.01)
                final_answer = streamed_response

            yield f"data: {json.dumps({'sources_data': results})}\n\n"

        except Exception as e:
            print(f"Error during streaming: {e}")
            yield f"data: {json.dumps({'error': str(e)})}\n\n"

        yield f"data: {json.dumps({'end_of_stream': True})}\n\n"

    return Response(stream_response(), mimetype='text/event-stream')

@ask_bp.route('/ask')
def ask_page():
    return render_template('workbench.html', source_files=sorted(list(set(c.get('source_file', 'unknown') for c in current_app.kb.chunks))))
