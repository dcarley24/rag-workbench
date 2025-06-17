import json
import time
from datetime import datetime
import hashlib

INPUT_FILE = "conversations.json"
OUTPUT_FILE = "tagged_chunks.json"

# Define cutoff timestamp: April 1, 2024
CUTOFF_TIMESTAMP = time.mktime(datetime(2024, 4, 1).timetuple())

def tag_chunk(role, content):
    if role == "user":
        return ["user_message"]
    elif role == "assistant":
        return ["assistant_message"]
    else:
        return ["unknown"]

def generate_chunk_id(text):
    return hashlib.md5(text.encode("utf-8")).hexdigest()

def extract_chunks(conversations):
    chunks = []
    for convo in conversations:
        title = convo.get("title", "untitled")
        mapping = convo.get("mapping", {})
        for node in mapping.values():
            message = node.get("message")
            if not message:
                continue

            role = message.get("author", {}).get("role")
            raw_parts = message.get("content", {}).get("parts", [])
            content_parts = [p for p in raw_parts if isinstance(p, str)]
            content = "\n".join(content_parts).strip()

            ts = message.get("create_time") or message.get("update_time")

            if not content or not ts or ts < CUTOFF_TIMESTAMP:
                continue

            chunk = {
                "id": generate_chunk_id(content + str(ts)),
                "role": role,
                "tags": tag_chunk(role, content),
                "content": content,
                "timestamp": ts,
                "title": title
            }
            chunks.append(chunk)
    return chunks

def main():
    with open(INPUT_FILE, "r") as f:
        conversations = json.load(f)

    print("Parsing and tagging conversations...")
    chunks = extract_chunks(conversations)
    print(f"Saved {len(chunks)} tagged chunks to {OUTPUT_FILE}")

    with open(OUTPUT_FILE, "w") as f:
        json.dump(chunks, f, indent=2)

if __name__ == "__main__":
    main()
