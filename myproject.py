import io
from pathlib import Path

import chromadb
import streamlit as st
from PIL import Image
from sentence_transformers import SentenceTransformer
import ollama


APP_TITLE = "AI Study Assistant"
DB_DIR = "./chroma_db"
COLLECTION_NAME = "study_documents"
DEFAULT_CHAT_MODEL = "gemma3:4b"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"


st.set_page_config(page_title=APP_TITLE, page_icon="📚", layout="wide")

st.title("📚 AI Study Assistant")
st.caption("Upload text documents, create a searchable knowledge base, and ask AI questions about your study material.")


@st.cache_resource
def load_embedding_model():
    return SentenceTransformer(EMBEDDING_MODEL)


@st.cache_resource
def get_chroma_collection():
    client = chromadb.PersistentClient(path=DB_DIR)
    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"description": "AI Study Assistant text document chunks"},
    )


embedding_model = load_embedding_model()
collection = get_chroma_collection()


def chunk_text(text, chunk_size=900, overlap=150):
    text = " ".join(text.split())

    if not text:
        return []

    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        if end >= len(text):
            break

        start = end - overlap

    return chunks


def add_text_document(uploaded_file):
    """Read a TXT document, split it, embed it, and store it."""
    raw_bytes = uploaded_file.getvalue()

    # UTF-8 is preferred; fall back to common encodings.
    try:
        text = raw_bytes.decode("utf-8")
    except UnicodeDecodeError:
        try:
            text = raw_bytes.decode("utf-16")
        except UnicodeDecodeError:
            text = raw_bytes.decode("latin-1")

    chunks = chunk_text(text)

    if not chunks:
        return 0

    metadata = []
    for i, chunk in enumerate(chunks, start=1):
        metadata.append(
            {
                "source": uploaded_file.name,
                "chunk": i,
            }
        )

    embeddings = embedding_model.encode(
        chunks,
        normalize_embeddings=True,
        show_progress_bar=False,
    ).tolist()

    safe_name = (
        uploaded_file.name.replace(" ", "_")
        .replace("/", "_")
        .replace("\\", "_")
    )

    ids = [
        f"{safe_name}_{i}_{abs(hash(chunk))}"
        for i, chunk in enumerate(chunks)
    ]

    collection.upsert(
        ids=ids,
        documents=chunks,
        embeddings=embeddings,
        metadatas=metadata,
    )

    return len(chunks)


def retrieve_context(question, top_k=5):
    if collection.count() == 0:
        return []

    query_embedding = embedding_model.encode(
        [question],
        normalize_embeddings=True,
    ).tolist()[0]

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=min(top_k, collection.count()),
        include=["documents", "metadatas", "distances"],
    )

    documents = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]
    distances = results.get("distances", [[]])[0]

    return [
        {
            "text": doc,
            "metadata": metadata,
            "distance": distance,
        }
        for doc, metadata, distance in zip(
            documents, metadatas, distances
        )
    ]


def build_context(retrieved_chunks):
    if not retrieved_chunks:
        return "No relevant study material was found."

    parts = []

    for i, item in enumerate(retrieved_chunks, start=1):
        source = item["metadata"].get("source", "Unknown")
        chunk_number = item["metadata"].get("chunk", "?")

        parts.append(
            f"[Source {i}: {source}, chunk {chunk_number}]\n"
            f"{item['text']}"
        )

    return "\n\n".join(parts)


def ask_ollama(question, context, model_name):
    system_prompt = """
You are an AI Study Assistant.

Answer the student's question using the supplied study material.

Rules:
1. Prefer the supplied study material over general knowledge.
2. If the answer is not present in the material, say:
   "I couldn't find this in the uploaded study material."
3. Do not invent facts.
4. Explain difficult concepts in simple student-friendly language.
5. Use bullets or numbered steps when useful.
6. Keep the answer focused.
7. Mention the source document when relevant.
"""

    user_prompt = f"""
STUDY MATERIAL:
{context}

STUDENT QUESTION:
{question}
"""

    response = ollama.chat(
        model=model_name,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )

    return response["message"]["content"]


def generate_quiz(topic, context, model_name, number_of_questions, difficulty):
    prompt = f"""
Create a multiple-choice quiz for a college student.

Topic: {topic}
Difficulty: {difficulty}
Number of questions: {number_of_questions}

Use ONLY the supplied study material.

For every question provide:
- Question
- A
- B
- C
- D
- Correct answer
- Short explanation

Study material:
{context}
"""

    response = ollama.chat(
        model=model_name,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a careful academic quiz generator. "
                    "Do not invent information absent from the study material."
                ),
            },
            {"role": "user", "content": prompt},
        ],
    )

    return response["message"]["content"]


def analyze_image(image_bytes, question, model_name):
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    max_dimension = 1600
    if max(image.size) > max_dimension:
        scale = max_dimension / max(image.size)
        image = image.resize(
            (
                int(image.size[0] * scale),
                int(image.size[1] * scale),
            )
        )

    buffer = io.BytesIO()
    image.save(buffer, format="PNG")

    response = ollama.chat(
        model=model_name,
        messages=[
            {
                "role": "user",
                "content": question,
                "images": [buffer.getvalue()],
            }
        ],
    )

    return response["message"]["content"]


with st.sidebar:
    st.header("⚙️ Settings")

    chat_model = st.text_input(
        "Ollama model",
        value=DEFAULT_CHAT_MODEL,
    )

    top_k = st.slider(
        "Number of retrieved chunks",
        2,
        8,
        5,
    )

    st.divider()

    st.subheader("📊 Knowledge Base")
    st.metric("Stored chunks", collection.count())

    if st.button("🗑️ Clear knowledge base", use_container_width=True):
        client = chromadb.PersistentClient(path=DB_DIR)
        try:
            client.delete_collection(COLLECTION_NAME)
        except Exception:
            pass

        st.cache_resource.clear()
        st.success("Knowledge base cleared.")
        st.rerun()

    st.divider()

    st.info(
        "Text documents are processed locally. "
        "The knowledge base is stored in ChromaDB."
    )


tab1, tab2, tab3, tab4 = st.tabs(
    [
        "📄 Upload Text",
        "💬 Ask AI",
        "📝 Generate Quiz",
        "🖼️ Ask About Image",
    ]
)


with tab1:
    st.header("Upload Study Material")

    text_files = st.file_uploader(
        "Upload one or more text documents",
        type=["txt"],
        accept_multiple_files=True,
    )

    if st.button("📥 Add Text Documents", type="primary"):
        if not text_files:
            st.warning("Please upload at least one .txt document.")
        else:
            total_chunks = 0

            with st.spinner("Reading, embedding, and storing your documents..."):
                for text_file in text_files:
                    try:
                        chunks_added = add_text_document(text_file)
                        total_chunks += chunks_added
                    except Exception as exc:
                        st.error(
                            f"Could not process {text_file.name}: {exc}"
                        )

            st.success(
                f"Added {total_chunks} text chunks to the knowledge base."
            )

    st.divider()

    st.subheader("How it works")

    st.markdown(
        """
        **Text document → Text extraction → Chunking → Embeddings → ChromaDB**

        Then:

        **Question → Embedding → Relevant chunks → Ollama → AI answer**
        """
    )


with tab2:
    st.header("Ask Questions")

    if collection.count() == 0:
        st.info("Upload a text document first.")
    else:
        question = st.text_area(
            "Ask something about your uploaded material",
            placeholder="Example: Explain this topic in simple words.",
            height=120,
        )

        if st.button("🤖 Ask AI", type="primary"):
            if not question.strip():
                st.warning("Please enter a question.")
            else:
                with st.spinner("Searching your study material..."):
                    retrieved = retrieve_context(
                        question,
                        top_k=top_k,
                    )

                context = build_context(retrieved)

                with st.spinner("Ollama is generating the answer..."):
                    try:
                        answer = ask_ollama(
                            question,
                            context,
                            chat_model,
                        )

                        st.subheader("Answer")
                        st.markdown(answer)

                        st.divider()
                        st.subheader("📚 Sources Used")

                        for item in retrieved:
                            source = item["metadata"].get(
                                "source", "Unknown"
                            )
                            chunk_number = item["metadata"].get(
                                "chunk", "?"
                            )

                            with st.expander(
                                f"{source} — Chunk {chunk_number}"
                            ):
                                st.write(item["text"])

                    except Exception as exc:
                        st.error(
                            "Ollama could not answer the question.\n\n"
                            f"Error: {exc}\n\n"
                            "Make sure Ollama is running and the selected "
                            "model is installed."
                        )


with tab3:
    st.header("Generate AI Quiz")

    if collection.count() == 0:
        st.info("Upload a text document first.")
    else:
        quiz_topic = st.text_input(
            "Quiz topic",
            placeholder="Example: Semiconductors",
        )

        col1, col2 = st.columns(2)

        with col1:
            difficulty = st.selectbox(
                "Difficulty",
                ["Easy", "Medium", "Hard"],
            )

        with col2:
            number_of_questions = st.selectbox(
                "Number of questions",
                [5, 10, 15],
            )

        if st.button("📝 Generate Quiz", type="primary"):
            if not quiz_topic.strip():
                st.warning("Enter a quiz topic.")
            else:
                with st.spinner("Finding relevant study material..."):
                    retrieved = retrieve_context(
                        quiz_topic,
                        top_k=top_k,
                    )

                context = build_context(retrieved)

                with st.spinner("Generating quiz with Ollama..."):
                    try:
                        quiz = generate_quiz(
                            quiz_topic,
                            context,
                            chat_model,
                            number_of_questions,
                            difficulty,
                        )

                        st.subheader("Your Quiz")
                        st.markdown(quiz)

                    except Exception as exc:
                        st.error(f"Quiz generation failed: {exc}")


with tab4:
    st.header("Ask AI About an Image")

    st.warning(
        "This feature requires a vision-capable Ollama model."
    )

    image_file = st.file_uploader(
        "Upload an image",
        type=["png", "jpg", "jpeg", "webp"],
    )

    image_question = st.text_area(
        "What do you want to know about the image?",
        value="Explain the content of this image in simple terms.",
        height=100,
    )

    if image_file:
        image = Image.open(image_file)
        st.image(
            image,
            caption="Uploaded image",
            use_container_width=True,
        )

    if st.button("🖼️ Analyze Image", type="primary"):
        if not image_file:
            st.warning("Please upload an image.")
        elif not image_question.strip():
            st.warning("Please enter a question.")
        else:
            try:
                with st.spinner("Analyzing image with Ollama..."):
                    result = analyze_image(
                        image_file.getvalue(),
                        image_question,
                        chat_model,
                    )

                st.subheader("AI Analysis")
                st.markdown(result)

            except Exception as exc:
                st.error(
                    f"Image analysis failed: {exc}\n\n"
                    "Use an Ollama model that supports images."
                )


st.divider()
st.caption(
    "AI Study Assistant • Streamlit + Sentence Transformers + "
    "ChromaDB + Ollama + Pillow"
)
