"""
Multimodal Soccer Match Event Retrieval - Streamlit Application

Streamlit web UI for natural-language soccer event search with LangChain RAG & video playback.
"""

import os
import yaml
import streamlit as st
from pathlib import Path

# Project modules
from src.event_extraction.parser import parse_tracking_json
from src.event_extraction.preprocessor import TrackingPreprocessor
from src.event_extraction.heuristics import EventExtractor
from src.rag.chunker import EventAdaptiveChunker
from src.rag.vector_store import VectorStoreManager
from src.rag.retriever import EventRetriever
from src.rag.qa_chain import SoccerQAChain
from src.video.ffmpeg_clipper import VideoClipper
from src.utils.mock_data_gen import generate_synthetic_match_data

# Set Page Layout & Config
st.set_page_config(
    page_title="Soccer Event Retrieval RAG",
    page_icon="⚽",
    layout="wide"
)

# Custom Styling CSS
st.markdown("""
<style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E293B;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.0rem;
        color: #64748B;
        margin-bottom: 1.5rem;
    }
    .badge-synthetic {
        background-color: #FEF3C7;
        color: #92400E;
        padding: 4px 10px;
        border-radius: 6px;
        font-weight: 600;
        font-size: 0.85rem;
    }
    .badge-real {
        background-color: #D1FAE5;
        color: #065F46;
        padding: 4px 10px;
        border-radius: 6px;
        font-weight: 600;
        font-size: 0.85rem;
    }
    .event-card {
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        padding: 12px;
        margin-bottom: 10px;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_config():
    with open("config/settings.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


@st.cache_resource
def initialize_pipeline(config):
    """Initializes tracking parser, event extraction, chunker, and vector DB."""
    tracking_path = config["paths"]["tracking_data"]
    video_path = config["paths"]["video_file"]

    # Ensure synthetic sample data exists if files are missing
    if not Path(tracking_path).exists() or not Path(video_path).exists():
        st.info("Generating synthetic match data & video...")
        generate_synthetic_match_data(
            output_tracking_path=tracking_path,
            output_ground_truth_path=config["paths"]["ground_truth_events"],
            output_video_path=video_path
        )

    # 1. Parse tracking data
    frames = parse_tracking_json(tracking_path)

    # 2. Preprocess & Noise Filter
    preprocessor = TrackingPreprocessor()
    frames = preprocessor.preprocess(frames)
    possessions = preprocessor.compute_smoothed_possessions(frames)
    pressing = preprocessor.compute_smoothed_pressing_states(frames, possessions)

    # 3. Rule-Based Event Extraction
    extractor = EventExtractor(pitch_config_path=config["paths"]["pitch_config"])
    events = extractor.extract_events(frames, possessions, pressing)

    # 4. Adaptive Semantic Chunking
    chunker = EventAdaptiveChunker()
    docs = chunker.chunk_events(events)

    # 5. Chroma Vector Store
    vstore_mgr = VectorStoreManager(
        persist_directory=config["paths"]["chroma_db_dir"],
        collection_name=config["rag"]["vector_store_collection"],
        embedding_model_name=config["rag"]["embedding_model"]
    )
    vector_store = vstore_mgr.index_documents(docs, reset=True)

    return events, vector_store


def main():
    config = load_config()

    st.markdown('<div class="main-title">⚽ Multimodal Soccer Match Event Retrieval</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">LangChain RAG Pipeline for Game State Reconstruction & Video Timestamp Search</div>', unsafe_allow_html=True)

    # Sidebar Config & Provenance
    with st.sidebar:
        st.header("⚙️ Configuration")
        data_src = config.get("data_source", "synthetic")

        if data_src == "synthetic":
            st.markdown('<span class="badge-synthetic">DATA SOURCE: SYNTHETIC</span>', unsafe_allow_html=True)
            st.caption("Generated via `mock_data_gen.py` for testing & demo.")
        else:
            st.markdown('<span class="badge-real">DATA SOURCE: REAL GSR</span>', unsafe_allow_html=True)
            st.caption("Using real camera tracking output.")

        st.divider()

        st.subheader("LLM & Retrieval Settings")
        provider = st.selectbox("LLM Provider", ["local", "huggingface", "ollama", "openai"], index=0)
        top_k = st.slider("Top-K Events to Retrieve", min_value=1, max_value=8, value=4)

        if st.button("🔄 Re-generate Synthetic Data"):
            generate_synthetic_match_data()
            st.cache_resource.clear()
            st.rerun()

    # Load Pipeline Data & Vector Store
    with st.spinner("Indexing match events into Chroma vector store..."):
        events, vector_store = initialize_pipeline(config)

    retriever = EventRetriever(vector_store=vector_store, top_k=top_k)
    qa_chain = SoccerQAChain(retriever=retriever, provider=provider)
    clipper = VideoClipper(output_dir=config["video"]["output_clip_dir"])

    # Query Section
    st.subheader("🔍 Ask a Natural Language Question about the Match")

    # Sample query buttons
    col_q1, col_q2, col_q3, col_q4 = st.columns(4)
    preset_query = None
    if col_q1.button("📌 Penalty Box Entries"):
        preset_query = "Show me when Team A crossed into the penalty box"
    if col_q2.button("⚠️ Turnovers"):
        preset_query = "Find turnovers or lost possession in the defensive third"
    if col_q3.button("🔥 High Pressing"):
        preset_query = "When was Team A subjected to high pressing sequence?"
    if col_q4.button("🚩 Offside Situations"):
        preset_query = "Find offside situations involving Player 9"

    user_query = st.text_input("Enter query:", value=preset_query or "", placeholder="e.g. Find all completed passes in the penalty box")

    if user_query:
        st.divider()
        col_left, col_right = st.columns([1.1, 1.0])

        with col_left:
            st.subheader("🤖 Tactical Answer & Timestamps")
            st.caption(f"Active LLM Engine: **{qa_chain.active_provider_name}**")
            with st.spinner("Synthesizing answer with LangChain LCEL..."):
                answer, retrieved_docs = qa_chain.answer_question(user_query)

            st.write(answer)

            st.subheader("📋 Retrieved Event Metadata")
            for doc in retrieved_docs:
                meta = doc.metadata
                with st.expander(f"⏱️ [{meta.get('start_timestamp_str')} - {meta.get('end_timestamp_str')}] {meta.get('event_type').upper()}"):
                    st.write(f"**Team**: {meta.get('team_id')}")
                    st.write(f"**Pitch Zones**: {meta.get('pitch_zones')}")
                    st.write(f"**Details**: {meta.get('raw_description')}")

        with col_right:
            st.subheader("🎬 Match Video Clip Playback")
            if retrieved_docs:
                top_meta = retrieved_docs[0].metadata
                start_sec = top_meta.get("start_timestamp", 0.0)
                end_sec = top_meta.get("end_timestamp", 10.0)
                ts_str = f"{top_meta.get('start_timestamp_str')} - {top_meta.get('end_timestamp_str')}"

                st.info(f"Displaying sub-clip for top retrieved event: **{ts_str}**")

                video_path = config["paths"]["video_file"]
                clip_path = clipper.extract_clip(video_path, start_sec, end_sec)

                if clip_path and Path(clip_path).exists():
                    st.video(clip_path)
                else:
                    st.warning("Video file not found or clip extraction pending.")


if __name__ == "__main__":
    main()
