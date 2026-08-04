"""
Multimodal Soccer Match Event Retrieval - Streamlit Application

Streamlit web UI for natural-language soccer event search with LangChain RAG & video playback.
"""

import os
import yaml
import json
import streamlit as st
from pathlib import Path
import plotly.graph_objects as go

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

    return events, vector_store, docs


@st.cache_data
def load_tracking_data_cached(path):
    """Caches tracking data load in memory to ensure visualizer is fast."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def draw_tactical_minimap(tracking_path, start_sec, end_sec):
    """Renders a beautiful 2D tactical pitch overlay with trajectories using Plotly."""
    try:
        data = load_tracking_data_cached(tracking_path)
    except Exception as e:
        st.error(f"Failed to load tracking data for visualizer: {e}")
        return

    frames = data.get("frames", [])
    
    # Filter frames within start and end time window
    clip_frames = [f for f in frames if start_sec <= f["timestamp_sec"] <= end_sec]
    if not clip_frames:
        st.warning("No coordinates history available for this clip time window.")
        return
    
    last_frame = clip_frames[-1]
    fig = go.Figure()
    
    # 1. Draw Green Pitch Background and Boundaries
    fig.add_shape(type="rect", x0=0, y0=0, x1=105, y1=68, fillcolor="#1E391E", opacity=1.0, line=dict(color="white", width=2))
    # Center Line
    fig.add_shape(type="line", x0=52.5, y0=0, x1=52.5, y1=68, line=dict(color="white", width=2))
    # Center Circle
    fig.add_shape(type="circle", x0=52.5-9.15, y0=34-9.15, x1=52.5+9.15, y1=34+9.15, line=dict(color="white", width=2))
    # Goal Boxes
    fig.add_shape(type="rect", x0=0, y0=24.84, x1=5.5, y1=43.16, line=dict(color="white", width=1))
    fig.add_shape(type="rect", x0=99.5, y0=24.84, x1=105, y1=43.16, line=dict(color="white", width=1))
    # Penalty Boxes
    fig.add_shape(type="rect", x0=0, y0=13.84, x1=16.5, y1=54.16, line=dict(color="white", width=2))
    fig.add_shape(type="rect", x0=88.5, y0=13.84, x1=105, y1=54.16, line=dict(color="white", width=2))
    
    # 2. Extract Trajectory Coordinate Lists
    ball_path_x = []
    ball_path_y = []
    histories = {}
    
    for f in clip_frames:
        if f.get("ball"):
            ball_path_x.append(f["ball"]["x"])
            ball_path_y.append(f["ball"]["y"])
            
        for p in f.get("players", []):
            tid = p["track_id"]
            if tid not in histories:
                histories[tid] = {"x": [], "y": [], "team": p["team_id"], "number": p.get("jersey_number", "")}
            histories[tid]["x"].append(p["x"])
            histories[tid]["y"].append(p["y"])
            
    # 3. Plot faded player trails
    for tid, hist in histories.items():
        color = "rgba(248, 113, 113, 0.45)" if hist["team"] == "Team A" else "rgba(96, 165, 250, 0.45)"
        fig.add_trace(go.Scatter(
            x=hist["x"], y=hist["y"],
            mode="lines",
            line=dict(color=color, width=2),
            hoverinfo="none",
            showlegend=False
        ))
        
    # 4. Plot current positions (Last frame of selection)
    team_a_x, team_a_y, team_a_text = [], [], []
    team_b_x, team_b_y, team_b_text = [], [], []
    
    for p in last_frame.get("players", []):
        if p["team_id"] == "Team A":
            team_a_x.append(p["x"])
            team_a_y.append(p["y"])
            team_a_text.append(str(p.get("jersey_number", "")))
        else:
            team_b_x.append(p["x"])
            team_b_y.append(p["y"])
            team_b_text.append(str(p.get("jersey_number", "")))
            
    fig.add_trace(go.Scatter(
        x=team_a_x, y=team_a_y,
        mode="markers+text",
        marker=dict(color="#EF4444", size=14, line=dict(color="white", width=2)),
        text=team_a_text, textposition="middle center",
        textfont=dict(color="white", size=9, weight="bold"),
        name="Team A"
    ))
    
    fig.add_trace(go.Scatter(
        x=team_b_x, y=team_b_y,
        mode="markers+text",
        marker=dict(color="#3B82F6", size=14, line=dict(color="white", width=2)),
        text=team_b_text, textposition="middle center",
        textfont=dict(color="white", size=9, weight="bold"),
        name="Team B"
    ))
    
    # 5. Plot Ball path and position
    if ball_path_x:
        fig.add_trace(go.Scatter(
            x=ball_path_x, y=ball_path_y,
            mode="lines",
            line=dict(color="#FBBF24", width=2.5, dash="dash"),
            hoverinfo="none",
            showlegend=False
        ))
        fig.add_trace(go.Scatter(
            x=[ball_path_x[-1]], y=[ball_path_y[-1]],
            mode="markers",
            marker=dict(color="#FBBF24", size=10, line=dict(color="black", width=1.5)),
            name="Ball"
        ))
        
    fig.update_layout(
        title=dict(text="2D Tactical Minimap (Trajectories & Positions)", font=dict(color="white")),
        xaxis=dict(range=[0, 105], showgrid=False, zeroline=False, visible=False),
        yaxis=dict(range=[0, 68], showgrid=False, zeroline=False, visible=False, scaleanchor="x", scaleratio=1),
        plot_bgcolor="#1E391E",
        paper_bgcolor="#182E18",
        margin=dict(l=15, r=15, t=45, b=15),
        legend=dict(font=dict(color="white"), bgcolor="rgba(24, 46, 24, 0.75)", bordercolor="white", borderwidth=1),
        height=380
    )
    st.plotly_chart(fig, use_container_width=True)


def main():
    config = load_config()

    st.markdown('<div class="main-title">⚽ Multimodal Soccer Match Event Retrieval</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">LangChain RAG Pipeline for Game State Reconstruction & Video Timestamp Search</div>', unsafe_allow_html=True)

    # Sidebar Config & Filters
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

        st.divider()

        # Step 4: Streamlit Interactive Pitch Map Filters
        st.subheader("🎯 Sidebar Tactical Filters")
        st.caption("Combine structured filters directly with your semantic RAG query.")
        
        filter_team = st.selectbox("Filter Team", ["All Teams", "Team A", "Team B"])
        filter_zone = st.selectbox("Filter Pitch Zone", [
            "All Zones", "Defensive Third", "Midfield", "Attacking Third", 
            "Defensive Penalty Box", "Attacking Penalty Box", "Left Flank", "Right Flank"
        ])
        filter_event = st.selectbox("Filter Event Type", [
            "All Events", "completed_pass", "turnover", "high_press_sequence", 
            "proxy_offside", "shot_on_goal", "penalty_box_entry", "multi_event_narrative"
        ])

        ui_filters = {}
        if filter_team != "All Teams":
            ui_filters["team_id"] = filter_team
        if filter_zone != "All Zones":
            ui_filters["pitch_zones"] = filter_zone
        if filter_event != "All Events":
            ui_filters["event_type"] = filter_event

        st.divider()

        if st.button("🔄 Re-generate Synthetic Data"):
            generate_synthetic_match_data()
            st.cache_resource.clear()
            st.rerun()

    # Load Pipeline Data, Vector Store, and raw docs
    with st.spinner("Indexing match events into Chroma vector store..."):
        events, vector_store, docs = initialize_pipeline(config)

    # Step 2: Initialize Hybrid Sparse-Dense Retriever (Chroma + BM25)
    retriever = EventRetriever(vector_store=vector_store, documents=docs, top_k=top_k)
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
                # Retrieve using hybrid search + metadata extraction + sidebar interactive filters
                docs = retriever.retrieve(user_query, ui_filters=ui_filters)
                
                # Format context manually for prompt
                formatted_context = qa_chain._format_docs(docs)
                try:
                    answer = qa_chain.chain.invoke({"context": formatted_context, "question": user_query})
                    retrieved_docs = docs
                except Exception as e:
                    err_msg = str(e)
                    qa_chain.active_provider_name = f"{qa_chain.provider.upper()} (Unavailable: {err_msg[:40]}...) ➔ Fallback: Local Analyst"
                    fallback_llm = SoccerQAChain(retriever=retriever, provider="local").chain
                    answer = fallback_llm.invoke({"context": formatted_context, "question": user_query})
                    retrieved_docs = docs

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

                # Step 1 & 5: Render 2D Tactical minimap chart next to the video
                st.divider()
                draw_tactical_minimap(config["paths"]["tracking_data"], start_sec, end_sec)


if __name__ == "__main__":
    main()
