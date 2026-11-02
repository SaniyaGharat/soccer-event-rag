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
    page_title="Soccer Match Event Retrieval",
    page_icon="⚽",
    layout="wide"
)

# Custom Styling CSS - Pastel, Human-Crafted Editorial Theme
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
        color: #2D3436;
    }

    /* Main Container & Headings */
    .header-container {
        padding: 0.5rem 0 1.25rem 0;
        border-bottom: 1px solid #ECE7DE;
        margin-bottom: 1.5rem;
    }
    .header-tag {
        font-size: 0.72rem;
        font-weight: 600;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        color: #7E9F8E;
        margin-bottom: 0.35rem;
    }
    .main-title {
        font-size: 2.1rem;
        font-weight: 700;
        color: #2D3436;
        letter-spacing: -0.02em;
        margin: 0;
        line-height: 1.2;
    }
    .sub-title {
        font-size: 0.95rem;
        color: #718096;
        margin-top: 0.35rem;
        margin-bottom: 0;
        font-weight: 400;
    }

    /* Soft Pastel Badges */
    .badge {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 999px;
        font-size: 0.76rem;
        font-weight: 600;
        letter-spacing: 0.03em;
        text-transform: uppercase;
    }
    .badge-synthetic {
        background-color: #FDF3E7;
        color: #8C5835;
        border: 1px solid #F5DCC8;
    }
    .badge-real {
        background-color: #EBF4EE;
        color: #2E6243;
        border: 1px solid #D0E5D7;
    }

    /* Tactical Event Pills */
    .event-pill {
        display: inline-block;
        padding: 3px 10px;
        border-radius: 999px;
        font-size: 0.74rem;
        font-weight: 600;
        letter-spacing: 0.02em;
        text-transform: uppercase;
        margin-right: 6px;
    }
    .pill-team-a {
        background-color: #FDF1F1;
        color: #B25353;
        border: 1px solid #F5D2D2;
    }
    .pill-team-b {
        background-color: #EEF4FA;
        color: #436E9B;
        border: 1px solid #D2E3F3;
    }
    .pill-zone {
        background-color: #F4F2F9;
        color: #635787;
        border: 1px solid #DFDAEE;
    }
    .pill-time {
        background-color: #EDF3EF;
        color: #3B6B50;
        border: 1px solid #D0E3D6;
    }

    /* Analysis Summary Card */
    .analysis-card {
        background-color: #FFFFFF;
        border: 1px solid #ECE7DE;
        border-radius: 12px;
        padding: 16px 20px;
        margin-bottom: 16px;
        box-shadow: 0 1px 4px rgba(45, 52, 54, 0.03);
    }
    .engine-indicator {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        font-size: 0.8rem;
        color: #718096;
        margin-bottom: 12px;
    }
    .engine-dot {
        width: 7px;
        height: 7px;
        border-radius: 50%;
        background-color: #7E9F8E;
    }

    /* Pastel Preset Query Buttons */
    .stButton > button {
        background-color: #F5F2EB !important;
        color: #3B444B !important;
        border: 1px solid #E4DFD5 !important;
        border-radius: 20px !important;
        padding: 6px 14px !important;
        font-size: 0.82rem !important;
        font-weight: 500 !important;
        box-shadow: none !important;
        transition: all 0.15s ease !important;
    }
    .stButton > button:hover {
        background-color: #EBE5DB !important;
        border-color: #D3CAB9 !important;
        color: #242B2E !important;
        transform: translateY(-1px);
    }
    .stButton > button:active {
        background-color: #E2DBD0 !important;
        transform: translateY(0);
    }

    /* Clean Input & Card Styles */
    .stTextInput > div > div > input {
        background-color: #FFFFFF !important;
        border: 1px solid #E2DCD2 !important;
        border-radius: 10px !important;
        color: #2D3436 !important;
        padding: 10px 14px !important;
    }
    .stTextInput > div > div > input:focus {
        border-color: #7E9F8E !important;
        box-shadow: 0 0 0 2px rgba(126, 159, 142, 0.25) !important;
    }

    /* Sidebar Styling */
    [data-testid="stSidebar"] {
        background-color: #F5F2EB !important;
        border-right: 1px solid #E8E3D8 !important;
    }

    /* Subtle Dividers */
    hr {
        border: 0;
        height: 1px;
        background: #ECE7DE;
        margin: 1.5rem 0;
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
    """Renders a refined 2D tactical pitch overlay with soft pastel palette."""
    try:
        data = load_tracking_data_cached(tracking_path)
    except Exception as e:
        st.error(f"Failed to load tracking data for visualizer: {e}")
        return

    frames = data.get("frames", [])
    
    # Filter frames within start and end time window
    clip_frames = [f for f in frames if start_sec <= f["timestamp_sec"] <= end_sec]
    if not clip_frames:
        st.warning("No coordinate history available for this clip time window.")
        return
    
    last_frame = clip_frames[-1]
    fig = go.Figure()

    # 1. Soft Sage Green Pitch Background and Subtle White Markings
    pitch_green = "#8FA895"
    line_color = "rgba(255, 255, 255, 0.85)"
    
    fig.add_shape(type="rect", x0=0, y0=0, x1=105, y1=68, fillcolor=pitch_green, opacity=1.0, line=dict(color=line_color, width=1.5), layer="below")
    # Center Line
    fig.add_shape(type="line", x0=52.5, y0=0, x1=52.5, y1=68, line=dict(color=line_color, width=1.5), layer="below")
    # Center Circle
    fig.add_shape(type="circle", x0=52.5-9.15, y0=34-9.15, x1=52.5+9.15, y1=34+9.15, line=dict(color=line_color, width=1.5), layer="below")
    # Goal Boxes
    fig.add_shape(type="rect", x0=0, y0=24.84, x1=5.5, y1=43.16, line=dict(color=line_color, width=1.2), layer="below")
    fig.add_shape(type="rect", x0=99.5, y0=24.84, x1=105, y1=43.16, line=dict(color=line_color, width=1.2), layer="below")
    # Penalty Boxes
    fig.add_shape(type="rect", x0=0, y0=13.84, x1=16.5, y1=54.16, line=dict(color=line_color, width=1.5), layer="below")
    fig.add_shape(type="rect", x0=88.5, y0=13.84, x1=105, y1=54.16, line=dict(color=line_color, width=1.5), layer="below")
    
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
            
    # 3. Soft pastel faded player trails
    for tid, hist in histories.items():
        # Soft dusty rose for Team A, soft powder periwinkle for Team B
        trail_color = "rgba(214, 115, 115, 0.40)" if hist["team"] == "Team A" else "rgba(110, 153, 194, 0.40)"
        fig.add_trace(go.Scatter(
            x=hist["x"], y=hist["y"],
            mode="lines",
            line=dict(color=trail_color, width=2),
            hoverinfo="none",
            showlegend=False
        ))
        
    # 4. Current positions (Last frame of selection) in pastel tones
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
            
    # Pastel Dusty Rose for Team A
    fig.add_trace(go.Scatter(
        x=team_a_x, y=team_a_y,
        mode="markers+text",
        marker=dict(color="#D97777", size=13, line=dict(color="#FAF8F5", width=1.5)),
        text=team_a_text, textposition="middle center",
        textfont=dict(color="#FAF8F5", size=8, weight="bold"),
        name="Team A"
    ))
    
    # Pastel Slate Periwinkle for Team B
    fig.add_trace(go.Scatter(
        x=team_b_x, y=team_b_y,
        mode="markers+text",
        marker=dict(color="#6B96C3", size=13, line=dict(color="#FAF8F5", width=1.5)),
        text=team_b_text, textposition="middle center",
        textfont=dict(color="#FAF8F5", size=8, weight="bold"),
        name="Team B"
    ))
    
    # 5. Pastel Honey/Warm Amber Ball
    if ball_path_x:
        fig.add_trace(go.Scatter(
            x=ball_path_x, y=ball_path_y,
            mode="lines",
            line=dict(color="rgba(235, 187, 107, 0.65)", width=2, dash="dot"),
            hoverinfo="none",
            showlegend=False
        ))
        fig.add_trace(go.Scatter(
            x=[ball_path_x[-1]], y=[ball_path_y[-1]],
            mode="markers",
            marker=dict(color="#E8BE74", size=9, line=dict(color="#5C4A2A", width=1)),
            name="Ball"
        ))
        
    fig.update_layout(
        title=dict(
            text="2D Tactical Minimap · Player & Ball Tracking",
            font=dict(color="#3B444B", size=12, family="Plus Jakarta Sans, sans-serif")
        ),
        xaxis=dict(range=[0, 105], showgrid=False, zeroline=False, visible=False),
        yaxis=dict(range=[0, 68], showgrid=False, zeroline=False, visible=False, scaleanchor="x", scaleratio=1),
        plot_bgcolor="#8FA895",
        paper_bgcolor="#F5F2EB",
        margin=dict(l=12, r=12, t=38, b=12),
        legend=dict(
            font=dict(color="#3B444B", size=10),
            bgcolor="rgba(255, 255, 255, 0.85)",
            bordercolor="#E4DFD5",
            borderwidth=1
        ),
        height=360
    )
    st.plotly_chart(fig, use_container_width=True)


def main():
    config = load_config()

    st.markdown("""
    <div class="header-container">
        <div class="header-tag">Tactical Match Intelligence · Multimodal RAG</div>
        <h1 class="main-title">Soccer Match Event Retrieval</h1>
        <p class="sub-title">Natural language query retrieval across player trajectories, tactical zones, and synchronized video</p>
    </div>
    """, unsafe_allow_html=True)

    # Sidebar Config & Filters
    with st.sidebar:
        st.subheader("Configuration")
        data_src = config.get("data_source", "synthetic")

        if data_src == "synthetic":
            st.markdown('<span class="badge badge-synthetic">Data: Synthetic Demo</span>', unsafe_allow_html=True)
            st.caption("Simulated coordinates generated via `mock_data_gen.py`.")
        else:
            st.markdown('<span class="badge badge-real">Data: Real Camera GSR</span>', unsafe_allow_html=True)
            st.caption("Real camera tracking output.")

        st.divider()

        st.subheader("Retrieval Model")
        provider = st.selectbox("LLM Provider", ["local", "huggingface", "ollama", "openai"], index=0)
        top_k = st.slider("Events to Retrieve (Top-K)", min_value=1, max_value=8, value=4)

        st.divider()

        st.subheader("Tactical Filters")
        st.caption("Narrow semantic search results by team, pitch zone, or event type.")
        
        filter_team = st.selectbox("Team", ["All Teams", "Team A", "Team B"])
        filter_zone = st.selectbox("Pitch Zone", [
            "All Zones", "Defensive Third", "Midfield", "Attacking Third", 
            "Defensive Penalty Box", "Attacking Penalty Box", "Left Flank", "Right Flank"
        ])
        filter_event = st.selectbox("Event Category", [
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

        if st.button("Rebuild Synthetic Match Data"):
            generate_synthetic_match_data()
            st.cache_resource.clear()
            st.rerun()

    # Load Pipeline Data, Vector Store, and raw docs
    with st.spinner("Indexing match events into Chroma vector store..."):
        events, vector_store, docs = initialize_pipeline(config)

    # Initialize Hybrid Retriever & QA Chain
    retriever = EventRetriever(vector_store=vector_store, documents=docs, top_k=top_k)
    qa_chain = SoccerQAChain(retriever=retriever, provider=provider)
    clipper = VideoClipper(output_dir=config["video"]["output_clip_dir"])

    # Query Section
    st.markdown("#### Search Match Events")

    col_q1, col_q2, col_q3, col_q4 = st.columns(4)
    preset_query = None
    if col_q1.button("Penalty Box Entries"):
        preset_query = "Show me when Team A crossed into the penalty box"
    if col_q2.button("Defensive Turnovers"):
        preset_query = "Find turnovers or lost possession in the defensive third"
    if col_q3.button("High Press Sequences"):
        preset_query = "When was Team A subjected to high pressing sequence?"
    if col_q4.button("Offside Situations"):
        preset_query = "Find offside situations involving Player 9"

    user_query = st.text_input(
        "Enter natural language tactical query:",
        value=preset_query or "",
        placeholder="e.g. Find all completed passes in the penalty box",
        label_visibility="collapsed"
    )

    if user_query:
        st.divider()
        col_left, col_right = st.columns([1.15, 1.0])

        with col_left:
            st.markdown("#### Tactical Analysis & Citations")
            st.markdown(f"""
            <div class="engine-indicator">
                <span class="engine-dot"></span>
                <span>Active engine: <strong>{qa_chain.active_provider_name}</strong></span>
            </div>
            """, unsafe_allow_html=True)

            with st.spinner("Synthesizing answer with LangChain LCEL..."):
                docs = retriever.retrieve(user_query, ui_filters=ui_filters)
                formatted_context = qa_chain._format_docs(docs)
                try:
                    answer = qa_chain.chain.invoke({"context": formatted_context, "question": user_query})
                    retrieved_docs = docs
                except Exception as e:
                    err_msg = str(e)
                    qa_chain.active_provider_name = f"{qa_chain.provider.upper()} (Unavailable) ➔ Local Analyst"
                    fallback_llm = SoccerQAChain(retriever=retriever, provider="local").chain
                    answer = fallback_llm.invoke({"context": formatted_context, "question": user_query})
                    retrieved_docs = docs

            st.markdown(f"""
            <div class="analysis-card">
                <div style="font-size: 0.94rem; line-height: 1.65; color: #2D3436; white-space: pre-line;">
{answer}
                </div>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("#### Retrieved Event Details")
            for doc in retrieved_docs:
                meta = doc.metadata
                team_class = "pill-team-a" if meta.get('team_id') == "Team A" else "pill-team-b"
                
                expander_title = f"{meta.get('start_timestamp_str')} – {meta.get('end_timestamp_str')} · {meta.get('event_type').replace('_', ' ').title()}"
                with st.expander(expander_title):
                    st.markdown(f"""
                    <div style="margin-bottom: 8px;">
                        <span class="event-pill pill-time">{meta.get('start_timestamp_str')} – {meta.get('end_timestamp_str')}</span>
                        <span class="event-pill {team_class}">{meta.get('team_id')}</span>
                        <span class="event-pill pill-zone">{meta.get('pitch_zones')}</span>
                    </div>
                    <div style="font-size: 0.88rem; color: #4A5568; line-height: 1.5; margin-top: 6px;">
                        {meta.get('raw_description')}
                    </div>
                    """, unsafe_allow_html=True)

        with col_right:
            st.markdown("#### Video Clip & Minimap Playback")
            if retrieved_docs:
                # 1. Event Selector for multi-event queries
                event_options = []
                for i, doc in enumerate(retrieved_docs):
                    meta = doc.metadata
                    ev_type = meta.get("event_type", "Event").replace("_", " ").title()
                    t_str = f"[{meta.get('start_timestamp_str')} – {meta.get('end_timestamp_str')}]"
                    team = meta.get("team_id", "")
                    team_label = f"({team})" if team and team != "N/A" else ""
                    event_options.append(f"{t_str} {ev_type} {team_label}")

                selected_idx = 0
                if len(retrieved_docs) > 1:
                    selected_idx = st.selectbox(
                        "Select Event to Inspect in Video & Minimap:",
                        options=list(range(len(event_options))),
                        format_func=lambda i: event_options[i],
                        key="selected_event_playback_idx"
                    )

                active_meta = retrieved_docs[selected_idx].metadata
                start_sec = float(active_meta.get("start_timestamp", 0.0))
                end_sec = float(active_meta.get("end_timestamp", 10.0))
                ts_str = f"{active_meta.get('start_timestamp_str')} – {active_meta.get('end_timestamp_str')}"
                dur_sec = max(1.0, end_sec - start_sec)

                # 2. Playback Mode Selection (Focused Sub-Clip vs Full Match Context)
                mode_col1, mode_col2 = st.columns([1.2, 1.0])
                with mode_col1:
                    st.markdown(f"""
                    <div style="margin-top: 4px; margin-bottom: 8px;">
                        <span class="event-pill pill-time">{ts_str}</span>
                        <span class="event-pill pill-zone">{dur_sec:.1f}s window</span>
                    </div>
                    """, unsafe_allow_html=True)
                with mode_col2:
                    playback_mode = st.radio(
                        "Playback View",
                        ["Focused Sub-Clip", "Full Match Timeline"],
                        horizontal=True,
                        label_visibility="collapsed",
                        key=f"pb_mode_{selected_idx}"
                    )

                video_path = config["paths"]["video_file"]

                # 3. Video Rendering
                if playback_mode == "Focused Sub-Clip":
                    clip_path = clipper.extract_clip(video_path, start_sec, end_sec)
                    if clip_path and Path(clip_path).exists():
                        st.video(clip_path, format="video/mp4", autoplay=False)
                    else:
                        st.warning("Video clip extraction in progress or file not found.")
                else:
                    # Full Match Video scrubbed to start_sec
                    if Path(video_path).exists():
                        st.video(video_path, format="video/mp4", start_time=int(start_sec), autoplay=False)
                        st.caption(f"Scrubbed directly to match start time: **{active_meta.get('start_timestamp_str')}**")
                    else:
                        st.warning("Full match video file not found.")

                # 4. Synchronized 2D Tactical Minimap
                st.markdown("<div style='margin-top: 14px;'></div>", unsafe_allow_html=True)
                draw_tactical_minimap(config["paths"]["tracking_data"], start_sec, end_sec)
            else:
                st.info("No matching match events to display for video playback.")


if __name__ == "__main__":
    main()
