import streamlit as st
import json
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer, util

# Page setup
st.set_page_config(page_title="LitReview AI+", page_icon="🧠", layout="wide")

@st.cache_resource
def load_model():
    # 'all-MiniLM-L6-v2' is fast, lightweight, and great for abstracts
    return SentenceTransformer('all-MiniLM-L6-v2')

@st.cache_data
def load_data():
    try:
        with open('semantic_scholar.json', 'r') as f:
            raw_data = json.load(f)
            data = raw_data.get("references", [])
        return pd.DataFrame(data)
    except Exception as e:
        st.error(f"Error loading JSON: {e}")
        return pd.DataFrame()

# Initialize data and AI model
df = load_data()
model = load_model()

# --- Sidebar Configuration ---
st.sidebar.title("🔍 Search Controls")

if not df.empty:
    # 1. Toggle between modes
    search_mode = st.sidebar.radio("Search Mode", ["Standard (Regex)", "Semantic (AI)"])
    
    # 2. Input
    query = st.sidebar.text_input("Enter search terms:", "")
    
    # Filters control
    filters_enabled = st.sidebar.checkbox("Enable filters (blurred when disabled)", value=False)
    
    # 3. Filters (Years & Keywords) — shown but disabled when not enabled
    min_yr, max_yr = int(df['year'].min()), int(df['year'].max())
    year_range = st.sidebar.slider("Publication Year", min_yr, max_yr, (min_yr, max_yr), disabled=not filters_enabled)
    
    all_keywords = sorted(list(set([kw for sublist in df['keywords'] for kw in sublist])))
    selected_keywords = st.sidebar.multiselect("Filter by Keywords", all_keywords, disabled=not filters_enabled)

    # Search Complete button
    search_complete = st.sidebar.button("Search Complete")

    # CSS blur overlay for disabled filters
    if not filters_enabled:
        st.sidebar.markdown(
            """
            <style>
            .filters-overlay {
                position: relative;
            }
            .filters-overlay::after {
                content: 'Filters disabled — click "Enable filters" to unlock';
                position: absolute;
                left: 0;
                top: 0;
                right: 0;
                bottom: 0;
                display: flex;
                align-items: center;
                justify-content: center;
                background: rgba(255,255,255,0.6);
                backdrop-filter: blur(4px);
                border-radius: 8px;
                z-index: 9;
                font-weight:600;
                color:#333;
            }
            </style>
            <div class="filters-overlay"></div>
            """, unsafe_allow_html=True
        )

    # --- Filtering Logic ---
    if filters_enabled:
        # Apply Year/Keyword filters
        mask = (df['year'] >= year_range[0]) & (df['year'] <= year_range[1])
        if selected_keywords:
            mask &= df['keywords'].apply(lambda x: any(kw in x for kw in selected_keywords))
        results_df = df[mask].copy()
    else:
        # Filters are disabled — do not filter by year/keywords
        results_df = df.copy()

    # Apply Text Search based on Mode
    if query and not results_df.empty:
        if search_mode == "Standard (Regex)":
            results_df = results_df[
                results_df['title'].str.contains(query, case=False) | 
                results_df['abstract'].str.contains(query, case=False)
            ]
        else:
            # AI Semantic Search logic
            with st.spinner("AI is analyzing semantics..."):
                # Encode the abstracts and the query
                corpus_embeddings = model.encode(results_df['abstract'].tolist(), convert_to_tensor=True)
                query_embedding = model.encode(query, convert_to_tensor=True)
                
                # Compute cosine similarity
                cosine_scores = util.cos_sim(query_embedding, corpus_embeddings)[0]
                
                # Add scores to dataframe and sort
                results_df['score'] = cosine_scores.tolist()
                results_df = results_df[results_df['score'] > 0.35] # Threshold for relevance
                results_df = results_df.sort_values(by='score', ascending=False)

    # --- Main UI ---
    st.title("Literature Review Interface")
    st.info(f"Currently in **{search_mode}** mode.")

    if results_df.empty:
        st.warning("No articles match your filters.")
    else:
        for _, row in results_df.iterrows():
            with st.container():
                h_col1, h_col2 = st.columns([0.8, 0.2])
                with h_col1:
                    score_tag = f" (Match: {row['score']:.2f})" if 'score' in row else ""
                    st.subheader(f"{row['title']}{score_tag}")
                with h_col2:
                    st.checkbox("Select", key=f"sel_{row['id']}")

                st.markdown(f"**{', '.join(row['authors'])}** | *{row['journal']}* ({row['year']})")
                
                with st.expander("Show Abstract & Metadata"):
                    st.write(row['abstract'])
                    st.markdown(f"**Keywords:** " + ", ".join([f"`{k}`" for k in row['keywords']]))
                st.divider()

    # Handle Search Complete Button
    if search_complete:
        selected_articles = [row for _, row in results_df.iterrows() if st.session_state.get(f"sel_{row['id']}", False)]
        if selected_articles:
            st.success("Search completed! Here are the selected articles:")
            for art in selected_articles:
                st.write(f"- **{art['title']}** by {', '.join(art['authors'])} ({art['year']})")
        else:
            st.info("Search completed, but no articles were selected.")