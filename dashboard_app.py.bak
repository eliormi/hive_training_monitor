import streamlit as st
import pandas as pd
from src.dashboard.data_loader import load_trip_data, load_oll_data, get_filtered_data, get_valid_trips
from src.dashboard.views.executive import render_executive_view
from src.dashboard.views.overfitting import render_overfitting_view
from src.dashboard.views.segments import render_segments_view

# Page Config
st.set_page_config(
    page_title="DSG Model Success | Obsidian Flow",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Load Premium CSS
with open('src/dashboard/styles_premium.css') as f:
    st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

def main():
    # --- Data Loading ---
    with st.spinner("Initializing Obsidian Flow..."):
        df_trips = load_trip_data()
        df_oll = load_oll_data()

    if df_trips.empty or df_oll.empty:
        st.error("System Error: Critical data files missing.")
        return

    # Preliminary Processing
    df_test = get_filtered_data(df_trips, 'test')
    df_valid = get_valid_trips(df_test)

    # --- ACT 1: THE HEADLINE (Hero) ---
    st.markdown('<div class="hero-subtitle">SYSTEM STATUS: ONLINE</div>', unsafe_allow_html=True)
    st.markdown('<h1 class="hero-title">Model Performance <span class="text-gradient">Registry</span></h1>', unsafe_allow_html=True)
    st.markdown("---")

    # Render Executive View (Refactored to match new style in next steps, currently using existing logic but will profit from CSS)
    render_executive_view(df_valid, df_oll)

    st.markdown("<br><br>", unsafe_allow_html=True)

    # --- ACT 2: THE NARRATIVE (Storytelling) ---
    from src.dashboard.components.story_module import render_scrollytelling
    import plotly.express as px
    
    # 1. Create the Sticky Graphic (Interactive Histogram of Confirmations)
    # This visualizes "Why we filter"
    fig_valid = px.histogram(
        df_trips[df_trips['confirmed_at_end'] > 0], 
        x="confirmed_at_end",
        nbins=50,
        color_discrete_sequence=['#00D4FF'],
        opacity=0.8
    )
    fig_valid.update_layout(
        template="plotly_dark",
        title="<b>CONFIRMED BOOKINGS DISTRIBUTION</b><br><span style='font-size:12px;color:#A0A5B0'>Only trips with >0 bookings shown</span>",
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(family="Inter", color="#FFFFFF"),
        xaxis=dict(showgrid=False, title="Confirmed Bookings"),
        yaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.1)'),
        margin=dict(l=20, r=20, t=60, b=20),
        height=400,
        showlegend=False
    )
    
    # 2. Define Story Beats
    story_blocks = [
        {
            "title": "1. The Noise Problem",
            "content": "Raw trip data contains thousands of low-quality records. Empty flights, cancelled routes, and placeholders create statistical noise that masks true model performance."
        },
        {
            "title": "2. The Filter Protocol",
            "content": """To pierce the noise, we apply a strict validity filter. <br><br>
            We keep only trips where:
            <ul>
                <li><strong style='color:#00CC96'>Actual Bookings > 0</strong> (Real demand exists)</li>
                <li><strong style='color:#00CC96'>Allocation > 50</strong> (Significant capacity managed)</li>
            </ul>
            This reduces the dataset from <strong style='color:#FFFFFF'>{:,}</strong> to <strong style='color:#00D4FF'>{:,} high-signal trips</strong>.""".format(len(df_trips), len(df_valid)),
            "highlight": True
        },
        {
            "title": "3. The Result",
            "content": "The chart on the left shows the distribution of the <b>filtered validation set</b>. This contains the flights that actually matter for revenue. <br><br><i>These are the trips driving the metrics shown above.</i>"
        }
    ]
    
    # 3. Render the Module (Passing the Figure Object directly)
    render_scrollytelling(fig_valid, story_blocks)
    
    st.markdown("---")
    
    # High Value Performance (Removed as redundant - shown in Executive View)

    st.markdown("---")

    # --- ACT 3: THE DEEP DIVE (Explorer) ---
    st.markdown("### 🔬 Sector Diagnostics")
    
    tab1, tab2 = st.tabs(["Overfitting Analysis", "Segment Breakdown"])
    
    with tab1:
        render_overfitting_view(df_trips)
        
    with tab2:
        render_segments_view(df_test)

    st.markdown("---")

    # --- ACT 4: THE BONEYARD (Graveyard Analysis) ---
    from src.dashboard.views.graveyard import render_graveyard_view
    from src.dashboard.data_loader import get_graveyard_trips

    # Calculate Graveyard Data
    df_grave = get_graveyard_trips(df_test)
    
    # We use an expander to keep the "Happy Path" focused, 
    # but allow digging into the "sad path".
    with st.expander("💀 View Graveyard Analysis (Invalid Trips)"):
        render_graveyard_view(df_grave, total_trips_count=len(df_test))

if __name__ == "__main__":
    main()
