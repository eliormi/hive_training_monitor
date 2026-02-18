import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd

def render_graveyard_view(df_grave, total_trips_count):
    """
    Renders the 'Graveyard' analysis for invalid trips.
    Theme: Ghost Protocol (Monochrome/Red).
    """
    
    st.markdown("### 💀 The Boneyard (Invalid Trips)")
    st.markdown("""
    <div style="background: rgba(255, 59, 48, 0.1); border: 1px solid rgba(255, 59, 48, 0.3); border-radius: 8px; padding: 1rem; margin-bottom: 2rem;">
        <p style="color: #FF453A; margin: 0; font-size: 0.9rem;">
            <strong>LOST SIGNAL DETECTED:</strong> This section analyzes trips excluded from the main model evaluation. 
            Understanding <em>why</em> these trips are invalid reveals potential data pipeline issues or lost commercial opportunities.
        </p>
    </div>
    """, unsafe_allow_html=True)

    if df_grave.empty:
        st.info("No graveyard trips found. Data is 100% clean!")
        return

    # 1. Metrics
    grave_count = len(df_grave)
    waste_ratio = (grave_count / total_trips_count) * 100
    
    # Approximate "Wasted Capacity"
    # We use the mid-point of allocation bins or raw if available. 
    # Since we only have 'allocation_at_end_grouped' which is categorical/interval, 
    # we can't sum it easily without more parsing. 
    # Let's stick to Volume for now.
    
    c1, c2, c3 = st.columns(3)
    
    with c1:
        st.markdown(f"""
        <div class="obsidian-card" style="border-color: rgba(255, 69, 58, 0.2);">
            <div class="metric-label" style="color: #FF453A;">Discarded Trips</div>
            <div class="metric-value" style="color: #FF453A;">{grave_count:,}</div>
            <div class="metric-delta-pos" style="color: #636366;">
                {waste_ratio:.1f}% of Total Volume
            </div>
        </div>
        """, unsafe_allow_html=True)
        
    with c2:
        # Reason Breakdown
        reason_counts = df_grave['death_reason'].value_counts().reset_index()
        reason_counts.columns = ['Reason', 'Count']
        
        fig_pie = px.pie(
            reason_counts, 
            values='Count', 
            names='Reason',
            color='Reason',
            color_discrete_map={'Zero Bookings': '#1C1C1E', 'Low Allocation': '#3A3A3C'},
            hole=0.6
        )
        fig_pie.update_traces(textinfo='percent+label', textfont_color='white', marker=dict(line=dict(color='#FF453A', width=1)))
        fig_pie.update_layout(
            template="plotly_dark",
            showlegend=True, # Show legend now that we have contrast
            legend=dict(font=dict(color="#FFFFFF")),
            margin=dict(t=0, b=0, l=0, r=0),
            height=120,
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)'
        )
        
        st.markdown('<div class="obsidian-card" style="border-color: rgba(255, 69, 58, 0.2); display: flex; justify-content: center;">', unsafe_allow_html=True)
        st.plotly_chart(fig_pie, use_container_width=True, config={'displayModeBar': False})
        st.markdown('</div>', unsafe_allow_html=True)

    with c3:
       # Contextual Insight
       dominant_reason = reason_counts.iloc[0]['Reason']
       dominant_pct = (reason_counts.iloc[0]['Count'] / grave_count) * 100
       
       insight = "Most trips are empty."
       if dominant_reason == 'Low Allocation':
           insight = "Most trips satisfy bookings but are too small to matter."
           
       st.markdown(f"""
        <div class="obsidian-card" style="border-color: rgba(255, 69, 58, 0.2);">
            <div class="metric-label" style="color: #FF453A;">Primary Cause</div>
            <div class="metric-value" style="font-size: 1.8rem; color: #FFFFFF;">{dominant_reason}</div>
            <div class="metric-delta-pos" style="color: #636366;">
                {dominant_pct:.0f}% of invalid data. {insight}
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # 2. "Zombie" Segments (Who is dying the most?)
    st.markdown("#### 🧟 Zombie Segments (Highest Rejection Rates)")
    
    # We need to calculate rejection rate per Trade
    # This requires the FULL dataset (which we have implicitly if we pass it or calculate it)
    # For now, let's just show raw counts of Graveyard trips by Trade
    
    if 'trade' in df_grave.columns:
        grave_by_trade = df_grave['trade'].value_counts().head(5).reset_index()
        grave_by_trade.columns = ['Trade', 'Dead_Trips']
        
        fig_bar = px.bar(
            grave_by_trade, 
            x='Dead_Trips', 
            y='Trade', 
            orientation='h',
            color_discrete_sequence=['#FF453A']
        )
        
        fig_bar.update_layout(
            template="plotly_dark",
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(family="Inter", color="#E0E0E0"),
            height=200,
            margin=dict(l=0, r=0, t=0, b=0),
            yaxis=dict(autorange="reversed", tickfont=dict(color="#FFFFFF")),
            xaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.1)', title=dict(text="Lost Trips", font=dict(color="#A0A5B0")))
        )
        
        st.markdown('<div class="obsidian-card" style="border-color: rgba(255, 69, 58, 0.2);">', unsafe_allow_html=True)
        st.plotly_chart(fig_bar, use_container_width=True, config={'displayModeBar': False})
        st.markdown('</div>', unsafe_allow_html=True)
