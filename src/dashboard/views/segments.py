import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from src.dashboard.metrics import get_segment_lift

def render_segments_view(df_trips: pd.DataFrame) -> None:
    
    # Control Panel in a Card
    st.markdown('<div class="obsidian-card" style="margin-bottom: 1rem;">', unsafe_allow_html=True)
    c1, c2 = st.columns([1, 3])
    with c1:
        st.markdown('<div class="metric-label" style="margin-top: 0.5rem;">GROUP BY</div>', unsafe_allow_html=True)
    with c2:
        segment_type = st.radio("Group By", ["Trade", "Line"], horizontal=True, label_visibility="collapsed")
    st.markdown('</div>', unsafe_allow_html=True)

    col_map = {"Trade": "trade", "Line": "line"}
    df_lift = get_segment_lift(df_trips, col_map[segment_type])
    
    if df_lift.empty:
        st.warning("No data for aggregation.")
        return

    # Sort and slice for visualization (Top/Bottom impacts)
    df_lift = df_lift.sort_values('Lift', ascending=True) # Ascending for BarH
    
    # Visualizing as a Horizontal Bar Chart is more "Premium" than a table
    # It allows immediate recognition of under/over performers
    
    # Logic for colors: Green for positive, Purple for negative
    colors = ['#00CC96' if x > 0 else '#7928CA' for x in df_lift['Lift']]
    
    fig = go.Figure(data=[
        go.Bar(
            y=df_lift.index,
            x=df_lift['Lift'],
            orientation='h',
            marker_color=colors,
            text=[f"{x:+.1f}" for x in df_lift['Lift']],
            textposition='auto',
            hovertemplate="<b>%{y}</b><br>Lift: %{x:.2f}<br>Trips: %{customdata}<extra></extra>",
            customdata=df_lift['Total_Trips']
        )
    ])
    
    fig.update_layout(
        template="plotly_dark",
        title=dict(text=f"LIFT BY {segment_type.upper()}", font=dict(family="Inter", color="#FFFFFF", size=14)),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        height=max(400, len(df_lift) * 30), # Dynamic height
        margin=dict(t=40, b=20, l=20, r=20),
        xaxis=dict(
            showgrid=True, 
            gridcolor='rgba(255,255,255,0.1)', 
            zeroline=True, 
            zerolinecolor='rgba(255,255,255,0.2)',
            tickfont=dict(color='#E0E0E0')
        ),
        yaxis=dict(
            tickfont=dict(color='#FFFFFF', family='JetBrains Mono'),
            dtick=1 # Ensure all labels show
        )
    )
    
    st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
    
    # Data Table Expander (for detailed view)
    with st.expander("View Raw Data Table"):
        st.dataframe(df_lift.style.format("{:.2f}"), use_container_width=True)
        

