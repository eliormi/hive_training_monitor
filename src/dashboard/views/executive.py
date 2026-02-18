import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from src.dashboard.metrics import calculate_global_metrics, calculate_oll_success_rate

def render_executive_view(df_trips: pd.DataFrame, df_oll: pd.DataFrame) -> None:
    
    # 1. Calculate Metrics
    # Handle empty frames gracefully
    if df_trips.empty:
        st.error("No trip data available for executive view.")
        return

    agent_mean, model_mean, lift = calculate_global_metrics(df_trips)
    
    # OLL might be empty for the filtered "Valid Trips" view
    succ_ratio, succ_count, total_count = 0, 0, 0
    if not df_oll.empty:
        succ_ratio, succ_count, total_count = calculate_oll_success_rate(df_oll)
    
    # 2. Layout: 3 Columns using Custom HTML Cards
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown(f"""
        <div class="obsidian-card">
            <div class="metric-label">Test Set Lift</div>
            <div class="metric-value">+{lift:.1f}</div>
            <div class="metric-delta-pos">
                Model: {model_mean:.1f} <span style="color:var(--text-gray)">vs</span> Agent: {agent_mean:.1f}
            </div>
        </div>
        """, unsafe_allow_html=True)
        
    with col2:
        if not df_oll.empty:
            # Label above the chart
            st.markdown(f"""
<div class="obsidian-card">
    <div class="metric-label">Origin-Leg Success</div>
    <div class="metric-delta-pos" style="margin-bottom: 0.5rem;">
        {succ_count} of {total_count} O-L-L combos outperform agent
    </div>
</div>
            """, unsafe_allow_html=True)

            # Donut Chart with Neon Style
            fig = go.Figure(data=[go.Pie(
                labels=['Success', 'Lower'], 
                values=[succ_count, total_count - succ_count], 
                hole=.75,
                marker=dict(colors=['#00CC96', '#1F242D'], line=dict(color='#0E1117', width=2)),
                textinfo='none',
                hoverinfo='label+percent'
            )])
            
            fig.update_layout(
                template="plotly_dark",
                showlegend=False,
                margin=dict(t=0, b=0, l=0, r=0),
                height=120,
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                annotations=[
                    dict(
                        text=f"{succ_ratio:.0f}%", 
                        x=0.5, y=0.5, 
                        font=dict(size=28, family="JetBrains Mono", color="#00CC96"),
                        showarrow=False
                    )
                ]
            )
            
            st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
            
        else:
             st.markdown("""
<div class="obsidian-card">
    <div class="metric-label">Origin-Leg Success</div>
    <div class="metric-value" style="color:var(--text-gray); font-size:1.5rem">N/A</div>
    <div class="metric-delta-pos" style="color:var(--text-gray)">No Leg Data</div>
</div>
            """, unsafe_allow_html=True)

    with col3:
        # Label above the chart
        st.markdown(f"""
<div class="obsidian-card">
    <div class="metric-label">Agent vs Model</div>
    <div class="metric-delta-pos" style="margin-bottom: 0.5rem;">
        Model scores <span style="color:#00D4FF">{model_mean:.1f}</span> vs Agent <span style="color:#7928CA">{agent_mean:.1f}</span>
    </div>
</div>
        """, unsafe_allow_html=True)

        # Comparative Bar Chart
        fig_bar = go.Figure(data=[
            go.Bar(
                name='Agent', x=['Agent'], y=[agent_mean], 
                marker_color='#7928CA',
                marker_line_width=0,
                opacity=0.8
            ),
            go.Bar(
                name='Model', x=['Model'], y=[model_mean], 
                marker_color='#00D4FF',
                marker_line_width=0
            )
        ])
        
        fig_bar.update_layout(
            template="plotly_dark",
            margin=dict(t=10, b=20, l=20, r=20), 
            height=120, 
            showlegend=False,
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            yaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.1)', zeroline=False, visible=True, tickfont=dict(color='#E0E0E0')),
            xaxis=dict(showgrid=False, tickfont=dict(color='#FFFFFF', family='Inter')),
            bargap=0.4
        )
        
        st.plotly_chart(fig_bar, use_container_width=True, config={'displayModeBar': False})
