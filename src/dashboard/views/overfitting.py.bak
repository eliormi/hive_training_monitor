import streamlit as st
import plotly.graph_objects as go
from src.dashboard.metrics import calculate_global_metrics
from src.dashboard.data_loader import get_filtered_data
import pandas as pd

def render_overfitting_view(df_raw):
    
    # Calculate Data
    datasets = ['train', 'val', 'test']
    results = []
    
    for ds in datasets:
        subset = get_filtered_data(df_raw, ds)
        _, _, lift = calculate_global_metrics(subset)
        results.append({'Dataset': ds.capitalize(), 'Lift': lift})
        
    df_res = pd.DataFrame(results)
    
    # Custom Colors per dataset
    colors = {
        'Train': '#2E3440', # Muted for training
        'Val': '#7928CA',   # Purple for validation
        'Test': '#00D4FF'   # Neon Cyan for Test (Hero)
    }
    
    fig = go.Figure(data=[
        go.Bar(
            x=df_res['Dataset'], 
            y=df_res['Lift'],
            text=df_res['Lift'].apply(lambda x: f"+{x:.1f}" if x>0 else f"{x:.1f}"),
            textposition='auto',
            marker_color=[colors.get(d, '#A0A5B0') for d in df_res['Dataset']],
            marker_line_width=0
        )
    ])
    
    fig.update_layout(
        template="plotly_dark",
        title=dict(
            text="GENERALIZATION CHECK",
            font=dict(family="Inter", size=14, color="#FFFFFF"),
            x=0
        ),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        height=300,
        margin=dict(l=20, r=20, t=40, b=20),
        yaxis=dict(
            showgrid=True, 
            gridcolor='rgba(255,255,255,0.1)', 
            zeroline=True, 
            zerolinecolor='rgba(255,255,255,0.2)',
            tickfont=dict(color='#E0E0E0')
        ),
        xaxis=dict(
            showgrid=False,
            tickfont=dict(color='#FFFFFF', family='Inter', size=12)
        )
    )
    
    st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
    
    # Narrative note check
    test_lift = df_res[df_res['Dataset']=='Test']['Lift'].values[0]
    train_lift = df_res[df_res['Dataset']=='Train']['Lift'].values[0]
    
    if test_lift < train_lift * 0.8:
         st.markdown(f"""
        <div style="margin-top: 1rem; padding: 0.5rem; border-left: 2px solid #F5A623; color: #F5A623; font-size: 0.85rem;">
            ⚠️ <strong>Generalization Gap</strong><br>
            Test performance is significantly lower than Training. Investigate overfitting.
        </div>
        """, unsafe_allow_html=True)
