import streamlit as st

def render_scrollytelling(
    fig,
    story_blocks: list
):
    """
    Renders a scrollytelling section: chart on top, story cards below.
    
    Args:
        fig: A Plotly Figure object.
        story_blocks: List of dicts containing title, content, highlight.
    """
    
    # Section header
    st.markdown("### 📊 Flight Recorder Analysis")
    
    # Chart first — full width
    st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
    
    # Story cards in balanced columns below the chart
    cols = st.columns(len(story_blocks))
    for i, block in enumerate(story_blocks):
        with cols[i]:
            render_story_card(block['title'], block['content'], block.get('highlight', False))


def render_story_card(title: str, content: str, highlight: bool = False):
    """Renders a single story card with guaranteed text visibility."""
    border_style = "border: 1px solid #00D4FF; box-shadow: 0 0 12px rgba(0,212,255,0.15);" if highlight else "border: 1px solid rgba(255,255,255,0.08);"
    
    html = f"""
<div class="obsidian-card" style="{border_style}">
    <h4 style="margin-bottom: 0.75rem; color: #FFFFFF; font-family: 'Inter', sans-serif; font-size: 1.1rem;">{title}</h4>
    <div style="color: #D0D0D0; line-height: 1.65; font-size: 0.95rem;">{content}</div>
</div>
    """
    st.markdown(html, unsafe_allow_html=True)

