import pandas as pd


def calculate_global_metrics(
    df: pd.DataFrame, target: str = 'trip_score_6_weeks', model_col: str = 'model_type'
) -> tuple[float, float, float]:
    """
    Calculates Mean Score for Agent and Model, and the Net Lift.
    Returns: (agent_mean, model_mean, lift)
    """
    if df.empty:
        return 0, 0, 0

    stats = df.groupby(model_col)[target].mean()
    agent_mean = stats.get('Agent', 0)
    model_mean = stats.get('DSG_model', 0)
    lift = model_mean - agent_mean

    return agent_mean, model_mean, lift

def calculate_oll_success_rate(df_oll: pd.DataFrame) -> tuple[float, int, int]:
    """
    Calculates the percentage of Origin-Line-Legs where Model > Agent.
    Assumes df_oll has 'diff_trip_score_6_weeks' or pre-calculated means.
    """
    if df_oll.empty:
        return 0, 0, 0

    # Valid filter if applicable (optional based on user request, but good practice)
    # df_valid = df_oll[df_oll['trip_count_gt_min'] == True] if 'trip_count_gt_min' in df_oll.columns else df_oll
    # Using all for now to match EDA script

    # Success defined as diff > 0
    success = df_oll[df_oll['diff_trip_score_6_weeks'] > 0]
    total = len(df_oll)
    success_count = len(success)
    ratio = (success_count / total) * 100 if total > 0 else 0

    return ratio, success_count, total

def get_segment_lift(
    df: pd.DataFrame, segment_col: str, target: str = 'trip_score_6_weeks', model_col: str = 'model_type'
) -> pd.DataFrame:
    """
    Calculates Lift by Segment (e.g., Trade, Line).
    Returns a DataFrame sorted by Lift descending.
    """
    if df.empty:
        return pd.DataFrame()

    # Pivot to get columns [Agent, DSG_model]
    pivot = df.pivot_table(index=segment_col, columns=model_col, values=target, aggfunc='mean')

    if 'Agent' not in pivot or 'DSG_model' not in pivot:
        return pd.DataFrame()

    pivot['Lift'] = pivot['DSG_model'] - pivot['Agent']

    # Add counts
    counts = df.groupby(segment_col)[target].count()
    pivot['Total_Trips'] = counts

    return pivot.sort_values('Lift', ascending=False)
