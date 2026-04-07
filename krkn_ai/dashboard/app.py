import os
import sys
import argparse
import time
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from krkn_ai.dashboard.data_loader import load_results_csv, load_config_yaml, load_health_check_csv, load_detailed_scenarios_data, load_logs

def get_monitor_config():
    """Retrieve monitor config from command line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=str, default="./")
    try:
        args, _ = parser.parse_known_args()
        return {"output_dir": args.output_dir}
    except SystemExit:
        return {"output_dir": "./"}

def is_execution_running(output_dir: str) -> bool:
    """Detect if krkn-ai is currently running by checking results.json."""
    results_file = os.path.join(output_dir, "results.json")
    if not os.path.exists(results_file):
        return False
    try:
        import json
        with open(results_file, 'r') as f:
            data = json.load(f)
            status = data.get("status")
            if status in ["created", "started", "in progress"]:
                return True
    except Exception:
        pass
    return False

def render_summary(df):
    st.header("Experiment Summary")
    if df is None or df.empty:
        st.warning("Results data not yet available. Waiting for Krkn-AI engine...")
        return

    # stats directly from CSV data
    generations_completed = int(df["generation_id"].max() + 1) if "generation_id" in df.columns else 0
    scenarios_executed = len(df)
    best_fitness = df["fitness_score"].max() if "fitness_score" in df.columns else 0.0
    avg_fitness = df["fitness_score"].mean() if "fitness_score" in df.columns else 0.0

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Generations Completed", generations_completed)
    col2.metric("Scenarios Executed", scenarios_executed)
    col3.metric("Best Fitness Score", f"{best_fitness:.4f}")
    col4.metric("Avg Fitness Score", f"{avg_fitness:.4f}")

def render_fitness_evolution(df):
    st.header("Fitness Score Evolution")
    if df is None or df.empty or "generation_id" not in df.columns:
        st.write("Not enough data to plot fitness evolution.")
        return

    # Grouping CSV by generation to plot Best vs Average
    grouped = df.groupby("generation_id")["fitness_score"].agg(['mean', 'max']).reset_index()
    grouped.rename(columns={"mean": "Average Fitness", "max": "Best Fitness"}, inplace=True)
    grouped["generation_id"] = grouped["generation_id"] + 1  
    
    if not grouped.empty:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=grouped["generation_id"], y=grouped["Average Fitness"], mode='lines+markers', name='Average Fitness'))
        fig.add_trace(go.Scatter(x=grouped["generation_id"], y=grouped["Best Fitness"], mode='lines+markers', name='Best Fitness'))
        
        fig.update_layout(
            title="Fitness Performance Over Generations",
            xaxis_title="Generation",
            yaxis_title="Fitness Score",
            hovermode="x unified",
            xaxis={"tickmode": 'linear', "tick0": 1, "dtick": 1}
        )
        st.plotly_chart(fig, width='stretch')
    else:
        st.info("No fitness scores recorded yet.")

def render_scenario_distribution(df):
    st.header("Scenario Distribution")
    if df is None or df.empty or "scenario" not in df.columns:
        st.write("Not enough data to plot distribution.")
        return

    fig = px.histogram(df, x="scenario", title="Executed Scenarios Frequency", color="scenario")
    fig.update_layout(xaxis_title="Scenario Name", yaxis_title="Execution Count")
    st.plotly_chart(fig, width='stretch')

def render_scenario_fitness_variation(df):
    st.header("Scenario-wise Fitness Variation")
    if df is None or df.empty or "generation_id" not in df.columns or "scenario" not in df.columns:
        st.write("Not enough data to plot scenario fitness variation.")
        return

    # Group by scenario and generation
    grouped = df.groupby(["generation_id", "scenario"])["fitness_score"].max().reset_index()
    grouped["generation_id"] = grouped["generation_id"] + 1

    if not grouped.empty:
        fig = px.line(grouped, x="generation_id", y="fitness_score", color="scenario", markers=True, 
                      title="Best Fitness Variation by Scenario")
        fig.update_layout(
            xaxis_title="Generation",
            yaxis_title="Best Fitness Score",
            hovermode="x unified",
            xaxis={"tickmode": 'linear', "tick0": 1, "dtick": 1}
        )
        st.plotly_chart(fig, width='stretch')
    else:
        st.info("Not enough data points yet.")

def render_generation_details(df, title="Generation & Scenario Details"):
    st.header(title)
    if df is None or df.empty or "generation_id" not in df.columns:
        st.write("No failed scenario details available yet!!")
        return

    # Extract all unique generation numbers for the dropdown
    gen_nums = sorted(df["generation_id"].unique().tolist())
    display_gens = ["All"] + [g + 1 for g in gen_nums]
    selected_gen_disp = st.selectbox("Select Generation to view executed scenarios:", options=display_gens)
    
    if selected_gen_disp == "All":
        st.subheader("Results for All Generations")
        gen_scenarios = df.copy()
    else:
        st.subheader(f"Results for Generation {selected_gen_disp}")
        selected_gen_raw = selected_gen_disp - 1
        gen_scenarios = df[df["generation_id"] == selected_gen_raw].copy()
    
    if not gen_scenarios.empty:
        # Default sort: best fitness first (user can click column headers to re-sort)
        gen_scenarios = gen_scenarios.sort_values(by="fitness_score", ascending=False)

        display_cols = [
            'generation_id', 'scenario_id', 'scenario',
            'duration_seconds',
            'health_check_failure_score', 'health_check_response_time_score', 'krkn_failure_score',
            'fitness_score', 'parameters',
        ]
        available_cols = [c for c in display_cols if c in gen_scenarios.columns]
        view = gen_scenarios[available_cols].copy()
        if "generation_id" in view.columns:
            view["generation_id"] = view["generation_id"] + 1

        column_cfg = {
            "generation_id":                    st.column_config.NumberColumn("Generation", format="%d"),
            "scenario_id":                      st.column_config.NumberColumn("Scenario ID", format="%d"),
            "scenario":                         st.column_config.TextColumn("Scenario Name", width="medium"),
            "duration_seconds":                 st.column_config.NumberColumn("Duration (s)", format="%.1f"),
            "health_check_failure_score":        st.column_config.NumberColumn("HC Failure Score", format="%.4f"),
            "health_check_response_time_score":  st.column_config.NumberColumn("HC Response Score", format="%.4f"),
            "krkn_failure_score":               st.column_config.NumberColumn("Krkn Failure Score", format="%.4f"),
            "fitness_score":                    st.column_config.NumberColumn("Fitness Score", format="%.4f"),
            "parameters":                       st.column_config.TextColumn("Parameters"),
        }
        st.dataframe(
            view,
            column_config=column_cfg,
            width='stretch',
            hide_index=True,
        )
    else:
        st.write("No testing details available for this specific generation.")

def render_health_checks(df, global_services=None):
    st.header("Service Health Checks")
    if df is None or df.empty:
        st.warning("Health check data not yet available.")
        return

    if "failure_rate" not in df.columns:
        df["failure_rate"] = df["failure_count"] / (df["success_count"] + df["failure_count"]).clip(lower=1)
    if "variance" not in df.columns:
        df["variance"] = (df["max_response_time"] - df["min_response_time"]) / df["average_response_time"].clip(lower=0.0001)

    # Apply global service filter
    all_comps = sorted(df["component_name"].unique().tolist())
    if global_services:
        df = df[df["component_name"].isin(global_services)]

    scenarios = ["All"] + sorted(df["scenario_id"].unique().tolist())

    st.subheader("Interactive Heatmap")
    metric_col = st.selectbox("Select Metric:", ["average_response_time", "max_response_time", "min_response_time"])

    heat_df = df.groupby(["component_name", "scenario_id"])[metric_col].mean().reset_index()
    heat_df["scenario_id"] = heat_df["scenario_id"].astype(str)

    fig = px.density_heatmap(heat_df, x="component_name", y="scenario_id", z=metric_col,
                             histfunc="avg", title=f"{metric_col} Heatmap",
                             color_continuous_scale="RdYlGn_r")
    fig.update_layout(
        xaxis_title="Component",
        yaxis_title="Scenario ID",
        yaxis={"type": "category"}
    )
    fig.update_traces(xgap=3, ygap=3)
    st.plotly_chart(fig, width='stretch')

    st.divider()

    # Scenario trends line chart
    st.subheader("Scenario Trends")
    line_metric = st.selectbox("Trend Metric:", ["average_response_time", "max_response_time", "min_response_time"], key="line_metric")

    line_df = df.sort_values("scenario_id")
    line_df["scenario_id"] = line_df["scenario_id"].astype(str)
    fig2 = px.line(line_df, x="scenario_id", y=line_metric, color="component_name", markers=True,
                   title=f"{line_metric} Trends")
    fig2.update_layout(xaxis={"type": "category"})
    st.plotly_chart(fig2, width='stretch')

    st.divider()

    st.subheader("Success vs Failure")
    bar_base_df = df.copy()
    bar_df = bar_base_df.groupby("component_name")[["success_count", "failure_count"]].sum().reset_index()
    melt_bar = bar_df.melt(id_vars=["component_name"], value_vars=["success_count", "failure_count"],
                           var_name="Status", value_name="Count")
    fig3 = px.bar(melt_bar, x="component_name", y="Count", color="Status", title="Success vs Failure Counts",
                  barmode="stack", color_discrete_map={"success_count": "#28a745", "failure_count": "#dc3545"})
    st.plotly_chart(fig3, width='stretch')

    st.divider()

    st.subheader("Resilience Radar Chart")
    radar_df = df.copy()
    radar_df["scenario_id"] = radar_df["scenario_id"].astype(str)
    if not radar_df.empty:
        radar_df["score"] = 1 / radar_df["average_response_time"].clip(lower=0.0001)
        fig4 = px.line_polar(radar_df, r='score', theta='component_name', line_close=True,
                             color="scenario_id", title="Resilience Profile")
        fig4.update_traces(fill='toself', opacity=0.5)
        st.plotly_chart(fig4, width='stretch')
    else:
        st.info("No data for radar chart.")

    st.divider()

    st.subheader("Response Range Plot (Min-Max)")
    range_df = df.groupby("component_name").agg({"min_response_time": "min", "max_response_time": "max"}).reset_index()
    fig5 = go.Figure()
    for _, row in range_df.iterrows():
        fig5.add_trace(go.Scatter(
            x=[row["component_name"], row["component_name"]],
            y=[row["min_response_time"], row["max_response_time"]],
            mode='lines+markers',
            name=row["component_name"],
            showlegend=False,
            marker={"symbol": "line-ew", "size": 15}
        ))
    fig5.update_layout(title="Min/Max Range per Component", xaxis_title="Component", yaxis_title="Response Time Range")
    st.plotly_chart(fig5, width='stretch')

    st.divider()

    st.subheader("Top-K Worst Components Table")
    sort_by = st.selectbox("Sort Table By (Descending):", ["average_response_time", "failure_count", "failure_rate", "variance"])
    worst_k = st.number_input("Top K Worst Components:", min_value=1, value=10, max_value=50, key="worst_k")
    worst_table = df.sort_values(by=sort_by, ascending=False).head(worst_k)
    st.dataframe(worst_table)

# def render_best_scenarios_summary(df_best):
#     if not df_best:
#         return
        
#     st.subheader("Best Scenarios Overview (Per Generation)")
    
#     best_rows = []
#     for item in df_best:
#         best_rows.append({
#             "Generation": item.get("generation_id", "N/A"),
#             "Scenario ID": item.get("scenario_id", "N/A"),
#             "Scenario Name": item.get("scenario", {}).get("name", "N/A"),
#             "Fitness Score": item.get("fitness_result", {}).get("fitness_score", 0.0),
#             "Duration (s)": round(item.get("duration_seconds", 0.0), 2)
#         })
        
#     if best_rows:
#         best_df = pd.DataFrame(best_rows)
#         best_df = best_df.sort_values(by="Generation")
#         st.dataframe(best_df, width='stretch')
        
#     st.divider()

def render_detailed_scenarios(df_details, global_scenarios=None, global_services=None, scen_id_to_name=None):
    st.header("Detailed Scenarios Runtime Tracking")
    if df_details is None or df_details.empty:
        st.warning("No detailed scenario YAML telemetry available.")
        return

    def label(scen_id):
        if scen_id_to_name:
            name = scen_id_to_name.get(str(scen_id)) or scen_id_to_name.get(int(scen_id) if str(scen_id).isdigit() else scen_id)
            if name:
                return f"{scen_id} – {name}"
        return str(scen_id)

    # Apply global filters
    target_df = df_details.copy()
    if global_scenarios:
        target_df = target_df[target_df["scenario_id"].isin([str(s) for s in global_scenarios])]
    if global_services:
        target_df = target_df[target_df["service"].isin(global_services)]

    if target_df.empty:
        st.info("No data available for the selected filters.")
        return

    fig = go.Figure()

    for scen in target_df["scenario_id"].unique():
        for srv in target_df[target_df["scenario_id"] == scen]["service"].unique():
            srv_df = target_df[(target_df["scenario_id"] == scen) & (target_df["service"] == srv)]

            fig.add_trace(go.Scatter(
                x=srv_df["seconds_into_scenario"],
                y=srv_df["response_time"],
                mode="lines+markers",
                name=f"{srv} ({label(scen)})",
                customdata=srv_df[["timestamp", "status_code", "error"]],
                hovertemplate="Service: " + srv + "<br>Scenario: " + label(scen) + "<br>Time: %{customdata[0]}<br>Seconds: %{x:.2f}s<br>Response Time: %{y:.4f}s<br>Status: %{customdata[1]}<br>Error: %{customdata[2]}<extra></extra>",
                marker=dict(size=6)
            ))

    fig.update_layout(
        title="Runtime Telemetry: Response Time vs Scenario Execution Time",
        xaxis_title="Seconds into Scenario (s)",
        yaxis_title="Response Time (s)",
        hovermode="closest"
    )

    st.plotly_chart(fig, width='stretch')

    st.divider()

    st.subheader("Success per Service Over Time")
    succ_df = target_df.copy()

    if not succ_df.empty:
        succ_df["time_sec"] = succ_df["seconds_into_scenario"].astype(int)
        agg_df = succ_df.groupby(["service", "time_sec"])["success"].min().reset_index()
        agg_df["success_int"] = agg_df["success"].astype(int)

        pivot_df = agg_df.pivot(index="service", columns="time_sec", values="success_int")

        scen_label = ", ".join(label(s) for s in succ_df["scenario_id"].unique())
        fig_succ = px.imshow(
            pivot_df,
            color_continuous_scale=[[0.0, "red"], [1.0, "green"]],
            zmin=0, zmax=1,
            labels=dict(x="Seconds into Scenario (s)", y="Application", color="Status"),
            aspect="auto",
            title=f"Success Timeline (Scenarios: {scen_label})"
        )
        fig_succ.update_layout(coloraxis_showscale=False)
        fig_succ.update_traces(xgap=1, ygap=1, hovertemplate="Application: %{y}<br>Seconds: %{x}s<br>Status (1=Success, 0=Fail): %{z}<extra></extra>")
        st.plotly_chart(fig_succ, width='stretch')
    else:
        st.info("No data available for Success Plot.")

@st.dialog("Raw Log File", width="large")
def show_raw_log_modal(raw_text):
    st.code(raw_text, language="log")

def render_logs(log_data, scen_id_to_name=None):
    st.header("Scenario Logs")
    if not log_data:
        st.warning("No log files found in the `logs/` directory.")
        return

    #Scenario selector
    all_ids = [d["scenario_id"] for d in log_data]

    def scen_label(sid):
        if scen_id_to_name:
            name = scen_id_to_name.get(str(sid)) or scen_id_to_name.get(sid)
            if name:
                return f"Scenario {sid} – {name}"
        return f"Scenario {sid}"

    options = [scen_label(s) for s in all_ids]
    id_map  = {scen_label(s): s for s in all_ids}
    
    scen_col, btn_col = st.columns([0.85, 0.15])
    with scen_col:
        chosen  = st.selectbox("Select Scenario:", options, key="logs_scen")
        
    sid     = id_map[chosen]
    d       = next((x for x in log_data if x["scenario_id"] == sid), {})
    
    with btn_col:
        st.write("")
        st.write("")
        if st.button("View Raw .log", type="tertiary", use_container_width=True):
            if d:
                show_raw_log_modal(d.get("raw_text", ""))

    if not d:
        st.info("No data for this scenario.")
        return

    # Colour helpers 
    LEVEL_DOT = {
        "INFO": "#4a9eff",
        "WARNING": "#f59e0b",
        "WARN": "#f59e0b",
        "ERROR": "#ef4444",
        "CRITICAL": "#ef4444",
        "DEBUG": "#9ca3af",
    }

    job_ok      = d.get("job_status") is True
    badge_label = "Job passed" if job_ok else "Job failed"
    run_uuid    = d.get("run_uuid", "—")
    ts_raw      = d.get("timestamp", "")
    ts_disp     = ts_raw.replace("T", " ").replace("Z", " UTC") if ts_raw else "—"

    st.subheader(f"Krkn chaos run report")
    st.write(f"**Status:** {badge_label} | **UUID:** {run_uuid} | **Time:** {ts_disp}")

    # Top metrics row 
    scen_type   = d.get("scenario_type", "—")
    cluster_ver = d.get("cluster_version", "—")
    node_cnt    = d.get("total_node_count", 0)
    node_info   = d.get("node", {})
    arch        = node_info.get("architecture", "—")
    os_ver      = (node_info.get("os_version") or "—").replace("GNU/Linux ", "")
    exit_st     = d.get("exit_status", "—")
    duration    = d.get("duration", "—")

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Duration", duration, scen_type)
    m2.metric("Cluster version", cluster_ver, "Kubernetes")
    m3.metric("Total nodes", node_cnt, f"{arch} · {os_ver}")
    m4.metric("Scenarios run", 1, f"exit status {exit_st}")

    st.divider()

    # Details anf Affected pods (two columns)
    col_left, col_right = st.columns(2)

    with col_left:
        st.markdown("### Scenario details")
        params = d.get("scen_params", {}) or {}
        env    = d.get("env_vars", {}) or {}

        detail_map = {
            "Type":             d.get("scenario_type", env.get("SCENARIO_TYPE", "—")),
            "Action":           params.get("action", env.get("ACTION", "—")),
            "Namespace":        params.get("namespace", env.get("NAMESPACE", "—")),
            "Label selector":   params.get("label_selector", env.get("LABEL_SELECTOR", "—")),
            "Container name":   params.get("container_name", env.get("CONTAINER_NAME", "—")),
            "Disruption count": params.get("count", env.get("DISRUPTION_COUNT", "—")),
            "Recovery time":    (str(params.get("expected_recovery_time", env.get("EXPECTED_RECOVERY_TIME", "—"))) + "s").replace("—s", "—"),
            "Wait duration":    (str(env.get("WAIT_DURATION", "—")) + "s").replace("—s", "—"),
        }
        
        md_table = "| Field | Value |\n|---|---|\n"
        for k, v in detail_map.items():
            if v in (None, "None", "", "—"):
                v = "—"
            md_table += f"| **{k}** | {v} |\n"
        st.markdown(md_table)

    with col_right:
        st.markdown("### Affected pods")
        rec   = d.get("affected_recovered", 0)
        unrec = d.get("affected_unrecovered", 0)
        c1, c2 = st.columns(2)
        c1.metric("Recovered", rec)
        c2.metric("Unrecovered", unrec)

        k8s = d.get("k8s_objects", {})
        if k8s:
            st.markdown("### Cluster objects")
            k_cols = st.columns(len(k8s) or 1)
            for col, (k, v) in zip(k_cols, k8s.items()):
                col.metric(f"{k.lower()}s", v)

    st.divider()

    # Node info
    if node_info:
        st.markdown("### Node info")
        net = ", ".join(d.get("net_plugins", ["Unknown"]))
        kubelet = node_info.get("kubelet_version", "—")
        kernel  = node_info.get("kernel_version", "—")
        inst    = node_info.get("instance_type", "unknown")
        
        ni1, ni2, ni3 = st.columns(3)
        ni1.metric("Architecture", arch)
        ni2.metric("OS", os_ver)
        ni3.metric("Kernel", kernel)
        
        ni4, ni5, ni6 = st.columns(3)
        ni4.metric("Kubelet", kubelet)
        ni5.metric("Instance type", inst)
        ni6.metric("Network plugin", net)

    # Run Timeline 
    timeline = d.get("timeline", [])
    if timeline:
        st.markdown("### Run timeline")
        
        SKIP_FRAGS = ["✅ type:", "✅ types:"]
        key_lines  = [t for t in timeline if not any(f in t["msg"] for f in SKIP_FRAGS)]

        tl_text = ""
        for item in key_lines:
            ts = item["ts"]
            level = item["level"]
            msg = item["msg"]
            tl_text += f"{ts}  [{level}]  {msg}\n"
            
        with st.expander("View Timeline", expanded=True):
            st.code(tl_text, language="log")


def render_config(config_data):
    st.header("Krkn-AI Configuration")
    if config_data:
        st.json(config_data)
    else:
        st.write("Configuration file not found.")

def main():
    st.set_page_config(page_title="Krkn-AI Monitor", layout="wide")
    st.title("Krkn-AI Execution Monitor")

    monitor_config = get_monitor_config()
    base_output_dir = monitor_config.get("output_dir", "./")

    run_dirs = []
    if os.path.exists(base_output_dir) and os.path.isdir(base_output_dir):
        # Determine if base_output_dir is a parent folder mapping to UUIDs
        for item in os.listdir(base_output_dir):
            full_path = os.path.join(base_output_dir, item)
            if os.path.isdir(full_path):
                # A run directory will typically contain results.json or config.yaml
                if os.path.exists(os.path.join(full_path, "results.json")) or os.path.exists(os.path.join(full_path, "config.yaml")):
                    run_dirs.append(item)

    if run_dirs:
        # Sort by latest modified
        run_dirs.sort(key=lambda x: os.path.getmtime(os.path.join(base_output_dir, x)), reverse=True)
        st.sidebar.header("Select Run")
        selected_uuid = st.sidebar.selectbox("Run UUID:", run_dirs)
        output_dir = os.path.join(base_output_dir, selected_uuid)
        st.sidebar.divider()
    else:
        output_dir = base_output_dir

    # Detect state purely from lockfile (reliable across st.rerun() cycles)
    running = is_execution_running(output_dir)

    st.sidebar.header("Controls")
    if running:
        st.sidebar.info("⏳ Execution in progress...")
        auto_refresh = True
    else:
        st.sidebar.success("Execution completed!")
        auto_refresh = False

    # Load data
    df_results = load_results_csv(output_dir)
    config_data = load_config_yaml(output_dir)
    df_health = load_health_check_csv(output_dir)
    df_details = load_detailed_scenarios_data(output_dir)
    # df_best = load_best_scenarios_yaml(output_dir)
    df_logs = load_logs(output_dir)

    # Build scenario_id -> scenario_name lookup from results
    scen_id_to_name = {}
    if df_results is not None and not df_results.empty and "scenario_id" in df_results.columns and "scenario" in df_results.columns:
        for _, row in df_results[["scenario_id", "scenario"]].drop_duplicates().iterrows():
            sid = row["scenario_id"]
            scen_id_to_name[str(sid)] = row["scenario"]
            try:
                sid_int = int(float(sid))
                scen_id_to_name[str(sid_int)] = row["scenario"]
                scen_id_to_name[sid_int] = row["scenario"]
            except (ValueError, TypeError):
                pass

    #Global Filters 
    st.sidebar.header("Global Filters")

    # Collect all known scenario names, IDs and generations (from results CSV)
    all_scenario_names = []
    all_scenario_ids = []
    all_generations = []
    if df_results is not None and not df_results.empty:
        if "scenario" in df_results.columns:
            all_scenario_names = sorted(df_results["scenario"].unique().tolist())
        if "scenario_id" in df_results.columns:
            def safe_cast(v):
                try:
                    return int(float(v))
                except (ValueError, TypeError):
                    return str(v)
            raw_ids = df_results["scenario_id"].dropna().unique()
            sorted_raw = sorted(raw_ids, key=lambda x: (isinstance(safe_cast(x), str), safe_cast(x)))
            all_scenario_ids = [safe_cast(x) for x in sorted_raw]
        if "generation_id" in df_results.columns:
            all_generations = sorted([int(x) + 1 for x in df_results["generation_id"].dropna().unique()])

    global_generations = st.sidebar.multiselect(
        "Filter by Generation:",
        options=all_generations,
        default=[],
        help="Leave empty to show all generations across every tab."
    )

    global_scenarios_name = st.sidebar.multiselect(
        "Filter by Scenario Name:",
        options=all_scenario_names,
        default=[],
        help="Leave empty to show all scenarios across every tab."
    )
    
    global_scenarios_id = st.sidebar.multiselect(
        "Filter by Scenario Number:",
        options=all_scenario_ids,
        default=[],
        help="Leave empty to show all scenarios across every tab."
    )


    # Collect all known services (from health-check CSV + detailed scenarios)
    all_services = set()
    if df_health is not None and not df_health.empty and "component_name" in df_health.columns:
        all_services.update(df_health["component_name"].unique().tolist())
    if df_details is not None and not df_details.empty and "service" in df_details.columns:
        all_services.update(df_details["service"].unique().tolist())
    all_services = sorted(all_services)

    global_services = st.sidebar.multiselect(
        "Filter by Service:",
        options=all_services,
        default=[],
        help="Leave empty to show all services across every tab."
    )

    # Best Iterations Scope 
    filter_type = "All"
    SCORE_COLS = ["fitness_score", "health_check_failure_score", "health_check_response_time_score", "krkn_failure_score"]
    if df_results is not None and not df_results.empty:
        st.sidebar.subheader("Best Iterations Scope")
        available_score_cols = [c for c in SCORE_COLS if c in df_results.columns]
        sort_col = st.sidebar.selectbox(
            "Sort by:",
            options=available_score_cols,
            format_func=lambda c: {
                "fitness_score": "Fitness Score",
                "health_check_failure_score": "Health Check Failure Score",
                "health_check_response_time_score": "Health Check Response Time Score",
                "krkn_failure_score": "Krkn Failure Score",
            }.get(c, c),
            key="best_iter_sort_col",
        )
        filter_type = st.sidebar.radio("Filter Generator Rows:", ["All", "Top K scenarios by above score", "Top P(%) scenarios by above score"])

        if filter_type == "Top K scenarios by above score":
            k_value = st.sidebar.number_input("Top K count:", min_value=1, value=3, step=1)
            df_results = df_results.sort_values(by=sort_col, ascending=False).head(int(k_value))
        elif filter_type == "Top P(%) scenarios by above score":
            p_value = st.sidebar.slider("Top Percentage (%):", min_value=1, max_value=100, value=25)
            cutoff = max(1, int(len(df_results) * (p_value / 100.0)))
            df_results = df_results.sort_values(by=sort_col, ascending=False).head(cutoff)

    df_results_all = df_results  # keeping a reference to unfiltered dataframe if needed later!!
    df_failed = None
    if df_results is not None and not df_results.empty:
        # Separating failed scenarios.
        # If krkn_failure_score < 0, it's considered a misconfiguration or krkn engine failure.
        if "krkn_failure_score" in df_results.columns:
            # krkn_failure_score < 0 means krkn engine failed / misconfiguration
            mask_failed = df_results["krkn_failure_score"] < 0
            df_failed = df_results[mask_failed]
            df_results = df_results[~mask_failed]
        else:
            df_failed = pd.DataFrame()

    # Apply global scenario filters to results
    active_scenario_names = global_scenarios_name if global_scenarios_name else all_scenario_names
    active_scenario_ids = global_scenarios_id if global_scenarios_id else all_scenario_ids
    active_generations = global_generations if global_generations else all_generations
    
    if df_results is not None and not df_results.empty:
        if active_scenario_names:
            df_results = df_results[df_results["scenario"].isin(active_scenario_names)]
        if active_scenario_ids:
            str_ids = [str(x) for x in active_scenario_ids]
            df_results = df_results[df_results["scenario_id"].astype(str).isin(str_ids)]
        if active_generations and "generation_id" in df_results.columns:
            df_results = df_results[(df_results["generation_id"] + 1).isin(active_generations)]
            
    if df_failed is not None and not df_failed.empty:
        if active_scenario_names:
            df_failed = df_failed[df_failed["scenario"].isin(active_scenario_names)]
        if active_scenario_ids:
            str_ids = [str(x) for x in active_scenario_ids]
            df_failed = df_failed[df_failed["scenario_id"].astype(str).isin(str_ids)]
        if active_generations and "generation_id" in df_failed.columns:
            df_failed = df_failed[(df_failed["generation_id"] + 1).isin(active_generations)]

    # Derive the filtered scenario IDs for cross-tab consistency (from successful runs only)
    filtered_scenario_ids = (
        df_results["scenario_id"].unique().tolist()
        if df_results is not None and not df_results.empty and "scenario_id" in df_results.columns
        else []
    )

    # Apply global scenario filter to health-check CSV
    if df_health is not None and not df_health.empty and filtered_scenario_ids:
        str_ids = [str(x) for x in filtered_scenario_ids]
        df_health = df_health[df_health["scenario_id"].astype(str).isin(str_ids)]

    # Apply global scenario filter to detailed scenarios CSV (scenario_id stored as str there)
    if df_details is not None and not df_details.empty and filtered_scenario_ids:
        str_ids = [str(x) for x in filtered_scenario_ids]
        df_details = df_details[df_details["scenario_id"].astype(str).isin(str_ids)]

    # Tabs
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "Dashboard", "Health Checks", "Detailed Scenarios", "Logs", "Configuration", "Failed Scenarios"
    ])

    with tab1:
        if df_results is None or df_results.empty:
            st.warning(f"Waiting for scenario reports in `{output_dir}/reports/all.csv`...")
        else:
            render_summary(df_results)
            st.divider()

            colA, colB = st.columns(2)
            with colA:
                render_scenario_distribution(df_results)
            with colB:
                render_scenario_fitness_variation(df_results)

            st.divider()
            render_fitness_evolution(df_results)
            st.divider()
            render_generation_details(df_results)

    with tab2:
        render_health_checks(df_health, global_services=global_services if global_services else None)

    with tab3:
        # render_best_scenarios_summary(df_best)
        render_detailed_scenarios(
            df_details,
            global_scenarios=filtered_scenario_ids if filtered_scenario_ids else None,
            global_services=global_services if global_services else None,
            scen_id_to_name=scen_id_to_name,
        )

    with tab4:
        render_logs(df_logs, scen_id_to_name=scen_id_to_name)

    with tab5:
        render_config(config_data)

    with tab6:
        render_generation_details(df_failed, title="Failed Scenarios")

    # Refresh mechanism
    if auto_refresh:
        time.sleep(3)
        st.rerun()


if __name__ == "__main__":
    main()
