import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy.optimize import curve_fit


# Configure main page
st.set_page_config(layout="wide", page_title="Hay Quality Analysis")

@st.cache_data(persist=True)
def load_data():
    # Read raw data as pandas dfs
    probe_hdd_df = pd.read_excel("probe_data.xlsx")
    probe_temp_df = pd.read_csv("AF0003706_meas.csv")

    #merged_df = probe_temp_df.merge(probe_hdd_df)
    merged_df = probe_hdd_df
    return merged_df, probe_temp_df

# Page title and text content
st.markdown('<div style="text-align: center; font-size: 4em;"><b>Hay Quality Analysis</b></div>', unsafe_allow_html=True)
st.markdown('<div style="text-align: justify; font-size: 2em;"></div>', unsafe_allow_html=True)

# Load the data using the cached function
merged_df, probe_temp_df = load_data()

col11, col11a, col22, col22a, col33 = st.columns([0.15, 0.25, 0.15, 0.75, 0.15])

with col11a:
    show_table = st.radio("Show table", ('temperature measurements (probes)', 'aggregated HDD (probes)'))

    if show_table == 'temperature measurements (probes)':
        with col22a:
            st.table(probe_temp_df.head())
    else:
        with col22a:
            st.table(merged_df.head())
# Add containers
#col1, col1a, col2, col2a, col3 = st.columns([0.25, 0.15, 0.75, 0.15, 0.75])

col1, col1a, col2, col2a, col3 = st.columns([0.15, 0.25, 0.15, 0.75, 0.15])

# Add checkbox and multiselect menu options
unique_pb_ids = merged_df['PB_ID'].unique()

with col1a:
    select_all = st.checkbox("Select All")
    selected_pb_ids = []
    if select_all:
        selected_pb_ids = list(unique_pb_ids)
    else:
        selected_pb_ids = st.multiselect("Select PB_IDs", unique_pb_ids)

# Filter df based on user selection
filtered_df = merged_df[merged_df['PB_ID'].isin(selected_pb_ids)]
#filtered_df['b'] = pd.to_datetime(filtered_df['b'], unit='s')

# Initialise and plot figure
fig = go.Figure()
#fig2 = go.Figure()
for index, row in filtered_df.iterrows():
    pb_id = row['PB_ID']
    cp1 = row['CP1']
    hdd = row['HDD']
    cp2 = row['CP2']
    hdd_values = np.arange(0, int(hdd), 15)
    curve_value = (cp1 * np.exp(-0.000000175 * hdd * hdd)) - cp2
    curve_value2 = [(cp1 + cp2 * np.exp(-0.000000175 * x * x)) + curve_value for x in hdd_values]
    diff = curve_value - curve_value2
    
    def curve_model(hdd, cp1, cp2):
        return (cp1 * np.exp(-0.000000175 * hdd * hdd)) - cp2

    # Fit the curve model to the data
    popt, _ = curve_fit(curve_model, hdd_values, curve_value2)

    # Get the fitted coefficients
    fitted_cp1 = popt[0]
    fitted_cp2 = popt[1]

    # Print or store the fitted coefficients for each plot
    st.write(f"Probe ID: {pb_id}, crude protein % (analysis 1): {round(fitted_cp1, 2)}, crude protein % (analysis 2): {round(fitted_cp2, 2)}, distance from prediction: {round(diff[-1], 2)}")

    # threshold = 0.1  # Define the convergence threshold

    # # Find the average HDD for which the curve becomes an asymptote
    # asymptote_hdd = None
    # for hdd, curve_value2 in zip(hdd_values, curve_value2):
    #     if abs(curve_value2) < threshold:
    #         asymptote_hdd = hdd
    #         break

    # st.write(f"HDD for asymptote: {asymptote_hdd}")

    # # Determine color based on temperature
    # temp = row['h']  # Assuming 'temp' is the column name in your dataframe
    # line_color = 'green' if temp < 30 else 'orange' if 30 <= temp <= 50 else 'red'

    # # Create y-values for the line plot
    # y_values = [temp] * len(filtered_df['b'])

    # st.write(f'Probe ID: 0{pb_id}, distance from prediction: {round(diff[-1], 2)}')
    fig.add_trace(go.Scatter(
        x=[0, hdd],
        y=[cp1, curve_value],
        mode='markers',
        name=f'Probe ID: 0{pb_id}',
        showlegend=False
    ))
    fig.add_trace(go.Scatter(
        x=hdd_values,
        y=curve_value2,
        # y=[cp1] + curve_value2[:-1],
        mode='lines',
        name=f'Probe ID: {pb_id}',
        showlegend=False
    ))
    fig.add_trace(go.Scatter(
        x=[hdd, hdd],
        y=[curve_value, curve_value2[-1]],
        line=dict(color='lightgrey', width=2, dash='dash'),
        showlegend=False
    ))
    # fig2.add_trace(go.Scatter(
    #     x=filtered_df['b'],
    #     y=y_values,
    #     mode='markers',
    #     #line=dict(color=line_color),
    #     showlegend=False
    # ))
fig.update_layout(
    title="",
    xaxis_title="Heating degree days",
    xaxis=dict(
        showgrid=False,
        showline=True,
    ),
    yaxis_title="change in crude protein (%)",
    yaxis=dict(
        showgrid=False,
        showline=True,
    ),
    autosize=False,
    width=1200,
    height=600,
)
# fig2.update_layout(
#     title="",
#     xaxis_title="",
#     xaxis=dict(
#         showgrid=False,
#         showline=True,
#     ),
#     yaxis_title="temperature (°C)",
#     yaxis=dict(
#         showgrid=False,
#         showline=True,
#     ),
#     autosize=False,
#     width=800,
#     height=500,
# )

with col2a:
    st.plotly_chart(fig)

# with col3:
#     st.plotly_chart(fig2)
