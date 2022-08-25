import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
import predict
import plotly.graph_objects as go
from darts import TimeSeries
from dashboard.info import info_text


def predict_deaths():
    deaths_so_far_minus, deaths_prediction = predict.main()
    deaths_so_far = deaths_so_far_minus.append(TimeSeries.from_series(deaths_prediction)[0])
    return(deaths_so_far, deaths_prediction)

def plot_death_prediction(deaths_so_far, deaths_prediction):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=deaths_prediction.index, y=deaths_prediction.values.reshape(-1), name="Deaths predictions",
                        line_shape='linear'))
    fig.add_trace(go.Scatter(x=deaths_so_far[-20:].time_index, y=deaths_so_far[-20:].values().reshape(-1), name="Past deaths",
                        line_shape='linear'))

    return(fig)

def main():
    deaths_so_far, deaths_prediction = predict_deaths()
    fig = plot_death_prediction(deaths_so_far, deaths_prediction)
    fig.update_layout(hovermode="x")
    fig.update_layout(width=1000)

    return fig


if __name__ == '__main__':
    st.set_page_config(layout="wide", initial_sidebar_state='expanded')
    st.title(' 💊 COVID-19 deaths prediction - France 💊')
    st.sidebar.write(info_text)
    plot_spot = st.empty()
    update_button = st.button('Update')
    fig = main()

    if update_button:
        fig = main()
    
    with plot_spot:
        st.plotly_chart(fig)

        

