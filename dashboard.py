import streamlit as st
import predict
import plotly.graph_objects as go
from darts import TimeSeries

st.set_page_config(layout="wide", initial_sidebar_state='expanded')
st.title('COVID-19 deaths prediction - France')

# def plot_death_prediction(deaths_so_far, deaths_prediction):
yy, y_pred = predict.main()


y = yy.append(TimeSeries.from_series(y_pred)[0])

fig = go.Figure()
fig.add_trace(go.Scatter(x=y_pred.index, y=y_pred.values.reshape(-1), name="Deaths predictions",
                    line_shape='linear'))
fig.add_trace(go.Scatter(x=y[-20:].time_index, y=y[-20:].values().reshape(-1), name="Past deaths",
                    line_shape='linear'))

st.plotly_chart(fig)

