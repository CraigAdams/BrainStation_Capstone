#!/usr/bin/env python
# coding: utf-8

# In[1]:


"""
Title: Picking the Best Predictive Model: A Case Study in Analytics Operationalization
Author: Craig Adams
Contact: craig.adams@elementaryinsights.com

The following application loads data from Part 4 of the project, "Part 4. Assessment of Predictive Model Performance, 
specifically the file: “failure_test_with_models_threshold10.csv”.  The app allows the user to interact with the model outputs,
select an engine, adjust the hard threshold for the DT3 decision tree and LogReg logistic regression models generated from Part 3
and compare their ability to predict the failure of the engines relative to the Sensor Alert and the Weibull curve alert 
generates from Part 2. 

"""


# In[2]:

import pandas as pd
import numpy as np
#import matplotlib.pyplot as plt
import plotly.graph_objects as go
#import plotly.express as px
from plotly.subplots import make_subplots

from dash import html, dcc, Input, Output, Dash


# In[3]:

# Import the data from the project directory
results_df = pd.read_csv('./data/failure_test_with_models_threshold10.csv')



# In[4]:
# Establish baseline threshold

# Set initial soft threshold at 0.5
results_df['dtree_alert'] = np.where(results_df['dtree_proba'] >= 0.5, 1,0)
results_df['logit_alert'] = np.where(results_df['logit_proba'] >= 0.5, 1,0)

# Add Weibull alert as well
results_df['weibull_alert'] = np.where(results_df['cycle_time'] >= 130, 1,0)

print(results_df.tail(12))


# In[5]:

# Set engine and threshold lists
engines = results_df['unit_number'].unique()
thresholds = [0.4,0.5,0.6,0.7,0.8,0.9,0.95]


# In[8]:

# Create an app in Dash

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = Dash(__name__, external_stylesheets=external_stylesheets)

# Define Layout
app.layout = html.Div(
    children=[
        # Add a title
        html.H2("Demonstration of The Business Value of Different Data Science Modeling Approaches", style={
                                'text-align':'left',
                                'background':'lightgrey'}),
        # Set up the select engine field
        html.P("This decision support tool shows the relative operating efficiencies that would be experienced for\
                the test engines using the different modeling techniques that send an alert when the engine is about to fail.\
                Each engine can be selected and the Alert Threshold adjusted for the two ML models, DT3 and Logit.\
                The operating efficiencies for each alerting system are updated at the bottom.  Be careful:  If you push it too far, your engine might fail!"),
        html.H4('Inputs:', style = {'background':'lightgrey'}),
        html.Label([
            "Engine Selector", dcc.Dropdown(id ='Engine_dropdown', clearable=False, multi=False, value = engines[0],
                                            options = [{'label': engine, 'value':engine} for engine in engines],
                                              style ={'height':'40px','width':'100px'})
                                                       
        ]),
        # Set up the select threshold field
         html.Label([
            "Threshold Selector", dcc.Dropdown(id ='Threshold_dropdown', clearable=False, multi=False, value= thresholds[1],
                                            options = [{'label': threshold, 'value':threshold} for threshold in thresholds],
                                              style ={'height':'40px','width':'100px'})
        ]), 
    
        # Set up the graph
        html.H4('Results:', style = {'background':'lightgrey'}),
        dcc.Graph(id='graph', figure ={}),  # placehold for fig to write back to
        
        # Set up the table
        html.H4('Operating Efficiency Values:', style = {'background':'lightgrey'}),
        # Generate the results table
        html.Table([
            html.Tr([html.Td(['Weibull Efficiency']), html.Td(id='weibull_op_eff'),html.Td(['%']) ]),
            html.Tr([html.Td(['Sensor Efficiency']), html.Td(id='sensor_op_eff'),html.Td(['%'])]),
            html.Tr([html.Td(['DT3 Efficiency']), html.Td(id='d_tree_op_eff'),html.Td(['%'])]),
            html.Tr([html.Td(['Logit Efficiency']), html.Td(id='logit_op_eff'),html.Td(['%'])],
                   style={'margin':'10px'})
            
        ]) 
])

  
# Callback to change the threshold for the engine and in the background, all other engines so they are ready

@app.callback(
    # Estalish graph outputs for updating
    [Output(component_id='graph', component_property='figure'),
    # Establish table outputs for udpdating
    Output('weibull_op_eff', 'children'),
    Output('sensor_op_eff', 'children'),
    Output('d_tree_op_eff', 'children'),
    Output('logit_op_eff', 'children')],
    # Confirm function inputs that will drive the output updates
    [Input(component_id='Engine_dropdown', component_property='value'),
    Input(component_id='Threshold_dropdown', component_property='value')]
)    

# Function to update the graph and table
def update_engine_and_flags(engine, threshold):  # First input value maps to the first argument, second to second
    results_dff = results_df.copy()
#     # Update the thresholds if needed    
    results_dff['dtree_alert'] = np.where(results_dff['dtree_proba'] >= threshold, 1,0)
    results_dff['logit_alert'] = np.where(results_dff['logit_proba'] >= threshold, 1,0)

    columns = ['dtree_alert','logit_alert', 'sensor_11_alert']

    for column in columns:
        # Find the first instance index value
        condition = (results_dff['unit_number'] ==engine)&(results_dff[column] ==1)
        idx = (condition).idxmax()
        # replace the current value with the correct value     
        results_dff.loc[((results_dff['unit_number'] ==engine)&(results_dff.index >= idx)), column] = 1
        
    # Update dataset
    engine_df = results_dff.loc[(results_dff['unit_number']==engine)]
    
    # Calculate the Op Efficiency for each track
    total_sensor_cycles = engine_df['sensor_11_alert'].sum()-1
    total_dtree_cycles = engine_df['dtree_alert'].sum()-1
    total_logit_cycles = engine_df['logit_alert'].sum()-1
    total_weibull_cycles = engine_df['weibull_alert'].sum()-1
    total_max_cycles = engine_df['RUL'].shape[0]-1

    weibull_op_eff =100-round(total_weibull_cycles/total_max_cycles*100,1)
    sensor_op_eff = 100-round(total_sensor_cycles/total_max_cycles*100,1)
    d_tree_op_eff = 100-round(total_dtree_cycles/total_max_cycles*100,1)
    logit_op_eff = 100-round(total_logit_cycles/total_max_cycles*100,1)
    
    # Check for engine failure
    if weibull_op_eff ==  total_max_cycles:
        weibull_op_eff =0
    else:
        pass
    
    if sensor_op_eff ==  total_max_cycles:
        sensor_op_eff =0
    else:
        pass
    
    if d_tree_op_eff ==  total_max_cycles:
           d_tree_op_eff =0
    else:
        pass
    
    if logit_op_eff ==  total_max_cycles:
           logit_op_eff =0
    else:
        pass
        
# Update plot
    # Set up the Figure 
    
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # Now add the scatter points for each alert individually 
    # Create the graph data objects
    chart_data1=go.Scatter(x=engine_df['RUL'],y=engine_df['sensor_11_alert'], 
                             mode='lines', name='Sensor alert', line_color='orangered', line_width=4)
    chart_data2=go.Scatter(x=engine_df['RUL'],y=engine_df['dtree_alert'], 
                             mode='lines', name='DT3 alert', line_color='forestgreen', line_width=4)
    chart_data3=go.Scatter(x=engine_df['RUL'],y=engine_df['logit_alert'], 
                             mode='lines', name='Logit alert',line_color='mediumblue', line_width=4)
    chart_data4=go.Scatter(x=engine_df['RUL'],y=engine_df['weibull_alert'], 
                         mode='lines+markers', name='Weibull alert',line_color='goldenrod',line_width=4)
    chart_data5=go.Scatter(x=engine_df['RUL'],y=engine_df['Failure_Threshold'], 
                             mode='lines', name='Failure threshold', line_color='black', line_width=4)
    chart_data6=go.Scatter(x=engine_df['RUL'],y=engine_df['sensor_11'], 
                             mode='lines+markers', name='Sensor 11', line_color='#BCCCDC', line_width=4)

    # Use add_trace()
    fig.add_trace(chart_data1, secondary_y=True)
    fig.add_trace(chart_data2, secondary_y=True)
    fig.add_trace(chart_data3, secondary_y=True)
    fig.add_trace(chart_data4, secondary_y=True)
    fig.add_trace(chart_data5, secondary_y=True)
    fig.add_trace(chart_data6)

    # Create the layout object and add it to the figure
    chart_layout=go.Layout(width=900,height=500, plot_bgcolor="#FFFFFF",
                          title='Engine Lifecycle Duration vs Alerts For Test Data',
                          xaxis=dict(title="Remaining Useful Life", linecolor="#BCCCDC"),
                          yaxis=dict(title="Sensor Value", linecolor="#BCCCDC"),
                          yaxis2=dict(title="Alert Status", linecolor="#BCCCDC")
                          )
    
    fig.update_layout(chart_layout)
    fig.update_xaxes(title_text = "Remaining Useful Life", range =(150,0))
    
    # Return results
     
    return fig, weibull_op_eff,sensor_op_eff,d_tree_op_eff,logit_op_eff

           
# Run app and display results inline in the notebook
if __name__ =='__main__':
    app.run_server(debug=False, port = 8088)



# In[ ]:


# Steps to migrate from Jupyter

#1. Download as a .py file and move it to the same directory as the data
#2. Change Jupyter Dash for Dash in the imports and on the app Jupyter part
#3. change the launch code at the bottom to external

# if __name__ = '__main__':
#     app.run_server(debug=False, port = 8088)





