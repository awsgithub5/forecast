import streamlit as st
import pandas as pd
import sys
import os

# importing forecast notebook utility from notebooks/common directory
#sys.path.insert( 0, os.path.abspath("../../common") )
import util
import util.fcst_utils
import boto3
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

plt.rcParams['figure.figsize'] = (15.0, 5.0)
PROJECT = 'walmart'
DATA_VERSION = 5
dataset_group = 'walmart_5'
dataset_group_arn = 'arn:aws:forecast:us-east-1:354360995214:dataset-group/walmart_5'
dataset_arns = []

region = boto3.Session().region_name
session = boto3.Session(region_name=region) 
forecast = session.client(service_name='forecast') 
forecastquery = session.client(service_name='forecastquery')
st.title('Sales Forecasting App')

#st.header('This is a header')

st.subheader('File uploader:')

uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

def plot_forecasts_wrapper(*args, **kwargs):
    util.fcst_utils.plot_forecasts(*args, **kwargs)

def extract_summary_metrics(metric_response, predictor_name):
    df = pd.DataFrame(metric_response['PredictorEvaluationResults']
                 [0]['TestWindows'][0]['Metrics']['WeightedQuantileLosses'])
    df['Predictor'] = predictor_name
    return df

if uploaded_file is not None:
  df = pd.read_csv(uploaded_file)
  
  st.dataframe(df)
  
  df['Weekly_Sales'] = df['Weekly_Sales'].astype('float')
  walmart = df.rename(columns={'Date': 'datetime'})
  st.write("Minimum Date: ",walmart.datetime.min())
  st.write("Maximum Date: ",walmart.datetime.max())
  walmart['Store'] = ['Store_' + str(i) for i in walmart['Store']]
  walmart['Dept'] = ['Dept_' + str(i) for i in walmart['Dept']]
  walmart['item_id'] = walmart[['Store', 'Dept']].apply(lambda x: '_'.join(x), axis=1)
  walmart.drop(['Store', 'Dept','MarkDown2','MarkDown3','MarkDown4','MarkDown5'], axis=1, inplace=True)
  all_ts = walmart['item_id'].unique()
  st.title('Time Series Plots')

  show_plots = st.button('Show Plots')

  if show_plots:

      for i in tqdm(all_ts):
        # create plot for item i 
        df_subset = walmart[walmart['item_id'] == i]
        fig = df_subset.plot(x='datetime', y='Weekly_Sales',
                             title=i, figsize=(15,8)).get_figure()    

        st.pyplot(fig)

  else:
      st.write('Click button to show plots')
        
  st.title('Forecast Configuration')

  FORECAST_LENGTH = st.number_input('Forecast Length', min_value=1, value=20)

  freq = st.selectbox('Frequency', ['H', 'D', 'W', 'M', 'Y'])  

  timestamp_format = st.text_input('Timestamp Format', 'yyyy-MM-dd')
  if st.button('Submit'):
    st.write('Forecast Config')
    st.write(f'- Length: {FORECAST_LENGTH}') 
    st.write(f'- Frequency: {freq}')
    st.write(f'- Timestamp: {timestamp_format}')
    
  target_df = walmart[['item_id', 'datetime', 'Weekly_Sales']][:-FORECAST_LENGTH]
  rts_df = walmart[['item_id', 'datetime','IsHoliday','Temperature', 'Fuel_Price',
       'MarkDown1', 'CPI', 'Unemployment', 'Type', 'Size']]
  #print(f"{len(target_df)} + {FORECAST_LENGTH} = {len(rts_df)}")
  assert len(target_df) + FORECAST_LENGTH == len(rts_df), "length doesn't match"
  target_df.to_csv("walmart_target.csv", index= False, header = False)
  rts_df.to_csv("walmart_rts.csv", index= False, header = False)
  


  algorithm = st.selectbox(
    'Select algorithm',
    ('Prophet', 'Deep_AR_Plus')
) 
  predictor_arn_deep_ar = 'arn:aws:forecast:us-east-1:354360995214:predictor/'+dataset_group+'_deep_ar_plus'
  error_metrics_deep_ar_plus = forecast.get_accuracy_metrics(PredictorArn=predictor_arn_deep_ar)
  predictor_arn_prophet = 'arn:aws:forecast:us-east-1:354360995214:predictor/'+dataset_group+'_prophet'
  error_metrics_prophet = forecast.get_accuracy_metrics(PredictorArn=predictor_arn_prophet)
  st.title("Accuracy Metrics")
  if algorithm == 'Deep_AR_Plus':
    
    st.write(error_metrics_deep_ar_plus)    
    
  else:
    st.write(error_metrics_prophet)
    
  #st.button("Summary_Metrics Comparison")  
  if st.button('Summary Metrics Comparison'):
    deep_ar_metrics = extract_summary_metrics(error_metrics_deep_ar_plus, "DeepAR")
    prophet_metrics = extract_summary_metrics(error_metrics_prophet, "Prophet")
    metrics_df = pd.concat([deep_ar_metrics, prophet_metrics])
    pivoted_df = metrics_df.pivot(index='Quantile', columns='Predictor', values='LossValue')
    ax = pivoted_df.plot.bar(figsize=(5,3), rot=0) 
    ax.set_xlabel('Quantile')
    ax.set_ylabel('Loss Value')
    fig = ax.get_figure()  
    st.pyplot(fig)

       
  st.title('Create a forecast')    

  st.header('Store and Department Selection')

  store = st.selectbox('Select Store:', ('1', '2'))
  dept = st.selectbox('Select Department:', ('1', '2'))

  item_id = 'Store_'+store+'_Dept_'+dept
     
    
    
  algorithm = st.selectbox(
    'Select an algorithm for forecasting',
    ('Prophet', 'Deep_AR_Plus')
)       
  fname = f'walmart_target.csv'
  exact = util.fcst_utils.load_exact_sol(fname, item_id)
  
  if algorithm == 'Deep_AR_Plus':
    forecast_arn_deep_ar = 'arn:aws:forecast:us-east-1:354360995214:forecast/'+dataset_group+'_deep_ar_plus'
    forecast_response_deep = forecastquery.query_forecast(ForecastArn=forecast_arn_deep_ar,Filters={"item_id": item_id})
    
    fig, ax = plt.subplots()
    plot_forecasts_wrapper(forecast_response_deep, exact)


    st.subheader('DeepAR Forecast')  
    st.pyplot(fig)
      
  else:
    forecast_arn_prophet = 'arn:aws:forecast:us-east-1:354360995214:forecast/'+dataset_group+'_prophet'
    forecast_response_prophet = forecastquery.query_forecast(ForecastArn=forecast_arn_prophet,
                                                     Filters={"item_id":item_id})
    fig, ax = plt.subplots()
    plot_forecasts_wrapper(forecast_response_prophet, exact)


    st.subheader('Prophet Forecast')  
    st.pyplot(fig)
  
    

 