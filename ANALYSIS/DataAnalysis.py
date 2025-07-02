# ANALYZE DATA AND EVALUTE POSSIBLE MODEL TO REBUILD THE MISSING DATA
# 1

import pandas as pd
import plotly.express as px
from src.utilis import detect_outliers, scatter_with_regression
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from scipy.stats import linregress
# ============================================================================
#                               GET DATASET
# ============================================================================
file = ".../BCFT.csv"
df =  pd.read_csv(file,sep=";", decimal=",",index_col=0 )
df.index = pd.to_datetime(df.index, format="%d/%m/%Y %H:%M:%S")
df.columns
fig = px.line(df, x=df.index, y="Internal temperature area 1 (Celsius degree)")
fig.show()

# ============================================================================
#                               CLEAN DATASET
# ============================================================================


def clean_series(df, name_series, method_detection_out):
    '''
    Clean series.
    Possibility to clean the dataset multiple time 
    '''
    # REMOVE NAN values
    df_ = df[df[f"{name_series}"].notna()]
    # DETECT OUTLIERS
    # method_1:  Using cluster algorithm
    outliers_df_ = detect_outliers(df_.loc[:,[f"{name_series}"]], name_series, method=method_detection_out)
    outliers_df_.columns = ['outliers_']
    # add outliers column to original dataset
    df_with_outliers = pd.merge(df_,outliers_df_, on='Date and time', how='outer')

    # Visualize: time series + outliers
    fig = px.line(df_with_outliers,x=df_with_outliers.index, y=f"{name_series}", title='')
    fig.add_trace(px.scatter(df_with_outliers, x=df_with_outliers.index, y='outliers_').data[0])

    # Remove outliers
    df_clean = df_with_outliers[df_with_outliers['outliers_'].isna()]
    # Visualize graph with removed outliers
    fig_clean = px.line(df_clean,x=df_clean.index, y=f"{name_series}")

    return df_with_outliers, fig, df_clean, fig_clean 

# APPLY BOTH METHOD IN TWO STEPS
methods_detection_out = ['DBSCAN','Z_SCORE']
dataset_cleaned = []
df_cleaned = df
for name in df.columns:
    for i in range(2):
        result = clean_series(df_cleaned, name ,methods_detection_out[i])
        df_cleaned = result[2].drop(columns="outliers_")
    dataset_cleaned.append(df_cleaned[f'{name}'])

df_dataset_cleaned = pd.DataFrame(dataset_cleaned).T

# REINDEX DATASET
# Specify the full date range you want in the result
full_date_rng = pd.date_range(start=df_dataset_cleaned.index[0], end=df_dataset_cleaned.index[-1], freq='15min')
# Reindex the DataFrame to include all dates and fill missing values with NaN
df_reindexed = df_dataset_cleaned.reindex(full_date_rng)


# VISUALIZE DATASET
from pyecharts import options as opts
from pyecharts.charts import Scatter, Line

timeFormat = '%Y-%m-%d %H:%M:%S'
name_y_axes = 'Temp'
name_series_y2 = 'HVAC power (kW)' 

def line_chart(df:pd.DataFrame, timeFormat:str, name_y_axes:str, opt_chart:str):
    '''
    Line_chart:
    opt1: single line chart
    opt2: line chart with 2 y-axes
    Param
    ------
    df: dataframe with DateTime index
    timeFormat: format of time index %Y-%m-%d %H:%M:%S
    name_y_axes: name of temperature y_axes 
    name_series_y2: name of the df-column for 2nd axes

    '''

    dfPlot = df_reindexed
    dfPlot.index = dfPlot.index.strftime(timeFormat)
    seriesName = dfPlot.columns[:-1]
    x_data = dfPlot.index.tolist()
    y_data = dfPlot[seriesName].T.values.tolist()


    line_chart = (
        Line()
        .add_xaxis(x_data)
        .add_yaxis(
            color="#675bba",
            series_name=name_series_y2, 
            y_axis = dfPlot[''].values.tolist(),
            yaxis_index=1
        )
        .extend_axis(
            yaxis=opts.AxisOpts(
                name="Power",
                type_="value",
                min_=0,
                max_=15,
                position="right",
                axisline_opts=opts.AxisLineOpts(
                    linestyle_opts=opts.LineStyleOpts(color="#d14a61")
                ),
                axislabel_opts=opts.LabelOpts(formatter="{value} kW"),
            )
        )
        .set_series_opts(label_opts=opts.LabelOpts(is_show=False))
        .set_global_opts(
            toolbox_opts=opts.ToolboxOpts(
                feature=opts.ToolBoxFeatureOpts(
                    save_as_image=opts.ToolBoxFeatureSaveAsImageOpts(title="Download as Image"),
                    restore=opts.ToolBoxFeatureRestoreOpts(title="Restore"),
                    data_view=opts.ToolBoxFeatureDataViewOpts(title="View Data", lang=["Data View", "Close", "Refresh"]),
                    data_zoom=opts.ToolBoxFeatureDataZoomOpts(zoom_title="Zoom In",back_title="Zoom Out"),
                    magic_type=opts.ToolBoxFeatureMagicTypeOpts(is_show=False)
                )
            ),
            tooltip_opts=opts.TooltipOpts(trigger="axis"),
        )
    )
    n=0
    for data in y_data:
        line_chart.add_yaxis(
            series_name=seriesName[n], 
            yaxis_index=0,
            y_axis=data, 
            is_smooth=True,
            symbol="emptyCircle",
            is_symbol_show=False,
            label_opts=opts.LabelOpts(is_show=True),
        )
        n=n+1
        print(n)
    line_chart.height = "500px"
    line_chart.width = "2000px"
    line_chart.render('test.html')



# VISUALIZE
df_plot = df_dataset_cleaned.reset_index()
import plotly.graph_objects as go
fig_tot = go.Figure()
for name in df_plot.columns[1:]:
    fig_tot.add_trace(px.scatter(df_plot,x='Date and time', y=name).data[0])
    # fig.add_trace(go.Scatter(df_plot,x='Date and time', y=name, mode='lines'))


# ==============================================================================
#                           MODEL OF HAVC POWER - REGRESSION 
# ==============================================================================
# Model 1: Regression and multiple regression lines

def simple_regression_two_variables(df:pd.DataFrame, XName:str, yName:str):
    '''
    simple regression between two variables
    Param
    ------
    df = pandas dataframe
    X =  name ofthe column in the df dataframe as predictor (XAxes)
    Y =  name ofthe column in the df dataframe as to be predicted (YAxex)
    '''
    df_clean = df.loc[:,[f"{XName}",f"{yName}"]].dropna()
    X = df_clean[f"{XName}"].to_numpy()
    y = df_clean[f"{yName}"].to_numpy()

    slope, intercept, r_value, p_value, std_err = linregress(X, y)
    # Calculate R-squared
    r_squared = r_value**2
    # Generate predicted y values based on the regression line
    regression_line = [slope * x + intercept for x in X]
    regression_line = [round(element, 2) for element in regression_line]

    return r_squared,regression_line


simple_regression_two_variables(df_dataset_cleaned, 'External temperature (Celsius degree)', 'HVAC power (kW)')
# Plot simple regression 
df_dataset_cleaned = df_dataset_cleaned.dropna()
x_data = df_dataset_cleaned['External temperature (Celsius degree)'].tolist()
y_data = [df_dataset_cleaned['HVAC power (kW)'].tolist()]
colorPoints = 'red'
colorRegression = 'black'
seriesName = 'HVAC power (kW)'
graphName = 'linearRegression.html'
name_y_axes = 'Power'
scatter_with_regression(x_data, y_data, colorPoints, colorRegression, 
                        seriesName,name_y_axes, graphName)


# ==============================================================================
#                           MODEL OF HAVC POWER - LSTM 
# ==============================================================================


# ============================================================================================================
#                                       NEURAL NETWORK - LSTM
# ============================================================================================================
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error,r2_score
import keras

class LossHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.losses = []

    def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))

# remove GLobal power
df_HVAC_power = df_dataset_cleaned.drop(['Global power (kW)'], axis=1)
combined_ = df_HVAC_power.dropna()
# combined_TH_['year'] = list(combined_TH_.index.year)
combined_['month'] = list(combined_.index.month)
combined_['day'] = list(combined_.index.day)
combined_['hour'] = list(combined_.index.hour)
X = combined_.drop(['HVAC power (kW)',], axis=1)
y = combined_['HVAC power (kW)']

# Split DATA
# tscv = TimeSeriesSplit(n_splits=2, test_size=200)
tscv = TimeSeriesSplit(n_splits=3, test_size=200)
all_splits = list(tscv.split(X, y))
R2_test_, R2_train_ = [],[]

for cv_folder in all_splits:
    train_0, test_0 =cv_folder
    X = combined_.drop(['HVAC power (kW)',], axis=1)
    y = combined_['HVAC power (kW)']
# train_0, test_0 = all_splits[0]
    X_train = X.iloc[train_0]
    X_test = X.iloc[test_0]
    y_train = y.iloc[train_0]
    y_test = y.iloc[test_0]
    y = np.array(y)

    # SCALE DATA
    scaler = StandardScaler() 
    X_train = scaler.fit_transform(X_train)
    # X_train =np.array(X_train)
    X_test = scaler.transform(X_test) 
    # X_test =np.array(X_test)
    y_train = np.array(y_train)
    y_test = np.array(y_test)
    trainX = np.reshape(X_train, (X_train.shape[0], 1, X_train.shape[1]))
    testX = np.reshape(X_test, (X_test.shape[0], 1, X_test.shape[1]))

    # create and fit the LSTM network
    model = Sequential()
    # model.add(LSTM(1))
    # model.add(Dense(10, use_bias=True, kernel_initializer=tf.keras.initializers.Constant(value=0.5)))
    model.add(LSTM(50, input_shape =(trainX.shape[1], trainX.shape[2])))
    model.add(Dense(40, use_bias=False, kernel_initializer=tf.keras.initializers.Constant(value=0.5)))
    model.add(Dense(5, use_bias=False))

    # model.add(Dense(4, use_bias=True))
    model.compile(loss='mean_squared_error', optimizer='adam')
    history = LossHistory()
    hist = model.fit(trainX,y_train, epochs=200, batch_size=100, verbose=2, callbacks=[history], validation_data=(testX, y_test))
    '''
    batch_size: It basically refers to the Number of Training examples utilized per Iteration. For example, 
    if you have 55K images and you specified the batch size as 32. Then in each epoch, there will be 1719 steps (55000/32~1719). After each step, the weight will be updated for the next step
    verbose: 
    '''
    # make predictions
    trainPredict = model.predict(trainX)
    train_values = []
    for element in trainPredict:
        train_values.append(element.tolist()[0])

    r2_score(train_values,y_train)

    testPredict = model.predict(testX)
    test_values = []
    for element in testPredict:
        test_values.append(element.tolist()[0])

    r2_score(test_values,y_test)

    # calculate root mean squared error
    trainScore = np.sqrt(mean_squared_error(y_train,train_values))
    print('Train Score: %.2f RMSE' % (trainScore))
    print('Train Score: %.2f R2' % (r2_score(train_values,y_train)))
    testScore = np.sqrt(mean_squared_error(y_test, test_values))
    print('Test Score: %.2f RMSE' % (testScore))
    print('Test Score: %.2f R2' % (r2_score(test_values,y_test)))

    import matplotlib.pyplot as plt
    plt.plot(train_values, color="red")
    plt.plot(y_train, color="blue")


    plt.plot(history.losses, label='train')
    #
    plt.plot(hist.history['loss'], label='train')
    plt.plot(hist.history['val_loss'], label='test')
    # Append result
    R2_test_.append(r2_score(train_values,y_train))
    R2_train_.append(r2_score(test_values,y_test))

# 








# ==============================================================================
#                           USING PYGWALER
# ==============================================================================

import pandas as pd
import streamlit.components.v1 as components
import streamlit as st
from pygwalker.api.streamlit import init_streamlit_comm, get_streamlit_html
 
st.set_page_config(
    page_title="Use Pygwalker with Streamlit",
    layout="wide"
)
 
st.title("PyGWalker with Streamlit")
 
# Initialize pygwalker communication
init_streamlit_comm()
 
# When using `use_kernel_calc=True`, you should cache your pygwalker html, if you don't want your memory to explode
@st.cache_resource
def get_pyg_html(df: pd.DataFrame) -> str:
    # When you need to publish your application, you need set `debug=False`,prevent other users to write your config file.
    # If you want to use feature of saving chart config, set `debug=True`
    html = get_streamlit_html(df, spec="./gw0.json", use_kernel_calc=True, debug=False)
    return html
 
@st.cache_data
def get_df() -> pd.DataFrame:
    file = "../BCFT.csv"
    return pd.read_csv(file,sep=";", decimal=",")
 
df = get_df()
 
components.html(get_pyg_html(df), width=1300, height=1000, scrolling=True)
