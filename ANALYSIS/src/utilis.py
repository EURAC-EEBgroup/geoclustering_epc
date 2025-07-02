from sklearn.cluster import DBSCAN
from scipy import stats
import numpy as np
import pandas as pd
import numpy as np
from pyecharts import options as opts
from pyecharts.charts import Scatter, Line
from scipy.stats import linregress

def detect_outliers(df:pd.DataFrame, colName:str, method:str='DBSCAN'):
    '''
    Detect Outliers, using the follwoing method:
    - 1. DBSCAN
    - 2. Z_SCORE
    Param
    ------
    df: dataframe of values
    colName: name of colum to perform the test
    method: could be 'DBSCAN' or 'Z_SCORE', default DBSCAN
    '''
    if method  == "DBSCAN":
        dbscan = DBSCAN(eps=0.5, min_samples=20)  # Adjust parameters as needed
        dbscan_labels = dbscan.fit_predict(df[[colName]])
        outliers = df[dbscan_labels == -1]
    elif method == "Z_SCORE":
        z_scores = np.abs(stats.zscore(df[colName]))
        outliers = df[(z_scores > 3)]
    return outliers
  
    

def scatter_with_regression(x_data:list, y_data:list, colorPoints:list, colorRegression:list, 
                            seriesName:list,name_y_axes:str="", graphName:str="graph.html"):
    '''
    scatter plot with regression
    Param
    ------
    x_data: list of data
    y_data: list of list. IE [[1,2,3],[4,5,6]]
    colorPoints: list  in which length is equal to number of lists in y_data. I.E. ['red','yellow','green']
    colorRegression: list of color for teh regression line
    seriesName: list of name for each regression
    graphName: Name of the graph
    '''
    # Create a line chart for the regression line
    line_chart = (
        Line()
        .add_xaxis(x_data)
        .set_series_opts(label_opts=opts.LabelOpts(is_show=False))
    )
    scatter_chart = (
        Scatter()
        .add_xaxis(xaxis_data=x_data)
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
            xaxis_opts=opts.AxisOpts(
                type_="value", splitline_opts=opts.SplitLineOpts(is_show=True)
            ),
            yaxis_opts=opts.AxisOpts(
                type_="value",
                axistick_opts=opts.AxisTickOpts(is_show=True),
                splitline_opts=opts.SplitLineOpts(is_show=True),
                name=name_y_axes
            )
        )
    )

    r_squared = []
    n=0
    for data in y_data:
        # calcuation of regression line
        # Perform linear regression using numpy
        slope, intercept, r_value, p_value, std_err = linregress(x_data, data)

        # Calculate R-squared
        r_squared.append(r_value**2)

        # Generate predicted y values based on the regression line
        regression_line = [slope * x + intercept for x in x_data]
        regression_line = [round(element, 2) for element in regression_line]

        line_chart.add_yaxis(
            series_name=seriesName[n], 
            y_axis=regression_line, 
            is_smooth=True,
            symbol="emptyCircle",
            is_symbol_show=False,
            label_opts=opts.LabelOpts(is_show=True),
            color=colorRegression[n])

        scatter_chart.add_yaxis(
            series_name=seriesName[n],
            y_axis=data,
            symbol_size=10,
            label_opts=opts.LabelOpts(is_show=False),
            color=colorPoints[n]
        )
        n=n+1

    chart = scatter_chart.overlap(line_chart)
    # Set dimension of graph
    chart.height = "500px"
    chart.width = "800px"
    
    return chart.render(graphName)
    # return chart
