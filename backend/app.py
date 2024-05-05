# from flask import Flask, request, render_template, jsonify, send_file, send_from_directory
# import io
# from PIL import Image
# from flask_cors import CORS
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# from prophet import Prophet
# import os

# app = Flask(__name__)
# CORS(app)
# accuracy = 0.0

# def forecast(file_name, p, f):
#     global accuracy
    
#     data = pd.read_csv(file_name)

#     data['date'] = pd.to_datetime(data['date'])
#     data = data.rename(columns={'date': 'ds', 'sales': 'y'})

#     train_data = data[:-2]
#     test_data = data[-2:]

#     model = Prophet()
#     model.fit(train_data)

#     future = model.make_future_dataframe(periods=p, freq=f)
#     forecast = model.predict(future)
   
#     fig = model.plot(forecast, figsize=(9, 5))
#     plt.xlabel('Time period')
#     plt.ylabel('Sales')
#     plt.savefig('plot.png')

#     # forecast_points = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]
#     # forecast_points.to_csv('forecast.csv', index=False)
#     forecast.to_csv('forecast.csv', index=False)
#     # Calculate accuracy
#     y_true = test_data['y'].values
#     y_pred = forecast['yhat'][-2:].values
#     accuracy = 100 - np.mean(np.abs((y_true - y_pred) / y_true)) * 100
#     print(accuracy)

# @app.route('/', methods=['GET', 'POST'])
# def hello_world():
#     global accuracy 

#     if request.method == 'POST':
#         val1 = request.form['period']
#         val2 = request.form['range']
#         print(val1)
#         print(val2)
#         if val1 == "week":
#             fre = "W"
#         elif val1 == "month":
#             fre = "M"
#         elif val1 == "year":
#             fre = "Y"
#         elif val1 == "day":
#             fre = "D"

#         per = int(val2)

#         file = request.files['file']
#         file_name = file.filename
#         file.save(file_name)
#         file_size = len(file.read())
#         file_stats = os.stat(file_name)

#         forecast(file_name, per, fre)

#         response_headers = {'Access-Control-Allow-Origin': '*'}
#         response = {'message': 'Success', 'accuracy': accuracy * 100}
#         return jsonify(response), 200, response_headers

#     if request.method == 'GET':
#         response = {'accuracy': accuracy * 100}
#         return send_file('plot.png', mimetype='image/png'),response

# @app.route('/accuracy', methods=['GET'])
# def get_accuracy():
#     global accuracy
#     response = {'accuracy': accuracy}
#     return jsonify(response)

# if __name__ == '_main_':
#     app.run(debug=True)


# from flask import Flask, request, render_template, jsonify, send_file, send_from_directory
# import io
# from PIL import Image
# from flask_cors import CORS
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# from prophet import Prophet
# import os
# from sklearn.cluster import KMeans
# from sklearn.preprocessing import StandardScaler

# app = Flask(__name__)
# CORS(app)
# accuracy = 0.0
# result= ""

# #new
# def analyze_csv(file_path, num_rows):
#     # Read the CSV file into a DataFrame
#     df = pd.read_csv(file_path)

#     # Convert the 'ds' column to datetime if it's not already in datetime format
#     df['ds'] = pd.to_datetime(df['ds'])

#     # Sort the DataFrame by the 'ds' column in ascending order
#     df.sort_values(by='ds', inplace=True)

#     # Get the last num_rows from the DataFrame
#     last_n_rows = df.tail(num_rows)

#     # Initialize counters for increasing and decreasing trends
#     increasing_count = 0
#     decreasing_count = 0

#     # Loop through each column (excluding 'ds' column)
#     for column in df.columns[1:]:
#         # Get the difference between consecutive rows for each column
#         diffs = last_n_rows[column].diff()

#         # Check if the difference is increasing or decreasing
#         increasing = all(diff > 0 for diff in diffs.dropna())
#         decreasing = all(diff < 0 for diff in diffs.dropna())

#         # Update the counters based on the trend of the current column
#         if increasing:
#             increasing_count += 1
#         elif decreasing:
#             decreasing_count += 1

#     # Determine the overall trend statement
#     if increasing_count > decreasing_count:
#         overall_trend_statement = "Overall trend is increasing."
#     elif increasing_count < decreasing_count:
#         overall_trend_statement = "Overall trend is decreasing."
#     else:
#         overall_trend_statement = "Overall trend is stable."

#     return overall_trend_statement

# # response_headers = {'Access-Control-Allow-Origin': '*'}
# #         response = {'message': 'Success', 'accuracy': accuracy * 100}
# #         return jsonify(response), 200, response_headers
# #new

# def forecast(file_name, p, f):
#     global accuracy
    
#     data = pd.read_csv(file_name) 

#     data['date'] = pd.to_datetime(data['date'])
#     data = data.rename(columns={'date': 'ds', 'sales': 'y'})

#     train_data = data[:-2]
#     test_data = data[-2:]

#     model = Prophet()
#     model.fit(train_data)

#     future = model.make_future_dataframe(periods=p, freq=f)
#     forecast = model.predict(future)
   
#     fig = model.plot(forecast, figsize=(9, 5))
#     plt.xlabel('Time period')
#     plt.ylabel('Sales')
#     plt.savefig('plot.png')

#     forecast.to_csv('forecast.csv', index=False)
    
#     # Calculate accuracy
#     y_true = test_data['y'].values
#     y_pred = forecast['yhat'][-2:].values
#     accuracy = 100 - np.mean(np.abs((y_true - y_pred) / y_true)) * 100
#     print(accuracy)
#     #new
    
#     #new

# @app.route('/', methods=['GET', 'POST'])
# def hello_world():
#     global accuracy 

#     if request.method == 'POST':
#         val1 = request.form['period']
#         val2 = request.form['range']
#         print(val1)
#         print(val2)
#         if val1 == "week":
#             fre = "W"
#         elif val1 == "month":
#             fre = "M"
#         elif val1 == "year":
#             fre = "Y"
#         elif val1 == "day":
#             fre = "D"

#         per = int(val2)

#         file = request.files['file']
#         file_name = file.filename
#         file.save(file_name)
#         file_size = len(file.read())
#         file_stats = os.stat(file_name)

#         forecast(file_name, per, fre)

#         #new
#         result = analyze_csv("forecast.csv", per)
#         print(result) 
#         #new

#         response_headers = {'Access-Control-Allow-Origin': '*'}
#         response = {'message': 'Success', 'accuracy': accuracy * 100}
#         return jsonify(response), 200, response_headers

#     if request.method == 'GET':
#         response = {'accuracy': accuracy * 100}
#         return send_file('plot.png', mimetype='image/png'),response






# @app.route('/accuracy', methods=['GET'])
# def get_accuracy():
#     global accuracy
#     response = {'accuracy': accuracy}
#     return jsonify(response)

# def clusters(filename):
#     df = pd.read_csv(filename)

# # Extract relevant features for clustering (sales)
#     features = df[['sales']].values

# # Standardize the features
#     scaler = StandardScaler()
#     features_standardized = scaler.fit_transform(features)

# # K-means clustering
#     num_clusters = 3  # You can adjust this based on your requirement
#     kmeans_model = KMeans(n_clusters=num_clusters, random_state=42)
#     df['cluster'] = kmeans_model.fit_predict(features_standardized)

# # Identify clusters with high sales values
#     high_sales_clusters = df.groupby('cluster')['sales'].mean().sort_values(ascending=False).index[:1]

# # Filter the dataframe to include only rows from high sales clusters
#     df_high_sales = df[df['cluster'].isin(high_sales_clusters)].copy()  # Create a copy to avoid SettingWithCopyWarning

# # Calculate discount for each row based on sales value (rounded to 2 decimal places)
#     df_high_sales['discount'] = (df_high_sales['sales'] * 0.1).round(2)

# # Save clustered data with discount to a new CSV file
#     df_high_sales.to_csv('high_sales_clusters_with_discount.csv', index=False)

# # Print the head of the new CSV file to verify the 'discount' column is present
#     verification_df = pd.read_csv('high_sales_clusters_with_discount.csv')
#     print(verification_df.head())

# @app.route('/high_sales_clusters_csv', methods=['GET'])
# def get_high_sales_clusters_csv():
#     return send_file('high_sales_clusters_with_discount.csv', as_attachment=True)

# @app.route('/cluster', methods=['GET', 'POST'])
# def get_cluster_results():
#     # Read clustered data with high sales values and discount
#     if request.method == 'POST':
#         file = request.files['file']
#         file_name = file.filename
#         file.save(file_name)
#         file_size = len(file.read())
#         file_stats = os.stat(file_name)
#         clusters(file_name)
        
#     # if request.method == 'POST':
#     #     # Handle the POST request, perform clustering, calculate discount, and return the necessary response
#     #     # ...
#     #     return jsonify({'message': 'Clustering and discount calculation completed successfully'})

#     # For GET requests, return the high sales clusters with discount
#     clustered_data_high_sales = pd.read_csv('high_sales_clusters_with_discount.csv')

#     # Convert to JSON format
#     clustered_json = clustered_data_high_sales.to_json(orient='records')

#     return clustered_json

# if __name__ == '__main__':
#     app.run(debug=True)



from flask import Flask, request, render_template, jsonify, send_file, send_from_directory
import io
from PIL import Image
from flask_cors import CORS
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from prophet import Prophet
import os
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)
CORS(app)
accuracy = 0.0
result= ""

#new
def analyze_csv(file_path, num_rows):
    try:
        # Read the CSV file into a DataFrame
        df = pd.read_csv(file_path)

        # Convert the 'ds' column to datetime if it's not already in datetime format
        df['ds'] = pd.to_datetime(df['ds'])

        # Sort the DataFrame by the 'ds' column in ascending order
        df.sort_values(by='ds', inplace=True)

        # Get the last num_rows from the DataFrame
        last_n_rows = df.tail(num_rows)

        # Initialize counters for increasing and decreasing trends
        increasing_count = 0
        decreasing_count = 0

        # Loop through each column (excluding 'ds' column)
        for column in df.columns[1:]:
            # Get the difference between consecutive rows for each column
            diffs = last_n_rows[column].diff()

            # Check if the difference is increasing or decreasing
            increasing = all(diff > 0 for diff in diffs.dropna())
            decreasing = all(diff < 0 for diff in diffs.dropna())

            # Update the counters based on the trend of the current column
            if increasing:
                increasing_count += 1
            elif decreasing:
                decreasing_count += 1

        # Determine the overall trend statement
        if increasing_count > decreasing_count:
            overall_trend_statement = "Overall trend is increasing."
        elif increasing_count < decreasing_count:
            overall_trend_statement = "Overall trend is decreasing."
        else:
            overall_trend_statement = "Overall trend is stable."

        return overall_trend_statement

    except Exception as e:
        return str(e)

# Route to handle GET requests for overall trend
@app.route('/trend', methods=['GET'])
def get_overall_trend():
    try:
        # Get the file path and number of rows from the request
        file_path = "forecast.csv"  # Assuming forecast.csv is already generated
        num_rows = 5  # Assuming a default value for the number of rows

        # Call the analyze_csv function with the provided parameters
        overall_trend_message = analyze_csv(file_path, num_rows)

        # Return the overall trend message as JSON in the response
        return jsonify({'message': overall_trend_message})

    except Exception as e:
        # Catch any exceptions and return an error message
        return jsonify({'error': str(e)}), 500

# response_headers = {'Access-Control-Allow-Origin': '*'}
#         response = {'message': 'Success', 'accuracy': accuracy * 100}
#         return jsonify(response), 200, response_headers
#new

def forecast(file_name, p, f):
    global accuracy
    
    data = pd.read_csv(file_name) 

    data['date'] = pd.to_datetime(data['date'])
    data = data.rename(columns={'date': 'ds', 'sales': 'y'})

    train_data = data[:-2]
    test_data = data[-2:]

    model = Prophet()
    model.fit(train_data)

    future = model.make_future_dataframe(periods=p, freq=f)
    forecast = model.predict(future)
   
    fig = model.plot(forecast, figsize=(9, 5))
    plt.xlabel('Time period')
    plt.ylabel('Sales')
    plt.savefig('plot.png')

    forecast.to_csv('forecast.csv', index=False)
    
    # Calculate accuracy
    y_true = test_data['y'].values
    y_pred = forecast['yhat'][-2:].values
    accuracy = 100 - np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    print(accuracy)
    #new
    
    #new

@app.route('/', methods=['GET', 'POST'])
def hello_world():
    global accuracy 

    if request.method == 'POST':
        val1 = request.form['period']
        val2 = request.form['range']
        print(val1)
        print(val2)
        if val1 == "week":
            fre = "W"
        elif val1 == "month":
            fre = "M"
        elif val1 == "year":
            fre = "Y"
        elif val1 == "day":
            fre = "D"

        per = int(val2)

        file = request.files['file']
        file_name = file.filename
        file.save(file_name)
        file_size = len(file.read())
        file_stats = os.stat(file_name)

        forecast(file_name, per, fre)

        #new
        result = analyze_csv("forecast.csv", per)
        print(result) 
        #new

        response_headers = {'Access-Control-Allow-Origin': '*'}
        response = {'message': 'Success', 'accuracy': accuracy * 100}
        return jsonify(response), 200, response_headers

    if request.method == 'GET':
        response = {'accuracy': accuracy * 100}
        return send_file('plot.png', mimetype='image/png'),response






@app.route('/accuracy', methods=['GET'])
def get_accuracy():
    global accuracy
    response = {'accuracy': accuracy}
    return jsonify(response)

def clusters(filename):
    df = pd.read_csv(filename)

# Extract relevant features for clustering (sales)
    features = df[['sales']].values

# Standardize the features
    scaler = StandardScaler()
    features_standardized = scaler.fit_transform(features)

# K-means clustering
    num_clusters = 3  # You can adjust this based on your requirement
    kmeans_model = KMeans(n_clusters=num_clusters, random_state=42)
    df['cluster'] = kmeans_model.fit_predict(features_standardized)

# Identify clusters with high sales values
    high_sales_clusters = df.groupby('cluster')['sales'].mean().sort_values(ascending=False).index[:1]

# Filter the dataframe to include only rows from high sales clusters
    df_high_sales = df[df['cluster'].isin(high_sales_clusters)].copy()  # Create a copy to avoid SettingWithCopyWarning

# Calculate discount for each row based on sales value (rounded to 2 decimal places)
    df_high_sales['discount'] = (df_high_sales['sales'] * 0.1).round(2)

# Save clustered data with discount to a new CSV file
    df_high_sales.to_csv('high_sales_clusters_with_discount.csv', index=False)

# Print the head of the new CSV file to verify the 'discount' column is present
    verification_df = pd.read_csv('high_sales_clusters_with_discount.csv')
    print(verification_df.head())

@app.route('/high_sales_clusters_csv', methods=['GET'])
def get_high_sales_clusters_csv():
    return send_file('high_sales_clusters_with_discount.csv', as_attachment=True)

@app.route('/cluster', methods=['GET', 'POST'])
def get_cluster_results():
    # Read clustered data with high sales values and discount
    if request.method == 'POST':
        file = request.files['file']
        file_name = file.filename
        file.save(file_name)
        file_size = len(file.read())
        file_stats = os.stat(file_name)
        clusters(file_name)
        
    # if request.method == 'POST':
    #     # Handle the POST request, perform clustering, calculate discount, and return the necessary response
    #     # ...
    #     return jsonify({'message': 'Clustering and discount calculation completed successfully'})

    # For GET requests, return the high sales clusters with discount
    clustered_data_high_sales = pd.read_csv('high_sales_clusters_with_discount.csv')

    # Convert to JSON format
    clustered_json = clustered_data_high_sales.to_json(orient='records')

    return clustered_json

if __name__ == '__main__':
    app.run(debug=True)

