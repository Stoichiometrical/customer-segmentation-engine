# import json
# import os
# import pandas as pd
# import shortuuid
# from flask import Flask, request, jsonify, send_file
# from flask_cors import CORS
# from io import StringIO, BytesIO
# import boto3
#
# from customer_segmentation import (
#     customer_segmentation, prepare_data, fit_models,
#     predict_variables, calculate_descriptive_statistics,
#     create_segment_columns, create_marketing_data
# )
# from data_generation import generate_products_data, get_segment_transactions, generate_bundle_info
# from order_analysis_offers import apply_discount_tier, load_custom_discounts, calculate_loyalty_points, \
#     calculate_next_tier_difference
# from product_bundles import save_custom_discounts, get_bundlesets, create_bogd_offers
# from utility_methods import segment_offers
#
# app = Flask(__name__)
# CORS(app)
#
# # Initialize S3 client
# s3_client = boto3.client('s3')
# S3_BUCKET_NAME = 'iso-datalake'  # Replace with your actual S3 bucket name
#
#
# def upload_to_s3(file_content, bucket_name, key):
#     try:
#         s3_client.put_object(Bucket=bucket_name, Key=key, Body=file_content)
#         return True
#     except Exception as e:
#         print(f"Error uploading to S3: {e}")
#         return False
#
#
# def download_from_s3(bucket_name, key):
#     try:
#         obj = s3_client.get_object(Bucket=bucket_name, Key=key)
#         return obj['Body'].read().decode('utf-8')
#     except Exception as e:
#         print(f"Error downloading from S3: {e}")
#         return None
#
#
# @app.route('/initiate-segments', methods=['POST'])
# def initiate_segments():
#     try:
#         print("Starting segmentation process")
#
#         # Get the form data
#         sales_file_path = request.form['sales_file_path']
#         folder_name = request.form['folder_name']
#         customer_file_path = request.form['customer_file_path']
#         print(f"Received form data: sales_file_path={sales_file_path}, folder_name={folder_name}, customer_file_path={customer_file_path}")
#
#         # Download files from S3
#         sales_file_content = download_from_s3(S3_BUCKET_NAME, sales_file_path)
#         # customer_file_content = download_from_s3(S3_BUCKET_NAME, customer_file_path)
#         print("Downloaded files from S3")
#
#         if sales_file_content is None:
#             raise Exception("Failed to download files from S3.")
#
#         sales_df = pd.read_csv(StringIO(sales_file_content))
#         # customer_df = pd.read_csv(StringIO(customer_file_content))
#         sales_df['id'] = [shortuuid.uuid() for _ in range(len(sales_df))]
#         print("Sales and customer files read into DataFrames")
#
#         # Process the sales data
#         products_data_path = f"{folder_name}/products_data.csv"
#         generate_products_data(sales_df, output_file=products_data_path)
#         print("Sales data processed and products data generated")
#
#         # Upload processed data to S3
#         if not upload_to_s3(sales_df.to_csv(index=False), S3_BUCKET_NAME, products_data_path):
#             raise Exception("Failed to upload processed data to S3.")
#         print("Processed data uploaded to S3")
#
#         sales_df['Date'] = pd.to_datetime(sales_df['Date'])
#         sales_df = sales_df[sales_df["TotalPrice"] > 0]
#         print("Sales data cleaned and filtered")
#
#         summary = prepare_data(sales_df, customer_id_col="CustomerID", datetime_col='Date',
#                                monetary_value_col='TotalPrice', observation_period_end=max(sales_df["Date"]))
#         print("Data prepared for model fitting")
#
#         bgf, ggf = fit_models(summary)
#         print("Models fitted")
#
#         summary = predict_variables(summary, bgf, ggf, threshold=0.5)
#         summary = create_segment_columns(summary, recency='recency', frequency='frequency',
#                                          monetary_value='monetary_value', clv='predicted_clv')
#         print("Predicted variables and created segment columns")
#
#         segment_stats, subsegment_stats, percentages_dict, all_data_dict = calculate_descriptive_statistics(
#             summary, fields=['probability_alive', 'predicted_purchases', 'predicted_clv', 'estimated_monetary_value']
#         )
#         print("Descriptive statistics calculated")
#
#         segments = summary[["CustomerID", "Segment", "Subsegment"]]
#         merged = pd.merge(sales_df, segments, on='CustomerID', how='left')
#         print("Sales data merged with segments")
#
#         summary_csv = summary.to_csv(index=False)
#         merged_csv = merged.to_csv(index=False)
#         segment_stats_csv = segment_stats.to_csv(index=False)
#         subsegment_stats_csv = subsegment_stats.to_csv(index=False)
#         percentages_json = json.dumps(percentages_dict)
#         all_data_json = json.dumps(all_data_dict)
#
#         print("Converted data to CSV and JSON formats")
#
#         upload_to_s3(summary_csv, S3_BUCKET_NAME, f"{folder_name}/segment_data.csv")
#         upload_to_s3(merged_csv, S3_BUCKET_NAME, f"{folder_name}/segment_transactions.csv")
#         upload_to_s3(segment_stats_csv, S3_BUCKET_NAME, f"{folder_name}/segment_stats.csv")
#         upload_to_s3(subsegment_stats_csv, S3_BUCKET_NAME, f"{folder_name}/subsegment_stats.csv")
#         upload_to_s3(percentages_json, S3_BUCKET_NAME, f"{folder_name}/segment_percentages.json")
#         upload_to_s3(all_data_json, S3_BUCKET_NAME, f"{folder_name}/segment_compositions.json")
#         print("Uploaded all processed files to S3")
#
#         # Save segment transactions
#         get_segment_transactions(merged, f"{folder_name}/")
#         print("Saved segment transactions")
#
#         return jsonify({'message': 'Customer segments generated successfully.'})
#
#     except Exception as e:
#         print("Exception:", e)
#         return jsonify({'error': str(e)}), 500
#
#
#
# @app.route('/get_customer_segments', methods=['GET'])
# def get_customer_segments():
#     api_name = request.args.get('api_name')
#     customer_segments_path = f"{api_name}/customer_segments.csv"
#
#     customer_segments_content = download_from_s3(S3_BUCKET_NAME, customer_segments_path)
#     if customer_segments_content:
#         return send_file(BytesIO(customer_segments_content.encode()), mimetype='text/csv', as_attachment=True,
#                          download_name='customer_segments.csv')
#     else:
#         return jsonify({'error': 'Customer segments file not found'}), 404
#
#
# @app.route("/<api_name>/get-segment-details", methods=["POST"])
# def get_percentages(api_name):
#     project_name = request.json.get('project_name')
#     file_path = f"{project_name}/segment_percentages.json"
#
#     percentages_content = download_from_s3(S3_BUCKET_NAME, file_path)
#     if percentages_content:
#         percentages = json.loads(percentages_content)
#         return jsonify(percentages)
#     else:
#         return jsonify({"error": "File not found"}), 404
#
#
# @app.route('/generate_api_data/<api_name>/<project_name>/', methods=['POST'])
# def generate_api_data(api_name, project_name):
#     try:
#         data = request.json
#         file_name = "custom_discounts.json"
#         report_path = f"{project_name}/{api_name}/"
#
#         save_custom_discounts(data, file_name, report_path)
#
#         high_path = f"{project_name}/high_value_transactions.csv"
#         nurture_path = f"{project_name}/nurture_transactions.csv"
#         risk_path = f"{project_name}/risk_transactions.csv"
#
#         high_content = download_from_s3(S3_BUCKET_NAME, high_path)
#         nurture_content = download_from_s3(S3_BUCKET_NAME, nurture_path)
#         risk_content = download_from_s3(S3_BUCKET_NAME, risk_path)
#
#         if high_content is None or nurture_content is None or risk_content is None:
#             raise Exception("Failed to download high/nurture/risk transactions from S3.")
#
#         high = pd.read_csv(StringIO(high_content))
#         nurture = pd.read_csv(StringIO(nurture_content))
#         risk = pd.read_csv(StringIO(risk_content))
#
#         sales_data_path = f"{project_name}/segment_transactions.csv"
#         segment_data_path = f"{project_name}/segment_data.csv"
#         sales_data_content = download_from_s3(S3_BUCKET_NAME, sales_data_path)
#         segment_data_content = download_from_s3(S3_BUCKET_NAME, segment_data_path)
#
#         if sales_data_content is None or segment_data_content is None:
#             raise Exception("Failed to download sales/segment data from S3.")
#
#         sales_data = pd.read_csv(StringIO(sales_data_content))
#         segment_data = pd.read_csv(StringIO(segment_data_content))
#         min_discount = float(data['marginReduction']['min'])
#         max_discount = float(data['marginReduction']['max'])
#
#         create_marketing_data(segment_data, report_path)
#         products_data_path = f"{report_path}products_data.csv"
#         generate_products_data(df=sales_data, min_margin=min_discount, min_discount=max_discount,
#                                output_file=products_data_path)
#
#         get_bundlesets(high, risk, nurture, root_path=report_path)
#
#         segment_names = ['high_value', 'nurture', 'risk']
#         generate_bundle_info(report_path, segment_names)
#         create_bogd_offers(report_path)
#
#         return jsonify({"message": "Data has been saved successfully"}), 200
#     except Exception as e:
#         print(f"An error occurred: {e}")
#         return jsonify({"message": "An error occurred", "error": str(e)}), 500
#
#
# @app.route('/get_products', methods=['GET'])
# def get_products():
#     try:
#         api_name = request.args.get('api_name')
#         project_name = request.args.get('project_name')
#         file_path = f"{project_name}/{api_name}/products_data.csv"
#
#         products_data_content = download_from_s3(S3_BUCKET_NAME, file_path)
#         if products_data_content:
#             return send_file(BytesIO(products_data_content.encode()), mimetype='text/csv', as_attachment=True,
#                              download_name='products_data.csv')
#         else:
#             return jsonify({"message": "Product data not found for the specified API."}), 404
#     except Exception as e:
#         return jsonify({"message": "An error occurred", "error": str(e)}), 500
#
#
# @app.route('/order_analysis_offers', methods=['POST'])
# def order_analysis_offers():
#     try:
#         data = request.json
#         api_name = data['api_name']
#         project_name = data['project_name']
#         path = f"{project_name}/{api_name}/"
#         folder_name = request.form['folder_name']
#
#         load_custom_discounts(path)
#         file_path = f"{folder_name}/segment_transactions.csv"
#         segment_transactions_content = download_from_s3(S3_BUCKET_NAME, file_path)
#
#         if segment_transactions_content is None:
#             raise Exception("Failed to download segment transactions from S3.")
#
#         segment_transactions = pd.read_csv(StringIO(segment_transactions_content))
#
#         segment_offers(segment_transactions, path)
#         segment_path = f"{folder_name}/segments.csv"
#         if not upload_to_s3(segment_transactions.to_csv(index=False), S3_BUCKET_NAME, segment_path):
#             raise Exception("Failed to upload segment offers to S3.")
#
#         return jsonify({'message': 'Segment offers calculated successfully.'})
#     except Exception as e:
#         return jsonify({'message': 'An error occurred', 'error': str(e)}), 500
#
#
# if __name__ == '__main__':
#     app.run(debug=True, port=5000)

import json
import os

import pandas as pd
import shortuuid
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS

from customer_segmentation import customer_segmentation, prepare_data, fit_models, \
    predict_variables, calculate_descriptive_statistics, create_segment_columns, create_marketing_data
from data_generation import generate_products_data, get_segment_transactions, \
    generate_bundle_info
from order_analysis_offers import apply_discount_tier, load_custom_discounts, \
    calculate_loyalty_points, calculate_next_tier_difference
from product_bundles import save_custom_discounts, get_bundlesets, \
    create_bogd_offers
from utility_methods import segment_offers

ROOT_PATH = "../s3"

app = Flask(__name__)
# CORS(app, resources={r"/*": {"origins": "http://localhost:3000"}})
CORS(app)



@app.route('/initiate-segments', methods=['POST'])
def initiate_segments():
    try:
        # Get the form data
        sales_file_path = request.form['sales_file_path']
        folder_name = request.form['folder_name']
        customer_file_path = request.form['customer_file_path']
        root_path = f"{ROOT_PATH}/{folder_name}"

        df = pd.read_csv(sales_file_path)
        df['id'] = [shortuuid.uuid() for _ in range(len(df))]
        products_data_path = os.path.join(root_path, 'products_data.csv')
        print("Generating products dataset...")
        generate_products_data(df, output_file=products_data_path)

        df['Date'] = pd.to_datetime(df['Date'])
        df = df[df["TotalPrice"] > 0]
        summary = prepare_data(df, customer_id_col="CustomerID", datetime_col='Date',
                               monetary_value_col='TotalPrice', observation_period_end=max(df["Date"]))

        bgf, ggf = fit_models(summary)
        summary = predict_variables(summary, bgf, ggf, threshold=0.5)
        summary = create_segment_columns(summary, recency='recency', frequency='frequency',
                                         monetary_value='monetary_value',
                                         clv='predicted_clv')

        segment_stats, subsegment_stats, percentages_dict, all_data_dict = calculate_descriptive_statistics(
            summary, fields=['probability_alive', 'predicted_purchases', 'predicted_clv', 'estimated_monetary_value']
        )
        segments = summary[["CustomerID", "Segment", "Subsegment"]]

        merged = pd.merge(df, segments, on='CustomerID', how='left')

        if not os.path.exists(root_path):
            os.makedirs(root_path)

        summary.to_csv(os.path.join(root_path, 'segment_data.csv'), index=False)
        merged.to_csv(os.path.join(root_path, 'segment_transactions.csv'), index=False)
        segment_stats.to_csv(os.path.join(root_path, 'segment_stats.csv'), index=False)
        subsegment_stats.to_csv(os.path.join(root_path, 'subsegment_stats.csv'), index=False)

        with open(os.path.join(root_path, 'segment_percentages.json'), 'w') as f:
            json.dump(percentages_dict, f)
        with open(os.path.join(root_path, 'segment_compositions.json'), 'w') as f:
            json.dump(all_data_dict, f)

        # Save segment transactions
        get_segment_transactions(merged, root_path)

        return jsonify({'message': 'Customer segments generated successfully.'})

    except Exception as e:
        print(e)
        return jsonify({'error': str(e)}), 500

# Route to get customer segments file
@app.route('/get_customer_segments', methods=['GET'])
def get_customer_segments():
    api_name = request.args.get('api_name')
    customer_segments_path = os.path.join(ROOT_PATH, api_name, 'customer_segments.csv')
    print(customer_segments_path)

    if os.path.exists(customer_segments_path):
        return send_file(customer_segments_path, as_attachment=True)
    else:
        return jsonify({'error': 'Customer segments file not found'}), 404

# Get products data

@app.route("/<api_name>/get-segment-details", methods=["POST"])
def get_percentages(api_name):
    project_name = request.json.get('project_name')
    file_path = f"{ROOT_PATH}/{project_name}/segment_percentages.json"
    print(file_path)

    if not os.path.exists(file_path):
        return jsonify({"error": "File not found"}), 404

    with open(file_path, "r") as file:
        percentages = json.load(file)

    return jsonify(percentages)


@app.route('/generate_api_data/<api_name>/<project_name>/', methods=['POST'])
def generate_api_data(api_name, project_name):
    try:
        data = request.json

        file_name = "custom_discounts.json"
        root_path= f"{ROOT_PATH}/{project_name}"
        report_path = f"{ROOT_PATH}/{project_name}/{api_name}/"

        print("Starting.........")
        save_custom_discounts(data, file_name, report_path)
        print("Done saving customer discounts")

        high_path = os.path.join(root_path, 'high_value_transactions.csv')
        nurture_path = os.path.join(root_path, 'nurture_transactions.csv')
        risk_path = os.path.join(root_path, 'risk_transactions.csv')

        high = pd.read_csv(high_path)
        nurture = pd.read_csv(nurture_path)
        risk = pd.read_csv(risk_path)

        sales_data_path = os.path.join(root_path, 'segment_transactions.csv')
        segment_data_path = os.path.join(root_path, 'segment_data.csv')
        sales_data = pd.read_csv(sales_data_path)
        segment_data = pd.read_csv(segment_data_path)
        min_discount = float(data['marginReduction']['min'])
        max_discount = float(data['marginReduction']['max'])

        print("Creating marketing data.....")
        create_marketing_data(segment_data, report_path)

        print("Starting product generation...")
        products_data_path = os.path.join(report_path, "products_data.csv")
        generate_products_data(df=sales_data, min_margin=min_discount, min_discount=max_discount, output_file=products_data_path)

        print("Generating Product Bundles....")
        get_bundlesets(high, risk, nurture, root_path=report_path)

        segment_names = ['high_value', 'nurture', 'risk']
        generate_bundle_info(report_path, segment_names)

        create_bogd_offers(report_path)

        return jsonify({"message": "Data has been saved successfully"}), 200
    except Exception as e:
        print(f"An error occurred: {e}")
        return jsonify({"message": "An error occurred", "error": str(e)}), 500


@app.route('/get_products', methods=['GET'])
def get_products():
    try:
        # Get the API name from the request
        api_name = request.args.get('api_name')
        project_name = request.args.get('project_name')
        root_path = f"{ROOT_PATH}/{project_name}/{api_name}/"

        # Construct the file path for the product data CSV
        file_path = os.path.join(root_path, 'products_data.csv')
        print(file_path)

        # Check if the file exists
        if os.path.exists(file_path):
            # Send the CSV file as a response
            return send_file(file_path, mimetype='text/csv', as_attachment=True)
        else:
            return jsonify({"message": "Product data not found for the specified API."}), 404

    except Exception as e:
        return jsonify({"message": "An error occurred", "error": str(e)}), 500

# Get the bundles for specified segment

@app.route('/<api_name>/get_offers', methods=['GET'])
def get_promotions(api_name):
    try:
        date = request.args.get('date')
        segment_name = request.args.get('segment')
        project_name = request.args.get('project_name')
        project_path = os.path.join(ROOT_PATH, project_name)
        full_folder_path = os.path.join(project_path, api_name)

        # Paths for promotion info and offers
        promo_info_filename = "promo_days.csv"
        promo_path = os.path.join(full_folder_path, promo_info_filename)
        offers_path = os.path.join(full_folder_path, "custom_discounts.json")

        # Load promotions data
        promotions_df = pd.read_csv(promo_path)

        # Check if the date is present in the CSV
        if date in promotions_df['Date'].values:
            print("Date matched")
            if segment_name in segment_offers:
                segment_offers_list = segment_offers[segment_name]

                # Load offers from the JSON file
                with open(offers_path, 'r') as f:
                    offers_data = json.load(f)

                # Filter offers based on segment offers list
                valid_offers = {key: value for key, value in offers_data.items() if
                                key in segment_offers_list and value}

                return jsonify(valid_offers)

        # Return empty dictionary if no match
        return jsonify({})

    except FileNotFoundError as e:
        print(f"FileNotFoundError: {e}")
        return jsonify({"error": f"File not found: {e}"}), 500
    except KeyError as e:
        print(f"KeyError: {e}")
        return jsonify({"error": str(e)}), 500
    except Exception as e:
        print(f"An error occurred: {e}")
        return jsonify({"error": "An internal error occurred."}), 500

@app.route('/<api_name>/get_bundle_info', methods=['GET'])
def get_bundle_info(api_name):
    try:
        segment_name = request.args.get('segment_name')
        project_name = request.args.get('project_name')
        root_path = os.path.join(ROOT_PATH, project_name, api_name)

        if not segment_name:
            return jsonify({"error": "Segment name parameter is required."}), 400

        # Convert segment name to lower case and replace spaces with underscores
        formatted_segment_name = segment_name.lower().replace(' ', '_')

        # Check if the bundle info file for the segment exists
        bundle_info_filename = f"{formatted_segment_name}_bundles_info.csv"
        bundle_info_path = os.path.join(root_path, bundle_info_filename)

        if os.path.exists(bundle_info_path):
            # If the file exists, return it
            return send_file(bundle_info_path, as_attachment=True)
        else:
            return jsonify({"error": "Bundle info file not found for the specified segment."}), 404

    except Exception as e:
        print(f"An error occurred: {e}")
        return jsonify({"error": "An internal error occurred."}), 500

@app.route('/get_bodg_offers', methods=['GET'])
def get_bodg_offers():
    try:
        # Get the api_name and segment_name from the request arguments
        api_name = request.args.get('api_name')
        segment_name = request.args.get('segment_name')
        project_name = request.args.get('project_name')
        root_path = f"{ROOT_PATH}/{project_name}/{api_name}/"

        if not api_name or not segment_name:
            return jsonify({"message": "api_name and segment_name are required"}), 400

        # Construct the file path
        file_path = os.path.join(root_path, 'bogd_offers.csv')

        # Check if the file exists
        if not os.path.isfile(file_path):
            return jsonify({"message": "File not found"}), 404

        # Read the CSV file
        df = pd.read_csv(file_path)

        # Filter the rows by segment_name
        filtered_df = df[df['segment'] == segment_name]

        # Convert the filtered DataFrame to a dictionary
        filtered_data = filtered_df.to_dict(orient='records')

        # Return the filtered data as JSON
        return jsonify(filtered_data), 200
    except Exception as e:
        print(f"An error occurred: {e}")
        return jsonify({"message": "An error occurred", "error": str(e)}), 500


# Check order to apply discounts
@app.route('/apply_discount', methods=['POST'])
def apply_discount():
    try:
        data = request.json
        total_price = data['total_price']
        api_name = data['api_name']
        project_name = data['project_name']
        print(data)

        root_path = f"{ROOT_PATH}/{project_name}/{api_name}/"

        # Apply discount tier
        discounted_price = apply_discount_tier(total_price, root_path)

        # Load custom discounts
        discounts = load_custom_discounts(root_path)
        print("Start 1")

        # Extract loyalty points and high-value loyalty points
        loyalty_points = discounts.get('loyalty_points')
        high_value_loyalty_points = discounts.get('high_value_loyalty_points')

        # Calculate loyalty points if available
        loyalty_points_message = None
        if isinstance(loyalty_points, int):
            loyalty_points = calculate_loyalty_points(total_price, loyalty_points)
            loyalty_points_message = f"You have gained {loyalty_points} loyalty points."

        if isinstance(high_value_loyalty_points, int):
            high_value_loyalty_points = calculate_loyalty_points(total_price, high_value_loyalty_points)

        # Calculate amount needed for next tier and next tier description
        print("Start 2")
        amount_needed, next_tier_description = calculate_next_tier_difference(total_price, root_path)


        response = {
            "discounted_price": discounted_price,
            "loyalty_points": loyalty_points,
            "high_value_loyalty_points": high_value_loyalty_points,
            "message": f"Your discounted price is ${discounted_price:.2f}, spend ${amount_needed:.2f} to qualify for {next_tier_description}.",
        }

        if loyalty_points_message:
            response["loyalty_points_message"] = loyalty_points_message

        return jsonify(response), 200

    except Exception as e:
        print(f"An error occurred: {e}")
        return jsonify({"message": "An error occurred", "error": str(e)}), 500

# Function to load the CSV file and create the dataset
def load_and_process_data(api_name):
    file_path = f"{ROOT_PATH}/{api_name}/segment_data.csv"
    df = pd.read_csv(file_path)
    result_df = create_marketing_data(df,api_name)
    return result_df


# Endpoint to get all customers
@app.route('/get_customers', methods=['GET'])
def get_customers():
    api_name = request.args.get('api_name')
    project_name = request.args.get('project_name')
    if not api_name:
        return jsonify({"error": "api_name parameter is required"}), 400
    try:
        file_path = f"{ROOT_PATH}/{project_name}/{api_name}/marketing.csv"
        df = pd.read_csv(file_path)
        # customers_df = load_and_process_data(api_name)
        customers = df.to_dict(orient='records')
        return jsonify(customers), 200
    except Exception as e:
        print(e)
        return jsonify({"error": str(e)}), 500


@app.route('/<api_name>/get_emails_by_segment', methods=['GET'])
def get_emails_by_segment(api_name):
    project_name = request.args.get('project_name')
    segment = request.args.get('segment')

    if not api_name:
        return jsonify({"error": "api_name parameter is required"}), 400
    if not project_name:
        return jsonify({"error": "project_name parameter is required"}), 400
    if not segment:
        return jsonify({"error": "segment parameter is required"}), 400

    try:
        file_path = f"{ROOT_PATH}/{project_name}/{api_name}/marketing.csv"
        df = pd.read_csv(file_path)

        # Filter customers by segment
        filtered_customers = df[df['Segment'] == segment]

        # Extract emails
        emails = filtered_customers['Email'].tolist()

        return jsonify({"emails": emails}), 200

    except FileNotFoundError:
        return jsonify({"error": "File not found"}), 404
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/<api_name>/get_emails_by_subsegment', methods=['GET'])
def get_emails_by_subsegment(api_name):
    project_name = request.args.get('project_name')
    subsegment = request.args.get('subsegment')

    if not api_name:
        return jsonify({"error": "api_name parameter is required"}), 400
    if not project_name:
        return jsonify({"error": "project_name parameter is required"}), 400
    if not subsegment:
        return jsonify({"error": "subsegment parameter is required"}), 400

    try:
        file_path = f"{ROOT_PATH}/{project_name}/{api_name}/marketing.csv"
        df = pd.read_csv(file_path)

        # Filter customers by subsegment
        filtered_customers = df[df['Subsegment'] == subsegment]

        # Extract emails
        emails = filtered_customers['Email'].tolist()

        return jsonify({"emails": emails}), 200

    except FileNotFoundError:
        return jsonify({"error": "File not found"}), 404
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    app.run(port=5000)


