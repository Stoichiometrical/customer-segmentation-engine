# Segmenting customers
# import os
#
# import numpy as np
# import pandas as pd
# from faker import Faker
# from lifetimes.utils import summary_data_from_transaction_data
# from lifetimes import BetaGeoFitter, GammaGammaFitter
#
#
#
#
# def customer_segmentation(data):
#     df = data
#     df = df[["InvoiceDate", "InvoiceNo", "CustomerID", "StockCode", "UnitPrice", "Quantity", "TotalPrice"]]
#
#     # Group transactions by CustomerID then aggregate total price and quantity for them
#     customer = df.groupby("CustomerID").agg(
#         {
#             "TotalPrice": "sum",
#             "Quantity": "sum"
#         }
#     ).reset_index()
#
#     # Add frequency
#     customer["Frequency"] = df.groupby('CustomerID')['InvoiceNo'].count().reset_index()["InvoiceNo"]
#
#     # Add Monetary
#     customer['Monetary'] = df.groupby('CustomerID')['TotalPrice'].mean().reset_index()['TotalPrice']
#
#     # Add Recency
#     # First get date
#     df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])
#     rfm_data = df.groupby('CustomerID')['InvoiceDate'].max().reset_index()
#     customer['Recency'] = (rfm_data['InvoiceDate'].max() - rfm_data['InvoiceDate']).dt.days
#
#     # Now add RFM scores
#     customer['R_score'] = pd.qcut(customer['Recency'], q=3, labels=[1, 2, 3])  # High recency will have a score of 1
#     customer['F_score'] = pd.qcut(customer['Frequency'], q=3, labels=[1, 2, 3])
#     customer['M_score'] = pd.qcut(customer['Monetary'], q=3, labels=[1, 2, 3])
#
#     # Assign Final RFM Score
#     customer['RFM'] = customer[['R_score', 'F_score', 'M_score']].astype(str).agg(''.join, axis=1)
#     customer.drop(columns=['R_score', 'F_score', 'M_score'], inplace=True)
#
#     # Segment the customers
#     customer["Segment"], customer["Subsegment"] = segment_customers(customer['RFM'])
#
#     # Merging segments into final df
#     segmented = df[["CustomerID", "InvoiceNo", "StockCode"]]
#     x = customer[["CustomerID", "Segment", "Subsegment"]]
#     transaction_df = pd.merge(segmented, x, on="CustomerID", how="left")
#
#     return customer, transaction_df
#
#
# def segment_customers(rfm_column):
#     """Segments customers into broad and subsegments based on RFM scores for a general retail store.
#
#     Args:
#         rfm_column (pd.Series): Series containing RFM scores.
#
#     Returns:
#         tuple: A tuple containing two pandas Series, one for broad segment and one for subsegment.
#     """
#     broad_segments = []
#     subsegments = []
#
#     # Define dictionaries for each segment and subsegment
#     high_value_segments = {
#         (1, 1, 1): 'Loyal Champions', (1, 1, 2): 'Frequent Spenders', (1, 1, 3): 'Rising Stars',
#         (1, 2, 1): 'Recent Big Spenders', (1, 2, 2): 'Frequent Spenders', (1, 2, 3): 'Rising Stars',
#         (1, 3, 1): 'Rekindled Spenders', (1, 3, 2): 'Needs Attention', (1, 3, 3): 'Value Seekers',
#         (2, 3, 1): 'Big Ticket Buyers'
#     }
#     nurture_segments = {
#         (2, 2, 2): 'Occasional Spenders', (2, 2, 3): 'Value Seekers', (2, 3, 2): 'Sleeping Giants',
#         (2, 3, 3): 'Value Seekers', (1, 3, 3): 'Needs Attention',
#         (2, 1, 2): 'Win-Back Target', (2, 1, 3): 'Win-Back Target',
#         (2, 2, 1): 'Potential Upscale'
#     }
#     risk_segments = {
#         (3, 1, 1): 'Lost Loyalists', (3, 1, 2): 'Fading Interest', (3, 1, 3): 'One-Time Buyers',
#         (3, 2, 1): 'At-Risk Customers', (3, 2, 2): 'Fading Interest', (3, 2, 3): 'One-Time Buyers',
#         (3, 3, 1): 'Window Shoppers', (3, 3, 2): 'Window Shoppers', (3, 3, 3): 'One-Time Buyers',
#         (2, 1, 1): 'At-Risk Customers'
#     }
#
#     all_segments = list(high_value_segments.keys()) + list(nurture_segments.keys()) + list(risk_segments.keys())
#     all_subsegments = list(high_value_segments.values()) + list(nurture_segments.values()) + list(
#         risk_segments.values())
#
#     # Check if the lengths of segment and subsegment lists match
#     assert len(all_segments) == len(all_subsegments), "Lengths of segment and subsegment lists must match"
#
#     for rfm in rfm_column:
#         recency = int(rfm[0])
#         frequency = int(rfm[1])
#         monetary = int(rfm[2])
#
#         if (recency, frequency, monetary) in all_segments:
#             broad_segments.append(
#                 'High Value' if (recency, frequency, monetary) in high_value_segments.keys()
#                 else 'Nurture' if (recency, frequency, monetary) in nurture_segments.keys()
#                 else 'Risk'
#             )
#             subsegments.append(all_subsegments[all_segments.index((recency, frequency, monetary))])
#         else:
#             broad_segments.append('Unknown')
#             subsegments.append('Unknown')
#
#     return pd.Series(broad_segments, name='Broad Segment'), pd.Series(subsegments, name='Subsegment')
#
#
# def prepare_data(df, customer_id_col, datetime_col, monetary_value_col, observation_period_end):
#     summary = summary_data_from_transaction_data(
#         df,
#         customer_id_col=customer_id_col,
#         datetime_col=datetime_col,
#         monetary_value_col=monetary_value_col,
#         observation_period_end=observation_period_end
#     )
#     summary = summary[summary["monetary_value"] > 0]
#     print("Done fitting..")
#     return summary
#
#
# def fit_models(summary):
#     print("Fitting...")
#     bgf = BetaGeoFitter(penalizer_coef=0.5)
#     bgf.fit(summary['frequency'], summary['recency'], summary['T'])
#     print("Gamma Fittiing...")
#
#     ggf = GammaGammaFitter(penalizer_coef=0.0)
#     ggf.fit(summary['frequency'], summary['monetary_value'])
#
#     return bgf, ggf
#
#
# def predict_variables(summary, bgf, ggf, threshold):
#     summary['probability_alive'] = bgf.conditional_probability_alive(
#         summary['frequency'],
#         summary['recency'],
#         summary['T']
#     )
#     summary['predicted_purchases'] = bgf.predict(30, summary['frequency'], summary['recency'], summary['T'])
#     summary['predicted_clv'] = ggf.customer_lifetime_value(
#         bgf,
#         summary['frequency'],
#         summary['recency'],
#         summary['T'],
#         summary['monetary_value'],
#         time=1,  # Lifetime expected for the user in months
#         freq='D',
#         discount_rate=0.01
#     )
#     summary["estimated_monetary_value"] = ggf.conditional_expected_average_profit(
#         summary['frequency'],
#         summary['monetary_value']
#     )
#     return summary
#
#
# def create_segment_columns(df, recency, frequency, monetary_value, clv):
#     df['R_score'] = pd.qcut(df[recency], q=3, labels=[1, 2, 3])
#     df['F_score'] = pd.qcut(df[frequency], q=3, labels=[1, 2, 3])
#     df['M_score'] = pd.qcut(df[monetary_value], q=3, labels=[1, 2, 3])
#     df['CLV_segment'] = pd.qcut(df[clv], q=3, labels=['Low CLV', 'Medium CLV', 'High CLV'])
#
#     rfm_mapping = {'111': 'High Value', '112': 'High Value', '113': 'Risk', '121': 'High Value', '122': 'Nurture',
#                    '123': 'Risk', '131': 'High Value', '132': 'Nurture', '133': 'Risk', '211': 'Nurture',
#                    '212': 'Nurture',
#                    '213': 'Risk', '221': 'Nurture', '222': 'Nurture', '223': 'Risk', '231': 'Risk', '232': 'Risk',
#                    '233': 'Risk',
#                    '311': 'Risk', '312': 'Risk', '313': 'Risk', '321': 'Risk', '322': 'Risk', '323': 'Risk',
#                    '331': 'Risk',
#                    '332': 'Risk', '333': 'Risk'}
#
#     df['RFM'] = df['R_score'].astype(str) + df['F_score'].astype(str) + df['M_score'].astype(str)
#     df['Segment'] = df['RFM'].map(rfm_mapping)
#     df['Subsegment'] = df['Segment'].astype(str) + ', ' + df['CLV_segment'].astype(str)
#     # df["CustomerID"] = df.index
#     df = df.reset_index()
#     df.drop(columns=['R_score', 'F_score', 'M_score'], inplace=True)
#
#     return df
#
#
# def calculate_segment_percentages(df):
#     total_customers = len(df)
#     segment_counts = df['Segment'].value_counts()
#     segment_percentages = (segment_counts / total_customers * 100).round(2)
#     subsegment_counts = df['Subsegment'].value_counts()
#     subsegment_percentages = (subsegment_counts / total_customers * 100).round(2)
#     percentages_dict = {
#         'Segment': segment_percentages.to_dict(),
#         'Subsegment': subsegment_percentages.to_dict()
#     }
#     return percentages_dict
#
#
# def calculate_descriptive_statistics(df, fields):
#     segment_stats = df.groupby('Segment')[fields].agg(['mean', 'std', 'median']).reset_index()
#     subsegment_stats = df.groupby('Subsegment')[fields].agg(['mean', 'std', 'median']).reset_index()
#     segment_stats.columns = ['_'.join(col).strip() if isinstance(col, tuple) else col for col in
#                              segment_stats.columns.values]
#     subsegment_stats.columns = ['_'.join(col).strip() if isinstance(col, tuple) else col for col in
#                                 subsegment_stats.columns.values]
#     percentages_dict = calculate_segment_percentages(df)
#     all_data_dict = {
#         'segment_stats': segment_stats.to_dict(orient='records'),
#         'subsegment_stats': subsegment_stats.to_dict(orient='records'),
#         'percentages': percentages_dict
#     }
#     return segment_stats, subsegment_stats, percentages_dict, all_data_dict
#
#
# def create_marketing_data(df, path):
#     # Create the 'Status' column based on 'probability_alive'
#     df['Status'] = df['probability_alive'].apply(
#         lambda x: 'inactive' if x < 0.4 else 'regular' if x < 0.6 else 'active')
#
#     # Create a Faker instance
#     fake = Faker()
#
#     # List of emails to include
#     specific_emails = ['davidtgondo@gmail.com', 'd.gondo@alustudent.com']
#
#     # Function to generate emails
#     def generate_email(index):
#         if index < len(specific_emails):
#             return specific_emails[index]
#         return fake.email()
#
#     # Generate emails and ensure specific emails are included
#     df['Email'] = [generate_email(i) for i in range(len(df))]
#
#     # Set segment to 'Test' for specific emails
#     df.loc[df['Email'].isin(specific_emails), 'Segment'] = 'Test'
#
#     # Select the required columns
#     result_df = df[['CustomerID', 'Email', 'Segment', 'Subsegment', 'Status']]
#
#     result_path = os.path.join(path, "marketing.csv")
#     result_df.to_csv(result_path, index=False)
#
#     return result_df


import os
import json
import numpy as np
import pandas as pd
from faker import Faker
from lifetimes.utils import summary_data_from_transaction_data
from lifetimes import BetaGeoFitter, GammaGammaFitter
from fpgrowth_py import fpgrowth
import shortuuid
from io import StringIO
from utility_methods import aggregate_transactions, multipart_download_from_s3, multipart_upload_to_s3, S3_BUCKET_NAME


def customer_segmentation(data):
    df = data
    df = df[["InvoiceDate", "InvoiceNo", "CustomerID", "StockCode", "UnitPrice", "Quantity", "TotalPrice"]]

    customer = df.groupby("CustomerID").agg({
        "TotalPrice": "sum",
        "Quantity": "sum"
    }).reset_index()

    customer["Frequency"] = df.groupby('CustomerID')['InvoiceNo'].count().reset_index()["InvoiceNo"]
    customer['Monetary'] = df.groupby('CustomerID')['TotalPrice'].mean().reset_index()['TotalPrice']

    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])
    rfm_data = df.groupby('CustomerID')['InvoiceDate'].max().reset_index()
    customer['Recency'] = (rfm_data['InvoiceDate'].max() - rfm_data['InvoiceDate']).dt.days

    customer['R_score'] = pd.qcut(customer['Recency'], q=3, labels=[1, 2, 3])
    customer['F_score'] = pd.qcut(customer['Frequency'], q=3, labels=[1, 2, 3])
    customer['M_score'] = pd.qcut(customer['Monetary'], q=3, labels=[1, 2, 3])

    customer['RFM'] = customer[['R_score', 'F_score', 'M_score']].astype(str).agg(''.join, axis=1)
    customer.drop(columns=['R_score', 'F_score', 'M_score'], inplace=True)

    customer["Segment"], customer["Subsegment"] = segment_customers(customer['RFM'])

    segmented = df[["CustomerID", "InvoiceNo", "StockCode"]]
    x = customer[["CustomerID", "Segment", "Subsegment"]]
    transaction_df = pd.merge(segmented, x, on="CustomerID", how="left")

    return customer, transaction_df


def segment_customers(rfm_column):
    broad_segments = []
    subsegments = []

    high_value_segments = {
        (1, 1, 1): 'Loyal Champions', (1, 1, 2): 'Frequent Spenders', (1, 1, 3): 'Rising Stars',
        (1, 2, 1): 'Recent Big Spenders', (1, 2, 2): 'Frequent Spenders', (1, 2, 3): 'Rising Stars',
        (1, 3, 1): 'Rekindled Spenders', (1, 3, 2): 'Needs Attention', (1, 3, 3): 'Value Seekers',
        (2, 3, 1): 'Big Ticket Buyers'
    }
    nurture_segments = {
        (2, 2, 2): 'Occasional Spenders', (2, 2, 3): 'Value Seekers', (2, 3, 2): 'Sleeping Giants',
        (2, 3, 3): 'Value Seekers', (1, 3, 3): 'Needs Attention',
        (2, 1, 2): 'Win-Back Target', (2, 1, 3): 'Win-Back Target',
        (2, 2, 1): 'Potential Upscale'
    }
    risk_segments = {
        (3, 1, 1): 'Lost Loyalists', (3, 1, 2): 'Fading Interest', (3, 1, 3): 'One-Time Buyers',
        (3, 2, 1): 'At-Risk Customers', (3, 2, 2): 'Fading Interest', (3, 2, 3): 'One-Time Buyers',
        (3, 3, 1): 'Window Shoppers', (3, 3, 2): 'Window Shoppers', (3, 3, 3): 'One-Time Buyers',
        (2, 1, 1): 'At-Risk Customers'
    }

    all_segments = list(high_value_segments.keys()) + list(nurture_segments.keys()) + list(risk_segments.keys())
    all_subsegments = list(high_value_segments.values()) + list(nurture_segments.values()) + list(
        risk_segments.values())

    assert len(all_segments) == len(all_subsegments), "Lengths of segment and subsegment lists must match"

    for rfm in rfm_column:
        recency = int(rfm[0])
        frequency = int(rfm[1])
        monetary = int(rfm[2])

        if (recency, frequency, monetary) in all_segments:
            broad_segments.append(
                'High Value' if (recency, frequency, monetary) in high_value_segments.keys()
                else 'Nurture' if (recency, frequency, monetary) in nurture_segments.keys()
                else 'Risk'
            )
            subsegments.append(all_subsegments[all_segments.index((recency, frequency, monetary))])
        else:
            broad_segments.append('Unknown')
            subsegments.append('Unknown')

    return pd.Series(broad_segments, name='Broad Segment'), pd.Series(subsegments, name='Subsegment')


def prepare_data(df, customer_id_col, datetime_col, monetary_value_col, observation_period_end):
    summary = summary_data_from_transaction_data(
        df,
        customer_id_col=customer_id_col,
        datetime_col=datetime_col,
        monetary_value_col=monetary_value_col,
        observation_period_end=observation_period_end
    )
    summary = summary[summary["monetary_value"] > 0]
    print("Done fitting..")
    return summary


def fit_models(summary):
    print("Fitting...")
    bgf = BetaGeoFitter(penalizer_coef=0.5)
    bgf.fit(summary['frequency'], summary['recency'], summary['T'])
    print("Gamma Fittiing...")

    ggf = GammaGammaFitter(penalizer_coef=0.0)
    ggf.fit(summary['frequency'], summary['monetary_value'])

    return bgf, ggf


def predict_variables(summary, bgf, ggf, threshold):
    summary['probability_alive'] = bgf.conditional_probability_alive(
        summary['frequency'],
        summary['recency'],
        summary['T']
    )
    summary['predicted_purchases'] = bgf.predict(30, summary['frequency'], summary['recency'], summary['T'])
    summary['predicted_clv'] = ggf.customer_lifetime_value(
        bgf,
        summary['frequency'],
        summary['recency'],
        summary['T'],
        summary['monetary_value'],
        time=1,
        freq='D',
        discount_rate=0.01
    )
    summary["estimated_monetary_value"] = ggf.conditional_expected_average_profit(
        summary['frequency'],
        summary['monetary_value']
    )
    return summary


def create_segment_columns(df, recency, frequency, monetary_value, clv):
    df['R_score'] = pd.qcut(df[recency], q=3, labels=[1, 2, 3])
    df['F_score'] = pd.qcut(df[frequency], q=3, labels=[1, 2, 3])
    df['M_score'] = pd.qcut(df[monetary_value], q=3, labels=[1, 2, 3])
    df['CLV_segment'] = pd.qcut(df[clv], q=3, labels=['Low CLV', 'Medium CLV', 'High CLV'])

    rfm_mapping = {'111': 'High Value', '112': 'High Value', '113': 'Risk', '121': 'High Value', '122': 'Nurture',
                   '123': 'Risk', '131': 'High Value', '132': 'Nurture', '133': 'Risk', '211': 'Nurture',
                   '212': 'Nurture', '213': 'Risk', '221': 'Nurture', '222': 'Nurture', '223': 'Risk', '231': 'Risk',
                   '232': 'Risk', '233': 'Risk', '311': 'Risk', '312': 'Risk', '313': 'Risk', '321': 'Risk',
                   '322': 'Risk', '323': 'Risk', '331': 'Risk', '332': 'Risk', '333': 'Risk'}

    df['RFM'] = df['R_score'].astype(str) + df['F_score'].astype(str) + df['M_score'].astype(str)
    df['Segment'] = df['RFM'].map(rfm_mapping)
    df['Subsegment'] = df['Segment'].astype(str) + ', ' + df['CLV_segment'].astype(str)
    df = df.reset_index()
    df.drop(columns=['R_score', 'F_score', 'M_score'], inplace=True)

    return df


def calculate_segment_percentages(df):
    total_customers = len(df)
    segment_counts = df['Segment'].value_counts()
    segment_percentages = (segment_counts / total_customers * 100).round(2)
    subsegment_counts = df['Subsegment'].value_counts()
    subsegment_percentages = (subsegment_counts / total_customers * 100).round(2)
    percentages_dict = {
        'Segment': segment_percentages.to_dict(),
        'Subsegment': subsegment_percentages.to_dict()
    }
    return percentages_dict


def calculate_descriptive_statistics(df, fields):
    segment_stats = df.groupby('Segment')[fields].agg(['mean', 'std', 'median']).reset_index()
    subsegment_stats = df.groupby('Subsegment')[fields].agg(['mean', 'std', 'median']).reset_index()
    segment_stats.columns = ['_'.join(col).strip() if isinstance(col, tuple) else col for col in
                             segment_stats.columns.values]
    subsegment_stats.columns = ['_'.join(col).strip() if isinstance(col, tuple) else col for col in
                                subsegment_stats.columns.values]
    percentages_dict = calculate_segment_percentages(df)
    all_data_dict = {
        'segment_stats': segment_stats.to_dict(orient='records'),
        'subsegment_stats': subsegment_stats.to_dict(orient='records'),
        'percentages': percentages_dict
    }
    return segment_stats, subsegment_stats, percentages_dict, all_data_dict


def create_marketing_data(df, project_name):
    df['Status'] = df['probability_alive'].apply(
        lambda x: 'inactive' if x < 0.4 else 'regular' if x < 0.6 else 'active')

    fake = Faker()
    specific_emails = ['davidtgondo@gmail.com', 'd.gondo@alustudent.com']

    def generate_email(index):
        if index < len(specific_emails):
            return specific_emails[index]
        return fake.email()

    df['Email'] = [generate_email(i) for i in range(len(df))]
    df.loc[df['Email'].isin(specific_emails), 'Segment'] = 'Test'
    result_df = df[['CustomerID', 'Email', 'Segment', 'Subsegment', 'Status']]

    result_csv_content = result_df.to_csv(index=False)
    if not multipart_upload_to_s3(result_csv_content, S3_BUCKET_NAME, f"{project_name}marketing.csv"):
        raise Exception(f"Failed to upload marketing data to S3: {project_name}marketing.csv")

    return result_df


def get_discounted_bundles(project_name, segment_names):
    bundle_info_dict = {}

    for segment_name in segment_names:
        bundles_csv_content = multipart_download_from_s3(S3_BUCKET_NAME, f"{project_name}/{segment_name}_itemsets.csv")
        if bundles_csv_content is None:
            raise Exception(f"Failed to download bundles CSV from S3: {project_name}/{segment_name}_itemsets.csv")
        bundles_df = pd.read_csv(StringIO(bundles_csv_content))

        first_10_bundles = bundles_df['itemset'].head(10)

        products_csv_content = multipart_download_from_s3(S3_BUCKET_NAME, f"{project_name}/products.csv")
        if products_csv_content is None:
            raise Exception(f"Failed to download products CSV from S3: {project_name}/products.csv")
        products_df = pd.read_csv(StringIO(products_csv_content))

        stock_code_to_description = dict(zip(products_df['StockCode'], products_df['Description']))

        bundle_info = []

        for bundle_str in first_10_bundles:
            bundle_stock_codes = [code.strip().strip("'") for code in bundle_str.strip('{}').split(',')]
            bundle_description = []
            actual_price = 0
            discounted_price = 0

            for stock_code in bundle_stock_codes:
                description = stock_code_to_description.get(stock_code)
                if description:
                    product_info = products_df[products_df['StockCode'] == stock_code]
                    discounted_price_next_product = product_info.iloc[0]['DiscountedPrice']
                    bundle_description.append(description)
                    discounted_price += discounted_price_next_product
                    unit_price = float(product_info.iloc[0]['UnitPrice'])
                    actual_price += unit_price

            bundle_description_str = ', '.join(bundle_description)
            bundle_id = shortuuid.uuid()
            bundle_info.append({
                'id': bundle_id,
                'bundle': bundle_description_str,
                'actual_price': actual_price,
                'discounted_price': discounted_price
            })

        bundle_info_df = pd.DataFrame(bundle_info)
        bundle_info_csv_content = bundle_info_df.to_csv(index=False)
        if not multipart_upload_to_s3(bundle_info_csv_content, S3_BUCKET_NAME,
                                      f"{project_name}/{segment_name}_discounted_bundles.csv"):
            raise Exception(
                f"Failed to upload discounted bundles CSV to S3: {project_name}/{segment_name}_discounted_bundles.csv")

        bundle_info_dict[segment_name] = bundle_info_df

    return bundle_info_dict


def get_association_info(project_name, segment_names):
    association_info_dict = {}
    all_associations = []

    products_csv_content = multipart_download_from_s3(S3_BUCKET_NAME, f"{project_name}/products_data.csv")
    if products_csv_content is None:
        raise Exception(f"Failed to download products CSV from S3: {project_name}/products_data.csv")
    products_df = pd.read_csv(StringIO(products_csv_content))

    stock_code_to_info = products_df.set_index('StockCode').to_dict('index')

    for segment_name in segment_names:
        associations_csv_content = multipart_download_from_s3(S3_BUCKET_NAME,
                                                              f"{project_name}/{segment_name}_associations.csv")
        if associations_csv_content is None:
            raise Exception(
                f"Failed to download associations CSV from S3: {project_name}/{segment_name}_associations.csv")
        associations_df = pd.read_csv(StringIO(associations_csv_content))

        association_info = []

        for _, row in associations_df.iterrows():
            basket_str = row['basket']
            next_product_str = row['next_product']
            basket_stock_codes = [code.strip().strip("'") for code in basket_str.strip('{}').split(',')]
            next_product_stock_codes = [code.strip().strip("'") for code in next_product_str.strip('{}').split(',')]
            basket_description = []
            next_product_description = []
            actual_price = 0
            next_product_actual_price = 0
            next_product_discounted_price = 0

            for stock_code in basket_stock_codes:
                if stock_code in stock_code_to_info:
                    product_info = stock_code_to_info[stock_code]
                    description = product_info['Description']
                    unit_price = product_info['UnitPrice']
                    basket_description.append(f"{description} (${unit_price})")
                    actual_price += unit_price

            for stock_code in next_product_stock_codes:
                if stock_code in stock_code_to_info:
                    product_info = stock_code_to_info[stock_code]
                    description = product_info['Description']
                    discounted_price_next_product = product_info['DiscountedPrice']
                    next_product_discounted_price += discounted_price_next_product
                    next_product_actual_price += discounted_price_next_product
                    next_product_description.append(
                        f"{description} [FROM  (${next_product_actual_price}) TO(${discounted_price_next_product})]")

            basket_description_str = ', '.join(basket_description)
            next_product_description_str = ', '.join(next_product_description)
            association_id = shortuuid.uuid()
            association_info.append({
                'id': association_id,
                'bundle': basket_description_str,
                'discount_bundle': next_product_description_str,
                'total_price': actual_price + next_product_actual_price,
                'discounted_price': actual_price + next_product_actual_price - next_product_discounted_price,
                'segment': segment_name
            })
            all_associations.append({
                'id': association_id,
                'bundle': basket_description_str,
                'discount_bundle': next_product_description_str,
                'total_price': actual_price + next_product_actual_price,
                'discounted_price': actual_price + next_product_actual_price - next_product_discounted_price,
                'segment': segment_name
            })

        association_info_dict[segment_name] = association_info

    all_associations_df = pd.DataFrame(all_associations)
    output_csv_content = all_associations_df.to_csv(index=False)
    if not multipart_upload_to_s3(output_csv_content, S3_BUCKET_NAME, f"{project_name}/bogd_offers.csv"):
        raise Exception(f"Failed to upload BOGD offers CSV to S3: {project_name}/bogd_offers.csv")

    return all_associations_df


def save_custom_discounts(data, file_name, project_name):
    file_content = json.dumps(data, indent=4)
    if not multipart_upload_to_s3(file_content, S3_BUCKET_NAME, f"{project_name}/{file_name}"):
        raise Exception(f"Failed to upload custom discounts JSON to S3: {project_name}/{file_name}")
    print(f"Data has been saved to {project_name}/{file_name}")


def get_bundlesets(high_bundle, risk_bundle, nurture_bundle, project_name):
    bundle_data = {"high_value": high_bundle, "risk": risk_bundle, "nurture": nurture_bundle}

    for bundle_name, bundle_df in bundle_data.items():
        associations, itemsets = get_bundles(bundle_df.copy())
        associations_csv_content = associations.to_csv(index=False)
        itemsets_csv_content = itemsets.to_csv(index=False)

        if not multipart_upload_to_s3(associations_csv_content, S3_BUCKET_NAME,
                                      f"{project_name}/{bundle_name}_associations.csv"):
            raise Exception(f"Failed to upload associations CSV to S3: {project_name}/{bundle_name}_associations.csv")
        if not multipart_upload_to_s3(itemsets_csv_content, S3_BUCKET_NAME,
                                      f"{project_name}/{bundle_name}_itemsets.csv"):
            raise Exception(f"Failed to upload itemsets CSV to S3: {project_name}/{bundle_name}_itemsets.csv")

        print(f"Bundle '{bundle_name}' associations and itemsets saved to S3 at {project_name}.")


def get_bundles(df):
    hbasket = aggregate_transactions(df)
    freqItemSet, rules = fpgrowth(hbasket['StockCode'].values, minSupRatio=0.01, minConf=0.9)
    print('Number of rules generated: ', len(rules))

    associations = pd.DataFrame(rules, columns=['basket', 'next_product', 'proba'])
    associations = associations.sort_values(by='proba', ascending=False)

    itemsets = pd.DataFrame({'itemset': freqItemSet})
    itemsets['support'] = itemsets['itemset'].apply(
        lambda x: hbasket[hbasket['StockCode'].apply(lambda y: set(x).issubset(set(y)))].shape[0] / len(hbasket))
    itemsets = itemsets[itemsets['itemset'].apply(lambda x: len(x) > 2)]
    itemsets = itemsets.sort_values(by='support', ascending=False)

    return associations, itemsets


def create_bogd_offers(project_name, segment_names=['high_value', 'nurture', 'risk']):
    association_info_dict = {}
    all_associations = []

    products_csv_content = multipart_download_from_s3(S3_BUCKET_NAME, f"{project_name}/products_data.csv")
    if products_csv_content is None:
        raise Exception(f"Failed to download products CSV from S3: {project_name}/products_data.csv")
    products_df = pd.read_csv(StringIO(products_csv_content))

    stock_code_to_info = products_df.set_index('StockCode').to_dict('index')

    for segment_name in segment_names:
        associations_csv_content = multipart_download_from_s3(S3_BUCKET_NAME,
                                                              f"{project_name}/{segment_name}_associations.csv")
        if associations_csv_content is None:
            raise Exception(
                f"Failed to download associations CSV from S3: {project_name}/{segment_name}_associations.csv")
        associations_df = pd.read_csv(StringIO(associations_csv_content))

        association_info = []

        for _, row in associations_df.iterrows():
            basket_str = row['basket']
            next_product_str = row['next_product']
            basket_stock_codes = [code.strip().strip("'") for code in basket_str.strip('{}').split(',')]
            next_product_stock_codes = [code.strip().strip("'") for code in next_product_str.strip('{}').split(',')]
            basket_description = []
            next_product_description = []
            actual_price = 0
            next_product_actual_price = 0
            next_product_discounted_price = 0

            for stock_code in basket_stock_codes:
                if stock_code in stock_code_to_info:
                    product_info = stock_code_to_info[stock_code]
                    description = product_info['Description']
                    unit_price = product_info['UnitPrice']
                    basket_description.append(f"{description} (${unit_price})")
                    actual_price += unit_price

            for stock_code in next_product_stock_codes:
                if stock_code in stock_code_to_info:
                    product_info = stock_code_to_info[stock_code]
                    description = product_info['Description']
                    discounted_price_next_product = product_info['DiscountedPrice']
                    next_product_discounted_price += discounted_price_next_product
                    next_product_actual_price += discounted_price_next_product
                    next_product_description.append(
                        f"{description} [FROM  (${next_product_actual_price}) TO(${discounted_price_next_product})]")

            basket_description_str = ', '.join(basket_description)
            next_product_description_str = ', '.join(next_product_description)
            association_id = shortuuid.uuid()
            association_info.append({
                'id': association_id,
                'bundle': basket_description_str,
                'discount_bundle': next_product_description_str,
                'total_price': actual_price + next_product_actual_price,
                'discounted_price': actual_price + next_product_actual_price - next_product_discounted_price,
                'segment': segment_name
            })
            all_associations.append({
                'id': association_id,
                'bundle': basket_description_str,
                'discount_bundle': next_product_description_str,
                'total_price': actual_price + next_product_actual_price,
                'discounted_price': actual_price + next_product_actual_price - next_product_discounted_price,
                'segment': segment_name
            })

        association_info_dict[segment_name] = association_info

    all_associations_df = pd.DataFrame(all_associations)
    output_csv_content = all_associations_df.to_csv(index=False)
    if not multipart_upload_to_s3(output_csv_content, S3_BUCKET_NAME, f"{project_name}/bogd_offers.csv"):
        raise Exception(f"Failed to upload BOGD offers CSV to S3: {project_name}/bogd_offers.csv")

    return all_associations_df

