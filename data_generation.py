import os

import numpy as np
import pandas as pd
import shortuuid
from shortuuid import uuid



def generate_products_data(df, min_margin=0.02, min_discount=0.02, max_discount=0.50,
                           output_file="products_data.csv"):
    """
    This function randomly assigns profit margins and discounts to products in a DataFrame.

    Args:
        df (pandas.DataFrame): DataFrame containing product data with columns:
            - StockCode (str): Unique product code.
            - UnitPrice (float): Original unit price of the product.
        min_margin (float, optional): Minimum profit margin percentage (default: 0.02).
        min_discount (float, optional): Minimum discount percentage (default: 0.02).
        max_discount (float, optional): Maximum discount percentage (default: 0.50).
        output_file (str, optional): Name of the output CSV file (default: "profit_margins_and_discounts.csv").

    Returns:
        str: Name of the output CSV file.
    """

    data = df[["id", "Description", "StockCode", "UnitPrice"]]
    # Drop duplicates based on the 'StockCode' column
    data = data.drop_duplicates(subset=['StockCode'])

    def generate_profit_margin(unit_price):
        """
        Generates a random profit margin for a given unit price.

        Args:
            unit_price (float): Original unit price of the product.

        Returns:
            float: Randomly generated profit margin.
        """
        # Generate a random percentage for profit margin
        random_margin = np.random.uniform(min_margin, 1.0)  # Ensure profit margin is at least 2%
        # Calculate the profit margin
        profit_margin = unit_price * random_margin
        profit_margin = round(profit_margin, 2)

        return profit_margin

    def calculate_discounted_price(unit_price, profit_margin):
        """
        Calculates a random discount that maintains a minimum profit margin.

        Args:
            unit_price (float): Original unit price of the product.
            profit_margin (float): Current profit margin of the product.

        Returns:
            tuple: Tuple containing the discounted price (float) and discount percentage (float).
        """
        max_discount = 1 - profit_margin / unit_price  # Maximum discount allowed to maintain minimum profit

        # Generate random discount between min_discount and max_discount
        discount = np.random.uniform(min_discount, max_discount)

        # Round the discount to 2 decimal places
        discount = round(discount, 2)

        # Calculate the discounted price
        discounted_price = round(unit_price * (1 - discount), 2)
        return discounted_price, discount

    # Apply profit margin and discount calculation to each row
    data['ProfitMargin'] = data['UnitPrice'].apply(generate_profit_margin)
    data['DiscountedPrice'], data['DiscountPct'] = zip(
        *data.apply(lambda x: calculate_discounted_price(x['UnitPrice'], x['ProfitMargin']), axis=1))

    # Save the DataFrame to a CSV file
    data.to_csv(output_file, index=False)

    return data


def get_segment_transactions(df, root_path):
    # Filter data based on segments
    high_value_df = df[df['Segment'] == 'High Value']
    risk_df = df[df['Segment'] == 'Risk']
    nurture_df = df[df['Segment'] == 'Nurture']

    # Write each segment to a CSV file
    high_value_path = os.path.join(root_path, 'high_value_transactions.csv')
    risk_path = os.path.join(root_path, 'risk_transactions.csv')
    nurture_path = os.path.join(root_path, 'nurture_transactions.csv')

    high_value_df.to_csv(high_value_path, index=False)
    risk_df.to_csv(risk_path, index=False)
    nurture_df.to_csv(nurture_path, index=False)

    return high_value_path, risk_path, nurture_path


def generate_bundle_info(root_path, segment_names):
    """
    Retrieve information about product bundles for multiple segments.

    Args:
        api_name (str): The root path for the API (also the folder name containing the files).
        segment_names (list): A list of segment names for which to retrieve bundle information.

    Returns:
        dict: A dictionary where keys are segment names and values are DataFrames containing information about
              the product bundles for each segment. Each DataFrame is also saved to a CSV file.
    """
    bundle_info_dict = {}

    for segment_name in segment_names:
        # Read the product bundles CSV file for the current segment
        bundles_filename = os.path.join(root_path, f"{segment_name}_itemsets.csv")
        bundles_df = pd.read_csv(bundles_filename)

        # Get the first 10 bundles
        first_10_bundles = bundles_df['itemset'].head(10)

        # Read the products CSV file
        products_filename = os.path.join(root_path, 'products_data.csv')
        products_df = pd.read_csv(products_filename)

        # Create a dictionary to map stock codes to descriptions
        stock_code_to_description = dict(zip(products_df['StockCode'], products_df['Description']))

        bundle_info = []

        for bundle_str in first_10_bundles:
            # Split the bundle string into individual stock codes and remove extra whitespaces
            bundle_stock_codes = [code.strip().strip("'") for code in bundle_str.strip('{}').split(',')]

            # Initialize variables to store information about the bundle
            bundle_description = []
            actual_price = 0
            discounted_price = 0

            for stock_code in bundle_stock_codes:
                # Find the description for the current stock code
                description = stock_code_to_description.get(stock_code)
                if description:
                    # Find the product information for the current stock code
                    product_info = products_df[products_df['StockCode'] == stock_code]

                    # Get the unit price, discounted price, and discount percentage
                    unit_price = float(product_info.iloc[0]['UnitPrice'])
                    discount_pct = float(product_info.iloc[0]['DiscountPct'])

                    # Calculate the discounted price
                    discounted_price += unit_price * (1 - discount_pct)

                    # Add the description to the bundle description
                    bundle_description.append(description)

                    # Add the unit price to the actual price
                    actual_price += unit_price

            # Format the bundle description as a string
            bundle_description_str = ', '.join(bundle_description)

            # Round the discounted price to 2 decimal places
            discounted_price = round(discounted_price, 2)

            # Generate a random UUID for the bundle ID
            # bundle_id = str(uuid)
            bundle_id = shortuuid.uuid()

            # Create a dictionary for the bundle information
            bundle_info.append({
                'id': bundle_id,
                'bundle': bundle_description_str,
                'actual_price': actual_price,
                'discounted_price': discounted_price
            })

        # Convert bundle_info to a DataFrame
        bundle_info_df = pd.DataFrame(bundle_info)

        # Save bundle_info_df to a CSV file
        output_filename = os.path.join(root_path, f"{segment_name}_bundles_info.csv")
        bundle_info_df.to_csv(output_filename, index=False)

        # Add the DataFrame to the dictionary
        bundle_info_dict[segment_name] = bundle_info_df

    return bundle_info_dict
