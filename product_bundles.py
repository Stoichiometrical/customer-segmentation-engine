import json
import os
import pandas as pd
import shortuuid
from fpgrowth_py import fpgrowth


from utility_methods import aggregate_transactions


# Get discounted bundles
def get_discounted_bundles(api_name, segment_names):
    """
    Retrieve information about product bundles for multiple segments.

    Args:
        api_name (str): The root path for the files.
        segment_names (list): A list of segment names for which to retrieve bundle information.

    Returns:
        dict: A dictionary where keys are segment names and values are DataFrames containing information about
            the product bundles for each segment. Each DataFrame is also saved to a CSV file.

    This function reads CSV files containing product bundles for the specified segments. It then retrieves
    information about the bundles, including their descriptions, actual prices, and discounted prices, by querying
    a separate CSV file containing information about individual products. The discounted prices are rounded to
    2 decimal places.
    """

    bundle_info_dict = {}

    for segment_name in segment_names:
        # Read the product bundles CSV file for the current segment
        bundles_filename = os.path.join(api_name, f"{segment_name}_itemsets.csv")
        bundles_df = pd.read_csv(bundles_filename)

        # Get the first 10 bundles
        first_10_bundles = bundles_df['itemset'].head(10)

        # Read the products CSV file
        products_filename = os.path.join(api_name, "products.csv")
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

                    # Get the discounted price
                    discounted_price_next_product = product_info.iloc[0]['DiscountedPrice']

                    # Add the description to the bundle description
                    bundle_description.append(description)

                    # Add the discounted price to the total discounted price
                    discounted_price += discounted_price_next_product

                    # Get the unit price and add it to the actual price
                    unit_price = float(product_info.iloc[0]['UnitPrice'])
                    actual_price += unit_price

            # Format the bundle description as a string
            bundle_description_str = ', '.join(bundle_description)

            # Generate a short UUID for the bundle ID
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
        bundle_info_df.to_csv(os.path.join(api_name, f"{segment_name}_discounted_bundles.csv"), index=False)

        # Add the DataFrame to the dictionary
        bundle_info_dict[segment_name] = bundle_info_df

    return bundle_info_dict


# Create BOGD dataset
def get_association_info(api_name, segment_names):
    """
    Retrieve association information for multiple segments.

    Args:
        api_name (str): The name of the API (also the folder name containing the files).
        segment_names (list): A list of segment names for which to retrieve association information.

    Returns:
        dict: A dictionary where keys are segment names and values are lists of dictionaries containing information about
            the associations for each segment.
        pd.DataFrame: A DataFrame containing the combined information for all segments.
    """
    association_info_dict = {}
    all_associations = []

    # Path to the products file
    products_file_path = os.path.join(api_name, 'products_data.csv')

    # Read the products CSV file
    products_df = pd.read_csv(products_file_path)

    # Create a dictionary to map stock codes to descriptions and prices
    stock_code_to_info = products_df.set_index('StockCode').to_dict('index')

    for segment_name in segment_names:
        # Read the association CSV file for the current segment
        associations_filename = os.path.join(api_name, f"{segment_name}_associations.csv")
        associations_df = pd.read_csv(associations_filename)

        association_info = []

        for _, row in associations_df.iterrows():
            basket_str = row['basket']
            next_product_str = row['next_product']

            # Parse the basket and next_product strings into lists of stock codes
            basket_stock_codes = [code.strip().strip("'") for code in basket_str.strip('{}').split(',')]
            next_product_stock_codes = [code.strip().strip("'") for code in next_product_str.strip('{}').split(',')]

            # Initialize variables to store information about the basket and next product
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
                    next_product_actual_price += discounted_price_next_product  # Use the discounted price as actual price

                    next_product_description.append(
                        f"{description} [FROM  (${next_product_actual_price}) TO(${discounted_price_next_product})]")

            # Format the basket and next product descriptions as strings
            basket_description_str = ', '.join(basket_description)
            next_product_description_str = ', '.join(next_product_description)

            # Generate a short UUID for the association ID
            association_id = shortuuid.uuid()

            # Create a dictionary for the association information
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
                'discounted_price': actual_price + next_product_discounted_price,
                'segment': segment_name
            })

        # Add the association information to the dictionary
        association_info_dict[segment_name] = association_info

    # Convert the list of all associations to a DataFrame
    all_associations_df = pd.DataFrame(all_associations)

    # Save the DataFrame to a CSV file
    output_csv_path = os.path.join(api_name, 'bogd_offers.csv')
    all_associations_df.to_csv(output_csv_path, index=False)

    return all_associations_df


def save_custom_discounts(data, file_name, root_path):
    """
    Converts a dictionary to a JSON file and saves it to the api.

    :param data: The dictionary to be converted.
    :param file_name: The name of the JSON file to save.
    :param root_path: The root directory where the JSON file will be saved.
    """
    try:
        # Ensure the root path exists, if not create it

        if not os.path.exists(root_path):
            os.makedirs(root_path)

        # Create the full file path
        file_path = os.path.join(root_path, file_name)

        # Write the data to the JSON file
        with open(file_path, 'w') as file:
            json.dump(data, file, indent=4)

        print(f"Data has been saved to {file_path}")
    except Exception as e:
        print(f"An error occurred: {e}")


##Get bundlesets
def get_bundlesets(high_bundle, risk_bundle, nurture_bundle, root_path):
    """
    This function takes three DataFrames (high_bundle, risk_bundle, nurture_bundle)
    and generates association rules and frequent itemsets (as CSV files) for each.

    Args:
        high_bundle (pandas.DataFrame): DataFrame containing high-potential customer data.
        risk_bundle (pandas.DataFrame): DataFrame containing risk customer data.
        nurture_bundle (pandas.DataFrame): DataFrame containing nurture customer data.
        root_path (str): Root directory where the CSV files will be saved.
    """

    # Ensure the root_path exists
    os.makedirs(root_path, exist_ok=True)

    bundle_data = {"high_value": high_bundle, "risk": risk_bundle, "nurture": nurture_bundle}

    for bundle_name, bundle_df in bundle_data.items():
        associations, itemsets = get_bundles(bundle_df.copy())

        # Define file paths
        associations_file_path = os.path.join(root_path, f"{bundle_name}_associations.csv")
        itemsets_file_path = os.path.join(root_path, f"{bundle_name}_itemsets.csv")

        # Save associations to CSV
        associations.to_csv(associations_file_path, index=False)

        # Save itemsets to CSV
        itemsets.to_csv(itemsets_file_path, index=False)

        print(f"Bundle '{bundle_name}' associations and itemsets saved to CSV files at {root_path}.")


def get_bundles(df):
    hbasket = aggregate_transactions(df)
    freqItemSet, rules = fpgrowth(hbasket['StockCode'].values, minSupRatio=0.01, minConf=0.9)
    print('Number of rules generated: ', len(rules))

    associations = pd.DataFrame(rules, columns=['basket', 'next_product', 'proba'])
    associations = associations.sort_values(by='proba', ascending=False)

    itemsets = pd.DataFrame({'itemset': freqItemSet})
    itemsets['support'] = itemsets['itemset'].apply(
        lambda x: hbasket[hbasket['StockCode'].apply(lambda y: set(x).issubset(set(y)))].shape[0] / len(hbasket))
    itemsets = itemsets[itemsets['itemset'].apply(lambda x: len(x) > 2)]  # Filter out itemsets with only one item
    itemsets = itemsets.sort_values(by='support', ascending=False)  # Sort itemsets by support

    return associations, itemsets


#Create BOGD Offers
segment_names = ['high_value', 'nurture', 'risk']


def create_bogd_offers(root_path, segment_names=segment_names):
    """
    Retrieve association information for multiple segments.

    Args:
        api_name (str): The name of the API (also the folder name containing the files).
        segment_names (list): A list of segment names for which to retrieve association information.

    Returns:
        dict: A dictionary where keys are segment names and values are lists of dictionaries containing information about
            the associations for each segment.
        pd.DataFrame: A DataFrame containing the combined information for all segments.
    """
    association_info_dict = {}
    all_associations = []

    # Path to the products file
    products_file_path = os.path.join(root_path, 'products_data.csv')

    # Read the products CSV file
    products_df = pd.read_csv(products_file_path)

    # Create a dictionary to map stock codes to descriptions and prices
    stock_code_to_info = products_df.set_index('StockCode').to_dict('index')

    for segment_name in segment_names:
        # Read the association CSV file for the current segment
        associations_filename = os.path.join(root_path, f"{segment_name}_associations.csv")
        associations_df = pd.read_csv(associations_filename)

        association_info = []

        for _, row in associations_df.iterrows():
            basket_str = row['basket']
            next_product_str = row['next_product']

            # Parse the basket and next_product strings into lists of stock codes
            basket_stock_codes = [code.strip().strip("'") for code in basket_str.strip('{}').split(',')]
            next_product_stock_codes = [code.strip().strip("'") for code in next_product_str.strip('{}').split(',')]

            # Initialize variables to store information about the basket and next product
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
                    next_product_actual_price += discounted_price_next_product  # Use the discounted price as actual price

                    next_product_description.append(
                        f"{description} [FROM  (${next_product_actual_price}) TO(${discounted_price_next_product})]")

            # Format the basket and next product descriptions as strings
            basket_description_str = ', '.join(basket_description)
            next_product_description_str = ', '.join(next_product_description)

            # Generate a short UUID for the association ID
            association_id = shortuuid.uuid()

            # Create a dictionary for the association information
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
                'discounted_price': actual_price + next_product_discounted_price,
                'segment': segment_name
            })

        # Add the association information to the dictionary
        association_info_dict[segment_name] = association_info

    # Convert the list of all associations to a DataFrame
    all_associations_df = pd.DataFrame(all_associations)

    # Save the DataFrame to a CSV file
    output_csv_path = os.path.join(root_path, 'bogd_offers.csv')
    all_associations_df.to_csv(output_csv_path, index=False)

    return all_associations_df
