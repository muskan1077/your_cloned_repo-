import pandas as pd
df = pd.read_csv('dataset-1.csv')


def generate_car_matrix(df)->pd.DataFrame:
    """
    Creates a DataFrame  for id combinations.

    Args:
        df (pandas.DataFrame)

    Returns:
        pandas.DataFrame: Matrix generated with 'car' values, 
                          where 'id_1' and 'id_2' are used as indices and columns respectively.
    """
    

    # Read the CSV file into a DataFrame
    #df = pd.read_csv(path)

    # Pivot the DataFrame using id_1 as index, id_2 as columns, and car as values
    result_df = df.pivot(index='id_1', columns='id_2', values='car').fillna(0)

    # Set diagonal values to 0
    result_df.values[[range(result_df.shape[0])]*2] = 0

    return result_df
    


def get_type_count(df)->dict:
    """
    Categorizes 'car' values into types and returns a dictionary of counts.

    Args:
        df (pandas.DataFrame)

    Returns:
        dict: A dictionary with car types as keys and their counts as values.
    """
    
    
    # Add a new categorical column 'car_type' based on the values of the 'car' column
    df['car_type'] = pd.cut(df['car'],
                            bins=[-float('inf'), 15, 25, float('inf')],
                            labels=['low', 'medium', 'high'],
                            include_lowest=True, right=False)

    # Calculate the count of occurrences for each 'car_type' category
    type_count = df['car_type'].value_counts().to_dict()

    # Sort the dictionary alphabetically based on keys
    sorted_type_count = dict(sorted(type_count.items()))

    return sorted_type_count



    


def get_bus_indexes(df)->list:
    """
    Returns the indexes where the 'bus' values are greater than twice the mean.

    Args:
        df (pandas.DataFrame)

    Returns:
        list: List of indexes where 'bus' values exceed twice the mean.
    """

    # Calculate the mean value of the 'bus' column
    bus_mean = df['bus'].mean()

    # Identify indices where the 'bus' values are greater than twice the mean
    bus_indexes = df[df['bus'] > 2 * bus_mean].index.tolist()

    # Sort the indices in ascending order
    bus_indexes.sort()

    return bus_indexes



def filter_routes(df) -> list:
    """
    Filters and returns routes with average 'truck' values greater than 7.

    Args:
        df (pandas.DataFrame): The DataFrame containing 'route' and 'truck' columns.

    Returns:
        list: List of route names with average 'truck' values greater than 7.
    """
    # Group by 'route' and calculate the average of 'truck' column for each group
    average_truck_by_route = df.groupby('route')['truck'].mean()

    # Filter routes where the average 'truck' value is greater than 7
    selected_routes = average_truck_by_route[average_truck_by_route > 7].index.tolist()

    # Sort the list of selected routes
    selected_routes.sort()

    return selected_routes



import pandas as pd

def multiply_matrix(result_df) -> pd.DataFrame:
    """
    Multiplies matrix values with custom conditions.

    Args:
        result_df (pandas.DataFrame): The input DataFrame to be modified.

    Returns:
        pandas.DataFrame: Modified DataFrame with values multiplied based on custom conditions.
    """
    # Copy the input DataFrame to avoid modifying the original DataFrame
    modified_matrix_df = result_df.copy()

    # Apply the multiplication logic based on the provided conditions
    modified_matrix_df = modified_matrix_df.applymap(lambda x: x * 0.75 if x > 20 else x * 1.25)

    # Round the values to 1 decimal place
    modified_matrix_df = modified_matrix_df.round(1)

    return modified_matrix_df



def time_check(df)->pd.Series:
    """
    Use shared dataset-2 to verify the completeness of the data by checking whether the timestamps for each unique (`id`, `id_2`) pair cover a full 24-hour and 7 days period

    Args:
        df (pandas.DataFrame)

    Returns:
        pd.Series: return a boolean series
    """
    # Write your logic here

    return pd.Series()
