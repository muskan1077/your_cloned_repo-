import pandas as pd
import networkx as nx
import numpy as np
import datetime


def calculate_distance_matrix(df) -> pd.DataFrame:
    """
    Calculate a distance matrix based on the DataFrame.

    Args:
        df (pandas.DataFrame): DataFrame with 'id_start', 'id_end', and 'distance' columns.

    Returns:
        pandas.DataFrame: Distance matrix.
    """
    # Create a graph from the DataFrame
    G = nx.from_pandas_edgelist(df, 'id_start', 'id_end', ['distance'])

    # Get a list of nodes
    nodes = list(G.nodes)

    # Initialize an empty distance matrix
    distance_matrix = pd.DataFrame(index=nodes, columns=nodes, dtype=float)

    # Fill in the distance matrix with cumulative distances
    for node1 in nodes:
        for node2 in nodes:
            if node1 == node2:
                # Diagonal values are set to 0
                distance_matrix.loc[node1, node2] = 0
            elif G.has_edge(node1, node2):
                # If there's a direct edge, use the distance value
                distance_matrix.loc[node1, node2] = G[node1][node2]['distance']
            else:
                # If no direct edge, calculate the shortest path distance
                shortest_path = nx.shortest_path_length(G, node1, node2, weight='distance')
                distance_matrix.loc[node1, node2] = shortest_path

    return distance_matrix



def unroll_distance_matrix(result_matrix):
    """
    Unroll a distance matrix to a DataFrame in the style of the initial dataset.

    Args:
        df (pandas.DataFrame)

    Returns:
        pandas.DataFrame: Unrolled DataFrame containing columns 'id_start', 'id_end', and 'distance'.
    """
    # Get the upper triangular matrix excluding the diagonal
    upper_triangle = result_matrix.where(pd.notna(np.triu(result_matrix, k=1)))

    # Stack the upper triangular matrix to create a Series
    stacked_series = upper_triangle.stack()

    # Reset index to convert MultiIndex to columns
    unrolled_df = stacked_series.reset_index()

    # Rename the columns
    unrolled_df.columns = ['id_start', 'id_end', 'distance']

    return unrolled_df

def find_ids_within_ten_percentage_threshold(result_df, reference_value):
    """
    Find all IDs whose average distance lies within 10% of the average distance of the reference ID.

    Args:
        df (pandas.DataFrame)
        reference_id (int)

    Returns:
        pandas.DataFrame: DataFrame with IDs whose average distance is within the specified percentage threshold
                          of the reference ID's average distance.
    """
    # Filter rows based on the reference value
    reference_rows = result_df[result_df['id_start'] == reference_value]

    # Calculate the average distance for the reference value
    average_distance = reference_rows['distance'].mean()

    # Calculate the threshold values
    lower_threshold = average_distance - (average_distance * 0.1)
    upper_threshold = average_distance + (average_distance * 0.1)

    # Filter rows within the 10% threshold
    result_df = result_df[(result_df['distance'] >= lower_threshold) & (result_df['distance'] <= upper_threshold)]

    # Extract and sort unique values from the 'id_start' column
    sorted_ids_within_threshold = sorted(result_df['id_start'].unique())

    return sorted_ids_within_threshold


def calculate_toll_rate(result_df):
    """
    Calculate toll rates for each vehicle type based on the unrolled DataFrame.

    Args:
        df (pandas.DataFrame)

    Returns:
        pandas.DataFrame
    """
    # Define rate coefficients for each vehicle type
    rate_coefficients = {'moto': 0.8, 'car': 1.2, 'rv': 1.5, 'bus': 2.2, 'truck': 3.6}

    # Create new columns for each vehicle type and calculate toll rates
    for vehicle_type, coefficient in rate_coefficients.items():
        result_df[vehicle_type] = result_df['distance'] * coefficient

    return result_df



def calculate_time_based_toll_rates(result_df):
     """
    Calculate time-based toll rates for different time intervals within a day.

    Args:
        df (pandas.DataFrame)

    Returns:
        pandas.DataFrame
    """
    # Create new columns for start_day, start_time, end_day, and end_time
    result_df['start_day'] = pd.to_datetime(result_df['id_start']).dt.day_name()
    result_df['start_time'] = pd.to_datetime(result_df['id_start']).dt.time
    result_df['end_day'] = pd.to_datetime(result_df['id_end']).dt.day_name()
    result_df['end_time'] = pd.to_datetime(result_df['id_end']).dt.time

    # Define time ranges
    weekdays_discounts = {
        (datetime.time(0, 0, 0), datetime.time(10, 0, 0)): 0.8,
        (datetime.time(10, 0, 0), datetime.time(18, 0, 0)): 1.2,
        (datetime.time(18, 0, 0), datetime.time(23, 59, 59)): 0.8
    }

    weekends_discount = 0.7

    # Apply discounts based on time ranges
    for index, row in df.iterrows():
        if row['start_day'] in ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']:
            for time_range, discount in weekdays_discounts.items():
                if time_range[0] <= row['start_time'] <= time_range[1]:
                    result_df.at[index, 'moto'] *= discount
                    result_df.at[index, 'car'] *= discount
                    result_df.at[index, 'rv'] *= discount
                    result_df.at[index, 'bus'] *= discount
                    result_df.at[index, 'truck'] *= discount
        else:
            result_df.at[index, 'moto'] *= weekends_discount
            result_df.at[index, 'car'] *= weekends_discount
            result_df.at[index, 'rv'] *= weekends_discount
            result_df.at[index, 'bus'] *= weekends_discount
            result_df.at[index, 'truck'] *= weekends_discount

    return result_df