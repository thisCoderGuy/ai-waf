import pandas as pd
import os
import io # Import io for StringIO

from loggers import global_logger, evaluation_logger

from config import (
    RAW_DATA_FILE_PATHS,  CLEANED_DATA_OUTPUT_PATH, CLEANED_DATA_PATH,
    PERFORM_DATA_CLEANING, COLUMNS_TO_DROP, PROBLEMATIC_ENDINGS,
    LABEL, LABEL_VALUES, CRITICAL_FEATURES,COLUMNS_TO_UPPERCASE
    )

from datetime import datetime

def _add_timestamp_to_filename(file_path: str) -> str:
    """
    Given a file path, inserts a timestamp before the file extension.

    Args:
        file_path (str): The original file path string (e.g., "data/output/results.csv").

    Returns:
        str: A new file path string with a timestamp (e.g., "data/output/results_20230709_183000.csv").
    """
    directory, full_filename = os.path.split(file_path)
    base_filename, extension = os.path.splitext(full_filename)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    new_filename = f"{base_filename}_{timestamp}{extension}"
    new_file_path = os.path.join(directory, new_filename)

    return new_file_path

def _load_raw_log_files():
    """
    Loads and merges CSV log entries from the specified list of files,
    performing initial line-by-line filtering based on problematic endings.

    Args:
        log_file_paths (list): A list of file paths to the raw CSV log files.
        problematic_endings (list): A list of string endings to filter out problematic lines.
        logger (logging.Logger): The logger instance for logging messages.

    Returns:
        pandas.DataFrame: A merged DataFrame of raw log entries after line-level filtering.
    """
    all_dfs = []
    global_logger.info("\dLoading raw log files and applying initial line filtering...")
    for raw_data_file in RAW_DATA_FILE_PATHS:
        try:
            with open(raw_data_file, 'r', encoding='utf-8', errors='ignore') as f:
                raw_lines = f.readlines()

            if not raw_lines:
                global_logger.warning(f"Warning: {raw_data_file} is empty. Skipping.")
                continue

            header_line = raw_lines[0]
            data_lines = raw_lines[1:]
            filtered_data_lines = []

            for data_line in data_lines:
                stripped_line = data_line.rstrip('\r\n')
                keep_this_line = True
                for ending in PROBLEMATIC_ENDINGS:
                    if stripped_line.endswith(ending):
                        keep_this_line = False
                        break
                if keep_this_line:
                    filtered_data_lines.append(data_line)

            processed_content = [header_line] + filtered_data_lines

            if len(processed_content) <= 1: # Only header or no valid data lines
                global_logger.warning(f"Warning: No valid CSV-like data lines found in {raw_data_file} after initial filtering. Skipping.")
                continue

            data_io = io.StringIO("".join(processed_content))
            df = pd.read_csv(data_io, on_bad_lines='skip')
            all_dfs.append(df)
            global_logger.info(f"\tSuccessfully loaded and pre-filtered {raw_data_file}")
            global_logger.info(f"  (Original lines: {len(raw_lines)}, Filtered lines for CSV parsing: {len(processed_content)})")

        except FileNotFoundError:
            global_logger.warning(f"Warning: Log file not found at {raw_data_file}. Skipping.")
        except Exception as e:
            global_logger.error(f"Error processing CSV file {raw_data_file}: {e}. Skipping.")

    if not all_dfs:
        global_logger.warning("No data loaded from any specified files.")
        return pd.DataFrame()

    df = pd.concat(all_dfs, ignore_index=True)
    global_logger.info(f"\tLoaded {len(df)} raw log entries from all files.")
    return df

def _remove_first_field_errors(df):
    """Removes rows where the first field contains 'Failed' or 'Error'."""
    if df.empty or len(df.columns) == 0:
        global_logger.warning("DataFrame is empty or has no columns. Skipping 'Failed'/'Error' row removal.")
        return df

    initial_rows = len(df)
    first_column_name = df.columns[0]
    df_filtered = df[~df[first_column_name].astype(str).str.contains("Failed|Error", case=False, na=False)]
    rows_removed = initial_rows - len(df_filtered)
    if rows_removed > 0:
        global_logger.info(f"\tRemoved {rows_removed} rows where the first field ('{first_column_name}') contained 'Failed' or 'Error'.")
    else:
        global_logger.info(f"\tNo rows removed based on 'Failed'/'Error' in the first field ('{first_column_name}').")
    return df_filtered

def _remove_exact_duplicates(df):
    """Removes exact duplicate rows from the DataFrame."""
    initial_rows = len(df)
    df.drop_duplicates(inplace=True)
    rows_removed = initial_rows - len(df)
    if rows_removed > 0:
        global_logger.info(f"\tRemoved {rows_removed} duplicate rows.")
    else:
        global_logger.info("\tNo duplicate rows found.")
    return df

def _filter_label(df):
    """
    Removes rows where LABEL is missing or not 'benign'/'malicious'.
    """
    if LABEL not in df.columns:
        global_logger.warning(f"'{LABEL}' column not found for cleaning. Skipping label-based row removal.")
        return df

    initial_rows = len(df)
    df = df[df[LABEL].isin(LABEL_VALUES)]
    rows_removed = initial_rows - len(df)
    if rows_removed > 0:
        global_logger.info(f"\tRemoved {rows_removed} rows with invalid '{LABEL}'.")
    else:
        global_logger.info(f"\tNo rows removed based on '{LABEL}' validity.")
    return df

def _handle_missing_critical_features(df):
    """
    Removes rows with missing values in specified critical feature columns.
    """
    
    existing_critical_columns = [col for col in CRITICAL_FEATURES if col in df.columns]

    if not existing_critical_columns:
        global_logger.warning("\tNo critical feature columns found for missing value cleaning.")
        return df

    initial_rows = len(df)
    df.dropna(subset=existing_critical_columns, inplace=True)
    rows_removed = initial_rows - len(df)
    if rows_removed > 0:
        global_logger.info(f"\tRemoved {rows_removed} rows with missing critical features: {', '.join(existing_critical_columns)}.")
    else:
        global_logger.info("\tNo rows removed due to missing critical features.")
    return df

def _uppercase_columns(df):
    """Standardizes some to uppercase."""

    for col in COLUMNS_TO_UPPERCASE:
        if col not in df.columns:
            global_logger.warning(f"\t'{col}' column not found for standardization.")
            return df

        df[col] = df[col].astype(str).str.upper()
        global_logger.info(f"\tStandardized '{col}' to uppercase.")
    return df

def _drop_specified_columns(df):
    """Removes specific columns from the DataFrame if they exist."""
    if not COLUMNS_TO_DROP:
        global_logger.info("No columns specified to drop.")
        return df

    cols_to_drop_existing = [col for col in COLUMNS_TO_DROP if col in df.columns]
    if cols_to_drop_existing:
        df.drop(columns=cols_to_drop_existing, inplace=True)
        global_logger.info(f"\tRemoved columns: {', '.join(cols_to_drop_existing)}")
    else:
        global_logger.info("\tNo specified columns to drop were found in the DataFrame.")
    return df

def load_and_clean_data():
    """
    Orchestrates loading data (either raw and cleaning or from a cleaned file).

    Args:
       
    Returns:
        pandas.DataFrame: A cleaned DataFrame of log entries.
    """
    
    df = pd.DataFrame() 
    evaluation_logger.info("--- Data Cleaning ---")

    if PERFORM_DATA_CLEANING:
        # Load lines from raw csv files, and perform cleaning
        global_logger.info("\tLoading data from Raw Files")
        df = _load_raw_log_files()

        if df.empty:
            evaluation_logger.error("No data to clean after initial loading. Returning empty DataFrame.")
            return pd.DataFrame()
        

        global_logger.info("\tApplying Data Cleaning Steps")
        initial_rows_before_cleaning = len(df)

        df = _remove_first_field_errors(df)
        df = _remove_exact_duplicates(df)
        df = _filter_label(df)
        df = _handle_missing_critical_features(df)
        df = _uppercase_columns(df)
        df = _drop_specified_columns(df)

        evaluation_logger.info(f"\tTotal rows removed during cleaning: {initial_rows_before_cleaning - len(df)}")
        evaluation_logger.info(f"\tRemaining {len(df)} entries after cleaning.")

        output_file_path_with_timestamp = _add_timestamp_to_filename(CLEANED_DATA_OUTPUT_PATH)
        # Save the newly cleaned data
        if not df.empty:
            try:
                df.to_csv(output_file_path_with_timestamp, index=False)
                evaluation_logger.info(f"\tCleaned data saved to {output_file_path_with_timestamp}")
            except Exception as e:
                evaluation_logger.error(f"Error saving cleaned data to CSV: {e}")
        else:
            evaluation_logger.warning("Cleaned DataFrame is empty. Not saving an empty file.")

        
    else:
        global_logger.info(f"\tLoading already cleaned data from {CLEANED_DATA_PATH}")
        try:
            # Load the single cleaned CSV file directly
            df = pd.read_csv(CLEANED_DATA_PATH)
            evaluation_logger.info(f"\tSuccessfully loaded {len(df)} entries from {CLEANED_DATA_PATH}")
        except FileNotFoundError:
            evaluation_logger.error(f"Error: Cleaned data file not found at {CLEANED_DATA_PATH}. Set 'perform_data_cleaning=True' if you want to clean from raw data.")
            return pd.DataFrame()
        except Exception as e:
            evaluation_logger.error(f"Error loading cleaned data from {CLEANED_DATA_PATH}: {e}")
            return pd.DataFrame()
   
        
    return df