import pandas as pd
import os
import io # Import io for StringIO

from config import (
    LOG_FILE_PATHS,  CLEANED_DATA_OUTPUT_PATH,
    PERFORM_DATA_CLEANING, COLUMNS_TO_DROP, PROBLEMATIC_ENDINGS,

    )

def _load_raw_log_files(logger):
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
    logger.info("\dLoading raw log files and applying initial line filtering...")
    for log_file in LOG_FILE_PATHS:
        try:
            with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
                raw_lines = f.readlines()

            if not raw_lines:
                logger.warning(f"Warning: {log_file} is empty. Skipping.")
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
                logger.warning(f"Warning: No valid CSV-like data lines found in {log_file} after initial filtering. Skipping.")
                continue

            data_io = io.StringIO("".join(processed_content))
            df = pd.read_csv(data_io, on_bad_lines='skip')
            all_dfs.append(df)
            logger.info(f"\tSuccessfully loaded and pre-filtered {log_file}")
            logger.info(f"  (Original lines: {len(raw_lines)}, Filtered lines for CSV parsing: {len(processed_content)})")

        except FileNotFoundError:
            logger.warning(f"Warning: Log file not found at {log_file}. Skipping.")
        except Exception as e:
            logger.error(f"Error processing CSV file {log_file}: {e}. Skipping.")

    if not all_dfs:
        logger.warning("No data loaded from any specified files.")
        return pd.DataFrame()

    df = pd.concat(all_dfs, ignore_index=True)
    logger.info(f"\tLoaded {len(df)} raw log entries from all files.")
    return df

def _remove_first_field_errors(df, logger):
    """Removes rows where the first field contains 'Failed' or 'Error'."""
    if df.empty or len(df.columns) == 0:
        logger.warning("DataFrame is empty or has no columns. Skipping 'Failed'/'Error' row removal.")
        return df

    initial_rows = len(df)
    first_column_name = df.columns[0]
    df_filtered = df[~df[first_column_name].astype(str).str.contains("Failed|Error", case=False, na=False)]
    rows_removed = initial_rows - len(df_filtered)
    if rows_removed > 0:
        logger.info(f"\tRemoved {rows_removed} rows where the first field ('{first_column_name}') contained 'Failed' or 'Error'.")
    else:
        logger.info(f"\tNo rows removed based on 'Failed'/'Error' in the first field ('{first_column_name}').")
    return df_filtered

def _remove_exact_duplicates(df, logger):
    """Removes exact duplicate rows from the DataFrame."""
    initial_rows = len(df)
    df.drop_duplicates(inplace=True)
    rows_removed = initial_rows - len(df)
    if rows_removed > 0:
        logger.info(f"\tRemoved {rows_removed} duplicate rows.")
    else:
        logger.info("\tNo duplicate rows found.")
    return df

def _filter_ai_verdict_label(df, logger):
    """
    Removes rows where 'AIVerdictLabel' is missing or not 'benign'/'malicious'.
    """
    if 'AIVerdictLabel' not in df.columns:
        logger.warning("'AIVerdictLabel' column not found for cleaning. Skipping label-based row removal.")
        return df

    initial_rows = len(df)
    df = df[df['AIVerdictLabel'].isin(['benign', 'malicious'])]
    rows_removed = initial_rows - len(df)
    if rows_removed > 0:
        logger.info(f"\tRemoved {rows_removed} rows with invalid 'AIVerdictLabel'.")
    else:
        logger.info("\tNo rows removed based on 'AIVerdictLabel' validity.")
    return df

def _handle_missing_critical_features(df, logger):
    """
    Removes rows with missing values in specified critical feature columns.
    """
    critical_columns = ['RequestURI', 'RequestMethod']
    existing_critical_columns = [col for col in critical_columns if col in df.columns]

    if not existing_critical_columns:
        logger.warning("\tNo critical feature columns found for missing value cleaning.")
        return df

    initial_rows = len(df)
    df.dropna(subset=existing_critical_columns, inplace=True)
    rows_removed = initial_rows - len(df)
    if rows_removed > 0:
        logger.info(f"\tRemoved {rows_removed} rows with missing critical features: {', '.join(existing_critical_columns)}.")
    else:
        logger.info("\tNo rows removed due to missing critical features.")
    return df

def _standardize_request_method(df, logger):
    """Standardizes 'RequestMethod' to uppercase."""
    if 'RequestMethod' not in df.columns:
        logger.warning("\t'RequestMethod' column not found for standardization.")
        return df

    df['RequestMethod'] = df['RequestMethod'].astype(str).str.upper()
    logger.info("\tStandardized 'RequestMethod' to uppercase.")
    return df

def _drop_specified_columns(df, logger):
    """Removes specific columns from the DataFrame if they exist."""
    if not COLUMNS_TO_DROP:
        logger.info("No columns specified to drop.")
        return df

    cols_to_drop_existing = [col for col in COLUMNS_TO_DROP if col in df.columns]
    if cols_to_drop_existing:
        df.drop(columns=cols_to_drop_existing, inplace=True)
        logger.info(f"\tRemoved columns: {', '.join(cols_to_drop_existing)}")
    else:
        logger.info("\tNo specified columns to drop were found in the DataFrame.")
    return df



def load_and_clean_data(logger):
    """
    Orchestrates loading data (either raw and cleaning or from a cleaned file).

    Args:
        log_file_paths (list): A list of file paths to the raw CSV log files.
        cleaned_data_output_path (str): Path to save/load the cleaned data CSV.
        perform_data_cleaning (bool): If True, loads raw and cleans; otherwise, loads from cleaned_data_output_path.
        logger (logging.Logger): The logger instance for logging messages.
        columns_to_drop (list): A list of column names to be removed from the DataFrame.
        problematic_endings (list): A list of string endings to filter out problematic lines during raw loading.

    Returns:
        pandas.DataFrame: A cleaned DataFrame of log entries.
    """
    
    df = pd.DataFrame() 

    if PERFORM_DATA_CLEANING:
        
        logger.info("--- Data Cleaning ---")
        # Load lines from raw csv files, and perform cleaning
        logger.info("\tStarting Data Loading and Cleaning from Raw Files")
        df = _load_raw_log_files(logger)

        if df.empty:
            logger.warning("No data to clean after initial loading. Returning empty DataFrame.")
            return pd.DataFrame()
        

        logger.info("\tApplying Data Cleaning Steps")
        initial_rows_before_cleaning = len(df)

        df = _remove_first_field_errors(df, logger)
        df = _remove_exact_duplicates(df, logger)
        df = _filter_ai_verdict_label(df, logger)
        df = _handle_missing_critical_features(df, logger)
        df = _standardize_request_method(df, logger)
        df = _drop_specified_columns(df, logger)

        logger.info(f"\tTotal rows removed during cleaning: {initial_rows_before_cleaning - len(df)}")
        logger.info(f"\tRemaining {len(df)} entries after cleaning.")

        # Save the newly cleaned data
        if not df.empty:
            try:
                df.to_csv(CLEANED_DATA_OUTPUT_PATH, index=False)
                logger.info(f"\tCleaned data saved to {CLEANED_DATA_OUTPUT_PATH}")
            except Exception as e:
                logger.error(f"Error saving cleaned data to CSV: {e}")
        else:
            logger.warning("Cleaned DataFrame is empty. Not saving an empty file.")

        
    else:
        logger.info(f"\tLoading already cleaned data from {CLEANED_DATA_OUTPUT_PATH}")
        try:
            # Load the single cleaned CSV file directly
            df = pd.read_csv(CLEANED_DATA_OUTPUT_PATH)
            logger.info(f"\tSuccessfully loaded {len(df)} entries from {CLEANED_DATA_OUTPUT_PATH}")
        except FileNotFoundError:
            logger.error(f"Error: Cleaned data file not found at {CLEANED_DATA_OUTPUT_PATH}. Set 'perform_data_cleaning=True' if you want to clean from raw data.")
            return pd.DataFrame()
        except Exception as e:
            logger.error(f"Error loading cleaned data from {CLEANED_DATA_OUTPUT_PATH}: {e}")
            return pd.DataFrame()
   
        
    return df