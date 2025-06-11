import pandas as pd
import os
import io # Import io for StringIO

def load_and_clean_data(log_file_paths, columns_to_drop, problematic_endings):
    """
    Loads and merges CSV log entries from the specified list of files,
    then performs initial data cleaning.

    Args:
        log_file_paths (list): A list of file paths to the raw CSV log files.
        columns_to_drop (list): A list of column names to be removed from the DataFrame.
        problematic_endings (list): A list of string endings to filter out problematic lines.

    Returns:
        pandas.DataFrame: A cleaned and merged DataFrame of log entries.
    """
    all_dfs = []
    for log_file in log_file_paths:
        try:
            # Read file content as raw lines first to pre-filter
            with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
                raw_lines = f.readlines()

            if not raw_lines:
                print(f"Warning: {log_file} is empty. Skipping.")
                continue

            # Identify the header line (usually the first line)
            header_line = raw_lines[0]
            data_lines = raw_lines[1:]
            filtered_data_lines = []

            for data_line in data_lines:
                stripped_line = data_line.rstrip('\r\n')

                # Check if the stripped line ends with any of the problematic endings
                keep_this_line = True
                for ending in problematic_endings:
                    if stripped_line.endswith(ending):
                        keep_this_line = False
                        break

                if keep_this_line:
                    # Re-add the original newline character before storing
                    filtered_data_lines.append(data_line)

            # Reconstruct the content for pandas, ensuring the header is always included
            processed_content = [header_line] + filtered_data_lines

            if len(processed_content) <= 1: # Only header or no data lines
                print(f"Warning: No valid CSV-like data lines found in {log_file} after initial filtering. Skipping.")
                continue

            # Use io.StringIO to treat the filtered lines as a file-like object for pandas
            data_io = io.StringIO("".join(processed_content))

            # Use on_bad_lines='skip' for any remaining parsing errors within the filtered lines
            df = pd.read_csv(data_io, on_bad_lines='skip')
            all_dfs.append(df)
            print(f"Successfully loaded and pre-filtered {log_file}")
            print(f"  (Original lines: {len(raw_lines)}, Filtered lines for CSV parsing: {len(processed_content)})")

        except FileNotFoundError:
            print(f"Warning: Log file not found at {log_file}. Skipping.")
        except Exception as e:
            print(f"Error processing CSV file {log_file}: {e}. Skipping.")

    if not all_dfs:
        print("No data loaded from any specified files.")
        return pd.DataFrame()

    df = pd.concat(all_dfs, ignore_index=True)
    print(f"Loaded {len(df)} raw log entries from all files.")

    # --- Data Cleaning Steps ---
    print("\nPerforming data cleaning...")
    initial_rows = len(df)

    # 1. Remove rows where the first field contains "Failed" or "Error"
    if not df.empty and len(df.columns) > 0:
        first_column_name = df.columns[0]
        # Ensure the column is treated as string to use .str.contains
        df_filtered = df[~df[first_column_name].astype(str).str.contains("Failed|Error", case=False, na=False)]
        rows_removed_first_field = initial_rows - len(df_filtered)
        if rows_removed_first_field > 0:
            print(f"Removed {rows_removed_first_field} rows where the first field ('{first_column_name}') contained 'Failed' or 'Error'.")
            df = df_filtered
        else:
            print(f"No rows removed based on 'Failed'/'Error' in the first field ('{first_column_name}').")
    else:
        print("Warning: DataFrame is empty or has no columns. Skipping 'Failed'/'Error' row removal.")
    initial_rows = len(df) # Update initial_rows for next step


    # 2. Remove exact duplicate rows
    df.drop_duplicates(inplace=True)
    print(f"Removed {initial_rows - len(df)} duplicate rows.")
    initial_rows = len(df) # Update initial_rows for next step

    # 3. Remove rows where 'AIVerdictLabel' is missing or not 'benign'/'malicious'
    if 'AIVerdictLabel' in df.columns:
        df = df[df['AIVerdictLabel'].isin(['benign', 'malicious'])]
        print(f"Removed {initial_rows - len(df)} rows with invalid AIVerdictLabel.")
    else:
        print("Warning: 'AIVerdictLabel' column not found for cleaning. Skipping label-based row removal.")
    initial_rows = len(df)

    # 4. Handle missing values in critical feature columns
    critical_columns = ['RequestURI', 'RequestMethod']
    existing_critical_columns = [col for col in critical_columns if col in df.columns]

    if existing_critical_columns:
        df.dropna(subset=existing_critical_columns, inplace=True)
        print(f"Removed {initial_rows - len(df)} rows with missing critical features.")
    else:
        print("Warning: No critical feature columns found for missing value cleaning.")

    # 5. Standardize 'RequestMethod' to uppercase
    if 'RequestMethod' in df.columns:
        df['RequestMethod'] = df['RequestMethod'].astype(str).str.upper()
        print("Standardized 'RequestMethod' to uppercase.")
    else:
        print("Warning: 'RequestMethod' column not found for standardization.")

    # 6. Remove specific columns if provided
    if columns_to_drop:
        cols_to_drop_existing = [col for col in columns_to_drop if col in df.columns]
        if cols_to_drop_existing:
            df.drop(columns=cols_to_drop_existing, inplace=True)
            print(f"Removed columns: {', '.join(cols_to_drop_existing)}")
        else:
            print("No specified columns to drop were found in the DataFrame.")

    print(f"Remaining {len(df)} entries after cleaning.")
    return df
