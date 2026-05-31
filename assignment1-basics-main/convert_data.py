"""
Convert parquet dataset files to txt format.

Usage:
    python convert_data.py --data_dir ../autodl-tmp/data/tiny/data/ --output_dir ./data/
"""

import os
import glob
import argparse
import pandas as pd


def load_parquet_files(data_dir: str, pattern: str) -> pd.DataFrame:
    """Load all parquet files matching the pattern."""
    parquet_files = sorted(glob.glob(os.path.join(data_dir, pattern)))
    if not parquet_files:
        raise FileNotFoundError(f"No parquet files found matching pattern: {pattern}")
    
    print(f"Found {len(parquet_files)} files matching pattern '{pattern}':")
    for f in parquet_files:
        print(f"  - {os.path.basename(f)}")
    
    dfs = []
    for f in parquet_files:
        print(f"Loading {os.path.basename(f)}...")
        df = pd.read_parquet(f)
        dfs.append(df)
    
    combined_df = pd.concat(dfs, ignore_index=True)
    print(f"Combined shape: {combined_df.shape}")
    return combined_df


def extract_text_from_df(df: pd.DataFrame, text_column: str = None) -> list:
    """Extract text from dataframe, auto-detecting text column if not specified."""
    if text_column is not None:
        if text_column not in df.columns:
            raise ValueError(f"Column '{text_column}' not found. Available columns: {list(df.columns)}")
        return df[text_column].tolist()
    
    # Auto-detect text column
    text_candidates = ['text', 'content', 'passage', 'document', 'input', 'sentence']
    for col in text_candidates:
        if col in df.columns:
            print(f"Auto-detected text column: '{col}'")
            return df[col].tolist()
    
    # If no text column found, use first string column
    for col in df.columns:
        if df[col].dtype == 'object':
            print(f"Using first string column: '{col}'")
            return df[col].tolist()
    
    raise ValueError(f"No suitable text column found. Available columns: {list(df.columns)}")


def save_to_txt(texts: list, output_path: str):
    """Save texts to a single txt file, one text per line."""
    with open(output_path, 'w', encoding='utf-8') as f:
        for text in texts:
            # Convert to string and clean up
            text_str = str(text).strip()
            # Replace newlines with space to keep one document per line
            text_str = text_str.replace('\n', ' ').replace('\r', ' ')
            if text_str:
                f.write(text_str + '\n')
    print(f"Saved {len(texts)} texts to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Convert parquet dataset to txt format")
    parser.add_argument("--data_dir", type=str, default="../autodl-tmp/data/tiny/data/",
                        help="Directory containing parquet files")
    parser.add_argument("--output_dir", type=str, default="../autodl-tmp/data/tiny/data/",
                        help="Directory to save output txt files")
    parser.add_argument("--train_pattern", type=str, default="train-*.parquet",
                        help="Pattern for training parquet files")
    parser.add_argument("--valid_pattern", type=str, default="valid-*.parquet",
                        help="Pattern for validation parquet files")
    parser.add_argument("--text_column", type=str, default=None,
                        help="Name of text column (auto-detected if not specified)")
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Check data directory exists
    if not os.path.exists(args.data_dir):
        raise FileNotFoundError(f"Data directory not found: {args.data_dir}")
    
    print("=" * 50)
    print("Loading training data...")
    print("=" * 50)
    train_df = load_parquet_files(args.data_dir, args.train_pattern)
    print(f"Columns: {list(train_df.columns)}")
    train_texts = extract_text_from_df(train_df, args.text_column)
    
    print("\n" + "=" * 50)
    print("Loading validation/test data...")
    print("=" * 50)
    valid_df = load_parquet_files(args.data_dir, args.valid_pattern)
    print(f"Columns: {list(valid_df.columns)}")
    valid_texts = extract_text_from_df(valid_df, args.text_column)
    
    print("\n" + "=" * 50)
    print("Saving to txt files...")
    print("=" * 50)
    train_output = os.path.join(args.output_dir, "train.txt")
    valid_output = os.path.join(args.output_dir, "test.txt")
    
    save_to_txt(train_texts, train_output)
    save_to_txt(valid_texts, valid_output)
    
    print("\n" + "=" * 50)
    print("Conversion complete!")
    print("=" * 50)
    print(f"Training data: {train_output} ({len(train_texts)} samples)")
    print(f"Test data: {valid_output} ({len(valid_texts)} samples)")


if __name__ == "__main__":
    main()