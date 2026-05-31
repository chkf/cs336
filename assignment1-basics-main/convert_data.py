"""
Convert parquet dataset files to txt format.

Dependency:
    pip install pyarrow

Usage:
    python convert_data.py --data_dir ../autodl-tmp/data/tiny/data/ --output_dir ./data/
"""

import os
import glob
import argparse

try:
    import pyarrow.parquet as pq
except ImportError:
    raise ImportError("Please install pyarrow first: pip install pyarrow")


def load_parquet_files(data_dir: str, pattern: str) -> list:
    """Load all parquet files matching the pattern and return text list."""
    parquet_files = sorted(glob.glob(os.path.join(data_dir, pattern)))
    if not parquet_files:
        raise FileNotFoundError(f"No parquet files found matching pattern: {pattern}")
    
    print(f"Found {len(parquet_files)} files matching pattern '{pattern}':")
    for f in parquet_files:
        print(f"  - {os.path.basename(f)}")
    
    all_texts = []
    total_rows = 0
    for f in parquet_files:
        print(f"Loading {os.path.basename(f)}...")
        table = pq.read_table(f)
        df = table.to_pandas()
        
        if total_rows == 0:
            print(f"Columns: {list(df.columns)}")
        
        # Auto-detect text column
        text_col = detect_text_column(df)
        texts = df[text_col].tolist()
        all_texts.extend(texts)
        total_rows += len(df)
    
    print(f"Total loaded: {total_rows} rows")
    return all_texts


def detect_text_column(df) -> str:
    """Auto-detect text column from dataframe."""
    text_candidates = ['text', 'content', 'passage', 'document', 'input', 'sentence']
    for col in text_candidates:
        if col in df.columns:
            return col
    
    # If no text column found, use first string column
    for col in df.columns:
        if df[col].dtype == 'object':
            return col
    
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
    parser.add_argument("--data_dir", type=str, default="../../autodl-tmp/data/tiny/data/",
                        help="Directory containing parquet files")
    parser.add_argument("--output_dir", type=str, default="../../autodl-tmp/data/tiny/data/",
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
    train_texts = load_parquet_files(args.data_dir, args.train_pattern)
    
    print("\n" + "=" * 50)
    print("Loading validation/test data...")
    print("=" * 50)
    valid_texts = load_parquet_files(args.data_dir, args.valid_pattern)
    
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