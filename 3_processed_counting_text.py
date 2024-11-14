import pandas as pd
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from collections import Counter
import re
import time
from datetime import datetime
from tqdm import tqdm

def print_timestamp(message):
    """Print message with timestamp"""
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{current_time}] {message}")

def tokenize(text):
    """Tokenize text with error handling"""
    try:
        if isinstance(text, str):
            return re.findall(r'\w+', text)
        else:
            print_timestamp(f"Warning: Non-string input detected: {type(text)}")
            return []
    except Exception as e:
        print_timestamp(f"Error in tokenization: {str(e)}")
        return []

def main():
    try:
        start_time = time.time()
        print_timestamp("Starting word frequency analysis...")

        # Load preprocessed data
        print_timestamp("Loading preprocessed data...")
        try:
            input_path = 'Output/2_preprocessed_text_gopay_output.csv'
            df = pd.read_csv(input_path)
            print_timestamp(f"Successfully loaded {len(df)} rows from {input_path}")
        except FileNotFoundError:
            print_timestamp("Error: Input CSV file not found!")
            return
        except pd.errors.EmptyDataError:
            print_timestamp("Error: Input CSV file is empty!")
            return
        except Exception as e:
            print_timestamp(f"Error loading CSV: {str(e)}")
            return

        # Initialize Sastrawi stemmer
        print_timestamp("Initializing Sastrawi stemmer...")
        try:
            factory = StemmerFactory()
            stemmer = factory.create_stemmer()
            print_timestamp("Stemmer initialized successfully")
        except Exception as e:
            print_timestamp(f"Error initializing stemmer: {str(e)}")
            return

        # Process texts and count words
        print_timestamp("Starting word tokenization and counting...")
        all_words = []
        
        # Using tqdm for progress bar
        for index, row in tqdm(df.iterrows(), total=len(df), desc="Processing texts"):
            try:
                if index % 1000 == 0 and index > 0:
                    print_timestamp(f"Processed {index} texts...")
                
                if 'preprocessed_text' not in row:
                    print_timestamp("Error: 'preprocessed_text' column not found in CSV!")
                    return
                
                tokens = tokenize(row['preprocessed_text'])
                all_words.extend(tokens)
                
            except Exception as e:
                print_timestamp(f"Error processing row {index}: {str(e)}")
                continue

        print_timestamp(f"Tokenization complete. Total words collected: {len(all_words)}")

        # Count word frequencies
        print_timestamp("Counting word frequencies...")
        word_freq = Counter(all_words)
        most_common_words = word_freq.most_common()
        print_timestamp(f"Found {len(word_freq)} unique words")

        # Create DataFrame
        print_timestamp("Creating frequency DataFrame...")
        df_word_freq = pd.DataFrame(most_common_words, columns=['Word', 'Frequency'])

        # Save results
        print_timestamp("Saving word frequencies...")
        try:
            output_path = 'Output/3_processed_counting_threads_V2_text.csv'
            df_word_freq.to_csv(output_path, index=False)
            print_timestamp(f"Results saved successfully to {output_path}")
        except Exception as e:
            print_timestamp(f"Error saving results: {str(e)}")
            return

        # Print summary
        end_time = time.time()
        duration = round(end_time - start_time, 2)
        print_timestamp("Analysis complete!")
        print_timestamp(f"Total execution time: {duration} seconds")
        print_timestamp(f"Total words processed: {len(all_words)}")
        print_timestamp(f"Unique words found: {len(word_freq)}")
        
        # Display top 10 most frequent words
        print_timestamp("\nTop 10 most frequent words:")
        for word, freq in word_freq.most_common(10):
            print(f"  {word}: {freq}")

    except Exception as e:
        print_timestamp(f"Critical error in main process: {str(e)}")

if __name__ == "__main__":
    main()