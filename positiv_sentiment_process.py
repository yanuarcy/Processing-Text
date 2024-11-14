import pandas as pd
import re
from datetime import datetime
from tqdm import tqdm

def print_timestamp(message):
    """Print message with timestamp"""
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{current_time}] {message}")

# Daftar kata-kata positif dalam Bahasa Indonesia
POSITIVE_WORDS = {
    # Kata sifat positif umum
    'bagus', 'baik', 'mantap', 'keren', 'hebat', 'sangat', 'mudah', 'praktis',
    'berguna', 'bermanfaat', 'membantu', 'memudahkan', 'memuaskan', 'puas',
    'senang', 'suka', 'nyaman', 'lancar', 'cepat', 'responsif', 'ramah',
    'informatif', 'lengkap', 'jelas', 'teratur', 'rapi', 'efektif', 'efisien',
    'optimal', 'profesional', 'kompeten', 'handal', 'terpercaya', 'aman',
    
    # Kata apresiasi
    'terima kasih', 'makasih', 'thank', 'thanks', 'mantul', 'mantab', 'jos',
    'recommended', 'rekomendasi', 'rekomen', 'worth', 'berguna', 'berfaedah',
    
    # Kata superlatif
    'terbaik', 'sangat bagus', 'sangat baik', 'sangat membantu', 'super',
    'luar biasa', 'sempurna', 'excellent', 'outstanding', 'perfect',
    
    # Kata pendukung positif
    'selalu', 'terus', 'konsisten', 'update', 'aktif', 'responsive',
    'inovatif', 'kreatif', 'modern', 'canggih', 'update', 'terkini',
    
    # Ekspresi positif
    'wow', 'keren', 'oke', 'ok', 'yes', 'mantul', 'joss', 'nice',
    'good', 'great', 'awesome', 'cool',
    
    # Kata apresiasi layanan
    'ramah', 'cepat', 'tepat', 'akurat', 'teliti', 'detail', 'lengkap',
    'informatif', 'komunikatif', 'kooperatif', 'profesional',
    
    # Kata kepuasan
    'puas', 'senang', 'suka', 'enjoy', 'nyaman', 'tenang', 'aman',
    'percaya', 'yakin', 'mantap',
}

def extract_positive_sentences(text):
    """
    Mengekstrak kalimat yang mengandung kata-kata positif
    """
    if pd.isna(text):
        return None
        
    sentences = re.split(r'[.!?]', str(text))
    positive_sentences = []
    
    for sentence in sentences:
        sentence = sentence.strip().lower()
        words = set(sentence.split())
        
        # Cek apakah ada kata positif dalam kalimat
        if words & POSITIVE_WORDS:
            positive_sentences.append(sentence)
            
    return positive_sentences if positive_sentences else None

def calculate_positive_score(sentences):
    """
    Menghitung skor positif berdasarkan jumlah kata positif
    """
    if not sentences:
        return 0
        
    total_positive_words = 0
    for sentence in sentences:
        words = set(sentence.split())
        positive_count = len(words & POSITIVE_WORDS)
        total_positive_words += positive_count
        
    return total_positive_words

def main():
    try:
        print_timestamp("Starting positive sentiment filtering...")
        
        # Load preprocessed data
        print_timestamp("Loading preprocessed data...")
        df = pd.read_csv('Output/2_preprocessed_text_gopay_output.csv')
        print_timestamp(f"Loaded {len(df)} preprocessed texts")
        
        # Extract positive sentences
        print_timestamp("Extracting positive sentences...")
        tqdm.pandas(desc="Processing")
        df['positive_sentences'] = df['preprocessed_text'].progress_apply(extract_positive_sentences)
        
        # Calculate positive scores
        print_timestamp("Calculating positive scores...")
        df['positive_score'] = df['positive_sentences'].apply(calculate_positive_score)
        
        # Filter significant positive content (score > 0)
        positive_df = df[df['positive_score'] > 0].copy()
        
        # Sort by positive score
        positive_df = positive_df.sort_values('positive_score', ascending=False)
        
        # Create output DataFrame with relevant columns
        output_df = pd.DataFrame({
            'text': positive_df['preprocessed_text'],
            'positive_sentences': positive_df['positive_sentences'].apply(lambda x: ' | '.join(x) if x else ''),
            'positive_score': positive_df['positive_score']
        })
        
        # Save results
        print_timestamp("Saving positive sentiment results...")
        output_df.to_csv('Output/3_positive_sentiment_output.csv', index=False)
        
        # Print summary
        print_timestamp(f"Total texts processed: {len(df)}")
        print_timestamp(f"Positive texts found: {len(positive_df)}")
        print_timestamp(f"Average positive score: {positive_df['positive_score'].mean():.2f}")
        
        # Display top positive examples
        print_timestamp("\nTop 5 Most Positive Reviews:")
        for idx, row in output_df.head().iterrows():
            print(f"\nScore: {row['positive_score']}")
            print(f"Positive Sentences: {row['positive_sentences']}")
            print("-" * 50)
        
    except Exception as e:
        print_timestamp(f"Error in main process: {str(e)}")

if __name__ == "__main__":
    main()