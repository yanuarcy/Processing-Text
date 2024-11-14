import pickle
import pandas as pd
import numpy as np
import re
import string
import json
import time
from datetime import datetime
from tqdm import tqdm
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt
import seaborn as sns
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('vader_lexicon')
# # Download required NLTK resources
# print("Downloading required NLTK resources...")
# nltk.download('punkt')
# nltk.download('vader_lexicon')
# print("NLTK resources downloaded successfully")

# from nltk.tokenize import word_tokenize
# from nltk.sentiment.vader import SentimentIntensityAnalyzer

def create_comparison_table(models_results):
    """
    Create a comparison table of different models
    """
    comparison_data = []
    for name, results in models_results.items():
        comparison_data.append({
            'Model': name,
            'Accuracy': results['accuracy'],
            'Macro Avg F1': results['macro_avg_f1'],
            'Weighted Avg F1': results['weighted_avg_f1']
        })
    return pd.DataFrame(comparison_data)

def experiment_with_tfidf(X, y):
    """
    Experiment with different TF-IDF parameters
    """
    experiments = [
        {
            'name': 'Default TF-IDF',
            'params': {'max_features': 5000}
        },
        {
            'name': 'TF-IDF with Bigrams',
            'params': {'max_features': 5000, 'ngram_range': (1, 2)}
        },
        {
            'name': 'TF-IDF with Custom Parameters',
            'params': {
                'max_features': 10000,
                'min_df': 2,
                'max_df': 0.9,
                'ngram_range': (1, 3)
            }
        }
    ]
    
    results = []
    for exp in experiments:
        print_timestamp(f"\nExperiment: {exp['name']}")
        tfidf = TfidfVectorizer(**exp['params'])
        X_transformed = tfidf.fit_transform(X)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_transformed, y, test_size=0.25, random_state=42, stratify=y
        )
        
        # Try with best performing model (Random Forest in this case)
        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        rf.fit(X_train, y_train)
        y_pred = rf.predict(X_test)
        
        results.append({
            'Experiment': exp['name'],
            'Accuracy': accuracy_score(y_test, y_pred),
            'Report': classification_report(y_test, y_pred,
                                         labels=[0, 1, 2],
                                         target_names=["Negative", "Neutral", "Positive"],
                                         zero_division=0)
        })
    
    return results

def hyperparameter_tuning(X_train, X_test, y_train, y_test):
    """
    Perform hyperparameter tuning on the best model
    """
    # Random Forest parameters
    rf_params = {
        'n_estimators': [100, 200, 300],
        'max_depth': [10, 20, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }
    
    rf = RandomForestClassifier(random_state=42)
    grid_search = GridSearchCV(rf, rf_params, cv=5, scoring='accuracy', n_jobs=-1)
    grid_search.fit(X_train, y_train)
    
    print_timestamp("\nBest parameters found:")
    print_timestamp(grid_search.best_params_)
    
    y_pred = grid_search.predict(X_test)
    return {
        'best_params': grid_search.best_params_,
        'accuracy': accuracy_score(y_test, y_pred),
        'report': classification_report(y_test, y_pred,
                                     labels=[0, 1, 2],
                                     target_names=["Negative", "Neutral", "Positive"],
                                     zero_division=0)
    }

def download_nltk_resources():
    """
    Download semua resource NLTK yang diperlukan
    """
    try:
        print("Downloading NLTK resources...")
        resources = [
            'punkt',
            'punkt_tab',
            'vader_lexicon',
            'averaged_perceptron_tagger',
            'universal_tagset'
        ]
        
        for resource in resources:
            try:
                nltk.download(resource, quiet=True)
                print(f"Successfully downloaded {resource}")
            except Exception as e:
                print(f"Error downloading {resource}: {str(e)}")
        
        print("NLTK resource download completed")
    except Exception as e:
        print(f"Error in downloading NLTK resources: {str(e)}")
        
# Download NLTK resources before importing modules that use them
download_nltk_resources()

# Now import NLTK modules
from nltk.tokenize import word_tokenize
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# Fungsi untuk tokenisasi yang lebih robust
def safe_tokenize(text):
    """
    Tokenisasi teks dengan penanganan error
    """
    try:
        return word_tokenize(text)
    except Exception as e:
        # Fallback ke tokenisasi sederhana jika NLTK tokenizer gagal
        return text.split()

def print_timestamp(message):
    """Print message with timestamp"""
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{current_time}] {message}")

def clean_repeated_chars(text):
    """
    Membersihkan karakter yang berulang
    Contoh: 'bagusssss' -> 'bagus'
    """
    pattern = r'(.)\1{2,}'
    return re.sub(pattern, r'\1', text)

def normalize_text(text, slang_dict, informal_dict):
    """
    Fungsi untuk menormalisasi teks menggunakan kedua kamus
    """
    words = text.split()
    normalized = []
    
    for word in words:
        # Coba cari di kamus slang
        if word in slang_dict:
            normalized.append(slang_dict[word])
        # Coba cari di kamus informal
        elif word in informal_dict:
            normalized.append(informal_dict[word])
        else:
            # Jika tidak ada di kedua kamus, gunakan kata asli
            normalized.append(word)
    
    return ' '.join(normalized)

def normalize_patterns(text):
    """
    Normalisasi pola-pola umum dalam teks
    """
    # Normalisasi akhiran
    text = re.sub(r'ny[auh]$', 'nya', text)
    text = re.sub(r'[x]+$', '', text)
    text = re.sub(r'[zs]k$', 's', text)
    
    # Normalisasi awalan
    text = re.sub(r'^ng', 'meng', text)
    text = re.sub(r'^nge', 'menge', text)
    
    # Normalisasi pola umum
    patterns = {
        r'\b(ga?k?|g\b|kga?|nga?k?)\b': 'tidak',
        r'\b(udh?|dah)\b': 'sudah',
        r'\bsdh\b': 'sudah',
        r'\b(aja|ajah?)\b': 'saja',
        r'\b(skrg|skg)\b': 'sekarang',
        r'\bkyk\b': 'seperti',
        r'\bkek\b': 'seperti',
        r'\b(gmn|gmana)\b': 'bagaimana',
        r'\b(dmn|dimana)\b': 'dimana',
        r'\b(dong|donk|dunk)\b': 'dong',
        r'\b(gpp|gapapa)\b': 'tidak apa-apa',
        r'\b(gbs|gabisa)\b': 'tidak bisa',
        r'\btrs\b': 'terus',
        r'\bsy\b': 'saya',
        r'\bbgt+\b': 'sangat',
        r'\bbngt+\b': 'sangat',
        r'\bbanget+\b': 'sangat',
        r'\byg\b': 'yang',
        r'\blg\b': 'lagi',
        r'\bklo\b': 'kalau',
        r'\bkl\b': 'kalau',
        r'\btq\b': 'terima kasih',
        r'\bmksh\b': 'terima kasih',
        r'\bthx\b': 'terima kasih',
        r'\btks\b': 'terima kasih',
        r'\bsma\b': 'sama',
        r'\bkmrn\b': 'kemarin',
        r'\bbs\b': 'bisa',
        r'\bbrp\b': 'berapa',
        r'\bpke?\b': 'pakai',
        r'\bpke?n\b': 'pakai',
    }
    
    for pattern, replacement in patterns.items():
        text = re.sub(pattern, replacement, text)
    
    return text

def get_sentiment_scores(text, senti_indo):
    """
    Menghitung skor sentiment untuk teks
    """
    try:
        tokens = safe_tokenize(text)
        token_scores = []
        
        # Analisis per token
        for token in tokens:
            score = senti_indo.polarity_scores(token)
            lexicon_value = senti_indo.lexicon.get(token, 0)
            token_scores.append({
                'token': token,
                'lexicon_value': lexicon_value,
                'neg': score['neg'],
                'neu': score['neu'],
                'pos': score['pos'],
                'compound': score['compound']
            })
        
        # Analisis keseluruhan teks
        full_text_score = senti_indo.polarity_scores(text)
        
        return {
            'token_scores': token_scores,
            'text_score': full_text_score
        }
    except Exception as e:
        print_timestamp(f"Error in sentiment analysis: {str(e)}")
        # Return default values if analysis fails
        return {
            'token_scores': [],
            'text_score': {'neg': 0, 'neu': 1, 'pos': 0, 'compound': 0}
        }

def preprocess_text(text, slang_dict, informal_dict, stemmer, senti_indo):
    try:
        if pd.isna(text):
            print_timestamp("Warning: Found NaN value in text")
            return {
                "original_text": "",
                "preprocessed_text": "",
                "sentiment_scores": None
            }
            
        print_timestamp(f"Processing text: {text[:50]}...")
        
        original_text = text
        
        # Convert to lowercase
        text = text.lower()
        print_timestamp("Completed: Converting to lowercase")
        
        # Remove URLs, mentions, hashtags
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        text = re.sub(r'@\w+', '', text)
        text = re.sub(r'#\w+', '', text)
        print_timestamp("Completed: Removing URLs and mentions")
        
        # Remove punctuation and numbers
        text = re.sub(r'[^\w\s]', ' ', text)
        text = re.sub(r'\d+', '', text)
        text = re.sub(r'\s+', ' ', text).strip()
        print_timestamp("Completed: Removing punctuation and numbers")
        
        # Clean repeated characters
        text = clean_repeated_chars(text)
        print_timestamp("Completed: Cleaning repeated characters")
        
        # Normalize patterns
        text = normalize_patterns(text)
        print_timestamp("Completed: Normalizing patterns")
        
        # Normalize using dictionaries
        text = normalize_text(text, slang_dict, informal_dict)
        print_timestamp("Completed: Normalizing using dictionaries")
        
        # Stemming
        words = text.split()
        stemmed_words = [stemmer.stem(word) for word in words]
        preprocessed_text = ' '.join(stemmed_words)
        print_timestamp("Completed: Stemming")
        
        # Sentiment Analysis
        sentiment_scores = get_sentiment_scores(preprocessed_text, senti_indo)
        print_timestamp("Completed: Sentiment Analysis")
        
        return {
            "original_text": original_text,
            "preprocessed_text": preprocessed_text.strip(),
            "sentiment_scores": sentiment_scores
        }
    
    except Exception as e:
        print_timestamp(f"Error processing text: {str(e)}")
        return {
            "original_text": text,
            "preprocessed_text": text,
            "sentiment_scores": None
        }

def perform_classification(df):
    """
    Perform text classification using multiple models
    """
    print_timestamp("Starting text classification...")
    
    # Create TF-IDF vectors
    tfidf = TfidfVectorizer(max_features=5000)
    X = tfidf.fit_transform(df['preprocessed_text'])
    
    # Convert sentiment labels to numeric
    label_map = {'negative': 0, 'neutral': 1, 'positive': 2}
    labels = df['sentiment_label'].map(label_map)
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, labels, test_size=0.25, random_state=42, stratify=labels
    )
    
    # Dictionary to store model results
    model_results = {}
    
    # 1. Logistic Regression
    print_timestamp("Training Logistic Regression...")
    log_reg = LogisticRegression(max_iter=1000)
    log_reg.fit(X_train, y_train)
    y_pred_log_reg = log_reg.predict(X_test)
    model_results['Logistic Regression'] = {
        'accuracy': accuracy_score(y_test, y_pred_log_reg),
        'report': classification_report(y_test, y_pred_log_reg, 
                                     labels=[0, 1, 2], 
                                     target_names=["Negative", "Neutral", "Positive"],
                                     zero_division=0)
    }

    # 2. Random Forest
    print_timestamp("Training Random Forest...")
    rf_clf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_clf.fit(X_train, y_train)
    y_pred_rf = rf_clf.predict(X_test)
    model_results['Random Forest'] = {
        'accuracy': accuracy_score(y_test, y_pred_rf),
        'report': classification_report(y_test, y_pred_rf,
                                     labels=[0, 1, 2],
                                     target_names=["Negative", "Neutral", "Positive"],
                                     zero_division=0)
    }

    # 3. Naive Bayes
    print_timestamp("Training Naive Bayes...")
    nb_clf = MultinomialNB()
    nb_clf.fit(X_train, y_train)
    y_pred_nb = nb_clf.predict(X_test)
    model_results['Naive Bayes'] = {
        'accuracy': accuracy_score(y_test, y_pred_nb),
        'report': classification_report(y_test, y_pred_nb,
                                     labels=[0, 1, 2],
                                     target_names=["Negative", "Neutral", "Positive"],
                                     zero_division=0)
    }

    # 4. KNN
    print_timestamp("Training KNN...")
    knn_clf = KNeighborsClassifier(n_neighbors=5)
    knn_clf.fit(X_train, y_train)
    y_pred_knn = knn_clf.predict(X_test)
    model_results['KNN'] = {
        'accuracy': accuracy_score(y_test, y_pred_knn),
        'report': classification_report(y_test, y_pred_knn,
                                     labels=[0, 1, 2],
                                     target_names=["Negative", "Neutral", "Positive"],
                                     zero_division=0)
    }

    return model_results, tfidf

def main():
    try:
        start_time = time.time()
        print_timestamp("Starting enhanced preprocessing and classification pipeline...")
        
        # Initialize stemmer
        factory = StemmerFactory()
        stemmer = factory.create_stemmer()
        print_timestamp("Stemmer initialized successfully")
        
        # Initialize sentiment analyzer
        senti_indo = SentimentIntensityAnalyzer()
        
        # Load Indonesian sentiment lexicon
        print_timestamp("Loading Indonesian sentiment lexicon...")
        try:
            # Load lexicon from file
            url = 'https://drive.google.com/file/d/1qPX0Uej3PqUQUI3op_oeEr8AdmrgOT2V/view?usp=sharing'
            path = 'https://drive.google.com/uc?export=download&id='+url.split('/')[-2]
            df_senti = pd.read_csv(path, sep=':', names=['word', 'value'])
            
            # Convert to dictionary
            senti_dict = {}
            for i in range(len(df_senti)):
                senti_dict[df_senti.iloc[i]['word']] = df_senti.iloc[i]['value']
            
            # Update sentiment analyzer lexicon
            senti_indo.lexicon.update(senti_dict)
            
            # Add additional sentiment words
            kata_tambahan = {
                "pudar": -5,
                "fast": 1,
            }
            senti_indo.lexicon.update(kata_tambahan)
            
            print_timestamp("Sentiment lexicon loaded and updated successfully")
            
        except Exception as e:
            print_timestamp(f"Error loading sentiment lexicon: {str(e)}")
            return
        
        # Load dictionaries
        try:
            # Kamus kata informal ke formal
            informal_dict = {
                'gak': 'tidak',
                'ga': 'tidak',
                'nggak': 'tidak',
                'gk': 'tidak',
                'donk': 'dong',
                'dong': 'dong',
                'nyaa': 'nya',
                'nya': 'nya',
                'aja': 'saja',
                'ajah': 'saja',
                'doang': 'saja',
                'udah': 'sudah',
                'dah': 'sudah',
                'udh': 'sudah',
                'tuh': 'itu',
                'neh': 'ini',
                'gini': 'begini',
                'gitu': 'begitu',
                'gmn': 'bagaimana',
                'gimana': 'bagaimana',
                'thx': 'terima kasih',
                'makasih': 'terima kasih',
                'mksh': 'terima kasih',
                'bgt': 'sangat',
                'banget': 'sangat',
                'bgtt': 'sangat',
                'bngtt': 'sangat',
                'skrg': 'sekarang',
                'skg': 'sekarang',
                'yg': 'yang',
                'yng': 'yang',
                'kalo': 'kalau',
                'klo': 'kalau',
                'gpp': 'tidak apa apa',
                'gapapa': 'tidak apa apa',
                'gabisa': 'tidak bisa',
                'gbs': 'tidak bisa',
                'gk bisa': 'tidak bisa',
                'krn': 'karena',
                'karna': 'karena',
                'lg': 'lagi',
                'lgi': 'lagi',
                'dgn': 'dengan',
                'dngn': 'dengan',
                'pke': 'pakai',
                'pake': 'pakai',
                'bisa': 'dapat',
                'bs': 'bisa',
                'trs': 'terus',
                'trus': 'terus',
                'truz': 'terus',
                'sih': 'saja',
                'si': 'saja',
                'nih': 'ini',
                'ni': 'ini',
                'gtu': 'begitu',
                'gt': 'begitu',
                'ngga': 'tidak',
                'nggak': 'tidak',
                'gada': 'tidak ada',
                'gaada': 'tidak ada',
                'kyk': 'seperti',
                'kyak': 'seperti',
                'kek': 'seperti',
                'biar': 'agar',
                'tar': 'nanti',
                'ntr': 'nanti',
                'ntaar': 'nanti',
                'dlu': 'dulu',
                'liat': 'lihat',
                'ngeliat': 'melihat'
            }
            
            # Load slang dictionary
            with open('Data/slang_abbrevations_words.txt', 'r') as file:
                slang_dict = json.load(file)
                
            print_timestamp("Dictionaries loaded successfully")
            
        except Exception as e:
            print_timestamp(f"Error loading dictionaries: {str(e)}")
            return
            
        # Load dataset
        print_timestamp("Loading dataset...")
        try:
            df = pd.read_csv('Output/1_extract_text.csv')
            print_timestamp(f"Loaded dataset with {len(df)} rows")
        except Exception as e:
            print_timestamp(f"Error loading dataset: {str(e)}")
            return
            
        # Process texts with progress bar
        print_timestamp("Starting text preprocessing...")
        tqdm.pandas(desc="Processing texts")
        results = df['content'].progress_apply(
            lambda x: preprocess_text(x, slang_dict, informal_dict, stemmer, senti_indo)
        )
        
        # Extract results
        df['original_text'] = results.apply(lambda x: x['original_text'])
        df['preprocessed_text'] = results.apply(lambda x: x['preprocessed_text'])
        
        # Extract sentiment scores
        df['sentiment_compound'] = results.apply(
            lambda x: x['sentiment_scores']['text_score']['compound'] if x['sentiment_scores'] else None
        )
        
        # Add sentiment labels
        df['sentiment_label'] = df['sentiment_compound'].apply(
            lambda x: 'positive' if x >= 0.05 else ('negative' if x <= -0.05 else 'neutral') if pd.notnull(x) else None
        )
        
        # Save preprocessing results
        print_timestamp("Saving preprocessed texts...")
        try:
            output_columns = [
                'original_text',
                'preprocessed_text',
                'sentiment_compound',
                'sentiment_label'
            ]
            df[output_columns].to_csv('Output/2_preprocessed_text_sentiment_analysis_gopay.csv', index=False)
            print_timestamp("Preprocessing results saved successfully")
        except Exception as e:
            print_timestamp(f"Error saving preprocessing results: {str(e)}")
            return

        # Create initial TF-IDF vectors
        print_timestamp("\nCreating TF-IDF vectors with different configurations...")
        
        # Experiment 1: Default TF-IDF
        print_timestamp("\nExperiment 1: Default TF-IDF")
        tfidf_default = TfidfVectorizer(max_features=5000)
        X_default = tfidf_default.fit_transform(df['preprocessed_text'].fillna(''))
        
        # Experiment 2: TF-IDF with Bigrams
        print_timestamp("Experiment 2: TF-IDF with Bigrams")
        tfidf_bigram = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
        X_bigram = tfidf_bigram.fit_transform(df['preprocessed_text'].fillna(''))
        
        # Experiment 3: TF-IDF with Custom Parameters
        print_timestamp("Experiment 3: TF-IDF with Custom Parameters")
        tfidf_custom = TfidfVectorizer(
            max_features=10000,
            min_df=2,
            max_df=0.9,
            ngram_range=(1, 3)
        )
        X_custom = tfidf_custom.fit_transform(df['preprocessed_text'].fillna(''))
        
        # Experiment 4: TF-IDF with Strict Filtering
        print_timestamp("Experiment 2: TF-IDF with Strict Filtering")
        tfidf_strict = TfidfVectorizer(
            max_features=8000,
            min_df=5,
            max_df=0.8,
            use_idf=True,
            norm='l2',
            sublinear_tf=True
        )
        X_strict = tfidf_strict.fit_transform(df['preprocessed_text'].fillna(''))
        
        # Experiment 5: TF-IDF with Enhanced Features
        print_timestamp("Experiment 3: TF-IDF with Enhanced Features")
        tfidf_enhanced = TfidfVectorizer(
            max_features=15000,
            min_df=3,
            max_df=0.95,
            use_idf=True,
            smooth_idf=True,
            sublinear_tf=True
        )
        X_enhanced = tfidf_enhanced.fit_transform(df['preprocessed_text'].fillna(''))
        
        # Convert labels
        label_map = {'negative': 0, 'neutral': 1, 'positive': 2}
        labels = df['sentiment_label'].map(label_map)
        
        # Create dictionary to store all results
        all_results = {
            'Default TF-IDF': {'X': X_default},
            'Bigram TF-IDF': {'X': X_bigram},
            'Custom TF-IDF': {'X': X_custom},
            'Strict Filtering': {'X': X_strict},
            'Enhanced Features': {'X': X_enhanced}
        }
        
        # Initialize models
        models = {
            'Logistic Regression': LogisticRegression(max_iter=1000),
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
            'Naive Bayes': MultinomialNB(),
            'KNN': KNeighborsClassifier(n_neighbors=5),
            'SVM': SVC(kernel='linear', random_state=42)
        }
        
        # Evaluate all combinations
        print_timestamp("\nEvaluating all model and TF-IDF combinations...")
        results_table = []
        
        for tfidf_name, tfidf_data in all_results.items():
            X = tfidf_data['X']
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, labels, test_size=0.25, random_state=42, stratify=labels
            )
            
            for model_name, model in models.items():
                print_timestamp(f"\nTraining {model_name} with {tfidf_name}...")
                
                # Train and evaluate
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                
                # Calculate metrics
                accuracy = accuracy_score(y_test, y_pred)
                report = classification_report(
                    y_test, y_pred,
                    labels=[0, 1, 2],
                    target_names=["Negative", "Neutral", "Positive"],
                    zero_division=0,
                    output_dict=True
                )
                
                # Store results
                results_table.append({
                    'TF-IDF Type': tfidf_name,
                    'Model': model_name,
                    'Accuracy': accuracy,
                    'Macro Avg F1': report['macro avg']['f1-score'],
                    'Weighted Avg F1': report['weighted avg']['f1-score']
                })
                
                # Print results
                print_timestamp(f"Accuracy: {accuracy:.4f}")
                print_timestamp("\nClassification Report:")
                print(classification_report(
                    y_test, y_pred,
                    labels=[0, 1, 2],
                    target_names=["Negative", "Neutral", "Positive"],
                    zero_division=0
                ))
        
        # Convert results to DataFrame and save
        results_df = pd.DataFrame(results_table)
        results_df.to_csv('Output/all_experiments_results.csv', index=False)
        
        # Find best combination
        best_result = max(results_table, key=lambda x: x['Accuracy'])
        print_timestamp(f"\nBest performing combination:")
        print_timestamp(f"TF-IDF: {best_result['TF-IDF Type']}")
        print_timestamp(f"Model: {best_result['Model']}")
        print_timestamp(f"Accuracy: {best_result['Accuracy']:.4f}")
        
        # Hyperparameter tuning for best model
        print_timestamp("\nPerforming hyperparameter tuning on best model...")
        
        # Get best configuration
        best_tfidf = all_results[best_result['TF-IDF Type']]['X']
        X_train, X_test, y_train, y_test = train_test_split(
            best_tfidf, labels, test_size=0.25, random_state=42, stratify=labels
        )
        
        if best_result['Model'] == 'Random Forest':
            classes = np.unique(y_train)
            class_weights = compute_class_weight('balanced', classes=classes, y=y_train)
            class_weight_dict = dict(zip(classes, class_weights))
            
            param_grid = {
                'n_estimators': [500, 800],          # Hanya 2 pilihan
                'max_depth': [50, None],             # Hanya 2 pilihan
                'min_samples_split': [5],            # Fixed value
                'min_samples_leaf': [2],             # Fixed value
                'max_features': ['sqrt', 'log2'],    # 2 pilihan yang paling umum
                'criterion': ['gini']                # Fixed value
            }
            base_model = RandomForestClassifier(
                n_jobs=-1,
                random_state=42,
                class_weight='balanced',             # Menggunakan built-in balanced
                bootstrap=True
            )
        elif best_result['Model'] == 'SVM':
            param_grid = {
                'C': [0.1, 1, 5, 10, 15, 20],
                'kernel': ['linear', 'rbf'],
                'gamma': ['scale', 'auto']
            }
            base_model = SVC(random_state=42)
        else:
            # Add other model configurations as needed
            param_grid = {}
            base_model = models[best_result['Model']]
        
        if param_grid:
            print_timestamp("\nStarting optimized Grid Search...")
            
            # Hitung total kombinasi
            n_combinations = np.prod([len(v) for v in param_grid.values()])
            print_timestamp(f"Total combinations to try: {n_combinations}")
            
            grid_search = GridSearchCV(base_model, param_grid, cv=5, scoring='accuracy', n_jobs=-1, verbose=1, return_train_score=True)
            print_timestamp("\nStarting Grid Search...")
            # Fit dengan progress bar
            with tqdm(total=n_combinations, desc="Grid Search Progress") as pbar:
                grid_search.fit(X_train, y_train)
                pbar.update(n_combinations)
            
            print_timestamp("\nBest parameters found:")
            print_timestamp(grid_search.best_params_)
            
            # Check for overfitting
            best_idx = grid_search.best_index_
            train_score = grid_search.cv_results_['mean_train_score'][best_idx]
            test_score = grid_search.cv_results_['mean_test_score'][best_idx]
            
            print_timestamp(f"\nTrain Score: {train_score:.4f}")
            print_timestamp(f"Test Score: {test_score:.4f}")
            
            # Evaluate final model
            final_y_pred = grid_search.predict(X_test)
            final_accuracy = accuracy_score(y_test, final_y_pred)
            final_report = classification_report(
                y_test, final_y_pred,
                labels=[0, 1, 2],
                target_names=["Negative", "Neutral", "Positive"],
                zero_division=0
            )
            
            print_timestamp(f"\nFinal Model Accuracy: {final_accuracy:.4f}")
            print_timestamp("\nFinal Classification Report:")
            print_timestamp(final_report)
            
            # Save final model
            print_timestamp("\nSaving final model and vectorizer...")
            with open('Output/final_model.pkl', 'wb') as f:
                pickle.dump(grid_search, f)
            
            with open('Output/final_tfidf.pkl', 'wb') as f:
                if best_result['TF-IDF Type'] == 'Default TF-IDF':
                    pickle.dump(tfidf_default, f)
                elif best_result['TF-IDF Type'] == 'Bigram TF-IDF':
                    pickle.dump(tfidf_bigram, f)
                else:
                    pickle.dump(tfidf_custom, f)
        
        # Create visualizations
        plt.figure(figsize=(12, 6))
        sns.barplot(
            data=results_df,
            x='Model',
            y='Accuracy',
            hue='TF-IDF Type'
        )
        plt.title('Model Performance Comparison')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig('Output/model_comparison.png')
        
        # Create comparison table
        pivot_table = results_df.pivot_table(
            values='Accuracy',
            index='Model',
            columns='TF-IDF Type',
            aggfunc='first'
        )
        
        # Save detailed report
        with open('Output/final_report.txt', 'w') as f:
                f.write("LAPORAN ANALISIS SENTIMEN DAN KLASIFIKASI TEKS\n")
                f.write("=" * 50 + "\n\n")
                
                # 1. Dataset Information
                f.write("1. INFORMASI DATASET\n")
                f.write("-" * 30 + "\n")
                f.write(f"Total data: {len(df)}\n")
                f.write("\nDistribusi Label Sentimen:\n")
                sentiment_dist = df['sentiment_label'].value_counts()
                for label, count in sentiment_dist.items():
                    percentage = (count / len(df)) * 100
                    f.write(f"{label}: {count} ({percentage:.2f}%)\n")
                f.write("\n")
                
                # 2. Preprocessing Results
                f.write("\n2. HASIL PREPROCESSING\n")
                f.write("-" * 30 + "\n")
                f.write("Statistik Sentimen:\n")
                f.write(f"Rata-rata skor compound: {df['sentiment_compound'].mean():.3f}\n")
                f.write(f"Median skor compound: {df['sentiment_compound'].median():.3f}\n")
                f.write(f"Standar deviasi skor compound: {df['sentiment_compound'].std():.3f}\n\n")
                
                # 3. Model Comparison
                f.write("\n3. PERBANDINGAN MODEL\n")
                f.write("-" * 30 + "\n")
                f.write("\nTabel Perbandingan Akurasi:\n")
                f.write(pivot_table.to_string())
                f.write("\n\n")
                
                # 4. Best Model Results
                f.write("\n4. HASIL MODEL TERBAIK\n")
                f.write("-" * 30 + "\n")
                f.write(f"TF-IDF Configuration: {best_result['TF-IDF Type']}\n")
                f.write(f"Model: {best_result['Model']}\n")
                f.write(f"Accuracy: {best_result['Accuracy']:.4f}\n")
                
                if param_grid:
                    f.write("\nHasil Hyperparameter Tuning:\n")
                    f.write(f"Best Parameters: {grid_search.best_params_}\n")
                    f.write(f"Final Accuracy: {final_accuracy:.4f}\n")
                    f.write("\nFinal Classification Report:\n")
                    f.write(final_report)
                
                # 5. Recommendations
                f.write("\n\n5. REKOMENDASI PENINGKATAN AKURASI\n")
                f.write("-" * 30 + "\n")
                f.write("""
1. Feature Engineering:
   - Eksplorasi berbagai konfigurasi TF-IDF
   - Pertimbangkan penggunaan word embeddings
   - Tambahkan fitur linguistik tambahan

2. Model Optimization:
   - Fine-tuning parameter lebih lanjut
   - Ensemble methods
   - Deep learning approaches

3. Data Enhancement:
   - Augmentasi data
   - Cleansing data lebih lanjut
   - Balanced sampling techniques
""")
        
        # Model comparison visualization
        plt.figure(figsize=(15, 6))

        # Get best accuracy dynamically
        best_accuracy = best_result['Accuracy']  # Mengambil dari hasil terbaik

        # Accuracy comparison dengan highlight best accuracy
        plt.subplot(1, 2, 1)
        sns.barplot(data=results_df, x='TF-IDF Type', y='Accuracy', hue='Model')
        plt.title('Model Accuracy by TF-IDF Type')
        plt.xticks(rotation=45)
        plt.axhline(y=best_accuracy, color='r', linestyle='--', 
                    label=f'Best Accuracy ({best_accuracy:.3f})')
        plt.legend(title='Model', bbox_to_anchor=(1.05, 1), loc='upper left')

        # F1-Score comparison
        plt.subplot(1, 2, 2)
        sns.barplot(data=results_df, x='TF-IDF Type', y='Macro Avg F1', hue='Model')
        plt.title('Model F1-Score by TF-IDF Type')
        plt.xticks(rotation=45)
        plt.legend(title='Model', bbox_to_anchor=(1.05, 1), loc='upper left')

        plt.tight_layout()
        plt.savefig('Output/model_metrics_comparison.png')
        plt.close()
        
        # Accuracy Heatmap
        plt.figure(figsize=(10, 6))
        pivot_accuracy = results_df.pivot(
            index='Model',
            columns='TF-IDF Type',
            values='Accuracy'
        )
        sns.heatmap(pivot_accuracy, annot=True, fmt='.3f', cmap='YlOrRd')
        plt.title('Accuracy Heatmap: Model vs TF-IDF Type')
        plt.tight_layout()
        plt.savefig('Output/accuracy_heatmap.png')

        # Parameter C Analysis (jika SVM adalah model terbaik)
        if best_result['Model'] == 'SVM' and grid_search.best_params_['kernel'] == 'linear':
            plt.figure(figsize=(10, 6))
            cv_results = pd.DataFrame(grid_search.cv_results_)
            c_values = [0.1, 1, 5, 10, 15, 20]
            accuracies = []
            stds = []
            
            for c in c_values:
                mask = (cv_results['param_C'] == c) & \
                    (cv_results['param_kernel'] == grid_search.best_params_['kernel']) & \
                    (cv_results['param_gamma'] == grid_search.best_params_['gamma'])
                accuracies.append(cv_results[mask]['mean_test_score'].iloc[0])
                stds.append(cv_results[mask]['std_test_score'].iloc[0])
            
            plt.errorbar(c_values, accuracies, yerr=stds, fmt='o-', capsize=5)
            plt.xlabel('Nilai C')
            plt.ylabel('Accuracy')
            plt.title(f'Pengaruh Parameter C pada Akurasi\n(Kernel={grid_search.best_params_["kernel"]}, gamma={grid_search.best_params_["gamma"]})')
            plt.grid(True)
            plt.savefig('Output/svm_c_parameter_analysis.png')
            plt.close()
        
        # Parameter Analysis untuk Random Forest
        if best_result['Model'] == 'Random Forest':
            plt.figure(figsize=(15, 5))
            
            # Plot pengaruh n_estimators
            plt.subplot(1, 3, 1)
            cv_results = pd.DataFrame(grid_search.cv_results_)
            n_estimator_values = [500, 800]
            accuracies = []
            stds = []
            
            for n_est in n_estimator_values:
                mask = (cv_results['param_n_estimators'] == n_est)
                accuracies.append(cv_results[mask]['mean_test_score'].mean())
                stds.append(cv_results[mask]['std_test_score'].mean())
            
            plt.errorbar(n_estimator_values, accuracies, yerr=stds, fmt='o-', capsize=5)
            plt.xlabel('Number of Trees')
            plt.ylabel('Accuracy')
            plt.title('Effect of n_estimators')
            plt.grid(True)
            
            # Plot pengaruh max_depth
            plt.subplot(1, 3, 2)
            accuracies_depth = {
                '50': cv_results[cv_results['param_max_depth'] == 50]['mean_test_score'].mean(),
                'None': cv_results[cv_results['param_max_depth'].isnull()]['mean_test_score'].mean()
            }
            stds_depth = {
                '50': cv_results[cv_results['param_max_depth'] == 50]['std_test_score'].mean(),
                'None': cv_results[cv_results['param_max_depth'].isnull()]['std_test_score'].mean()
            }
            
            plt.bar(accuracies_depth.keys(), accuracies_depth.values())
            plt.xlabel('Max Depth')
            plt.ylabel('Accuracy')
            plt.title('Effect of max_depth')
            
            # Plot pengaruh max_features
            plt.subplot(1, 3, 3)
            feature_accuracies = {
                'sqrt': cv_results[cv_results['param_max_features'] == 'sqrt']['mean_test_score'].mean(),
                'log2': cv_results[cv_results['param_max_features'] == 'log2']['mean_test_score'].mean()
            }
            
            plt.bar(feature_accuracies.keys(), feature_accuracies.values())
            plt.xlabel('Max Features')
            plt.ylabel('Accuracy')
            plt.title('Effect of max_features')
            
            plt.tight_layout()
            plt.savefig('Output/rf_parameter_analysis.png')
            plt.close()
            
        # Save detailed results
        results_detail = {
            'Experiment': [],
            'Value': []
        }

        results_detail['Experiment'].extend([
            'Best TF-IDF Configuration',
            'Best Model',
            'Best Parameters',
            'Initial Accuracy',
            'Final Tuned Accuracy'
        ])

        results_detail['Value'].extend([
            best_result['TF-IDF Type'],
            best_result['Model'],
            str(grid_search.best_params_),
            best_result['Accuracy'],
            final_accuracy
        ])

        pd.DataFrame(results_detail).to_csv('Output/detailed_results.csv', index=False)
        
        # Save detailed evaluation metrics
        model_evaluation = {
            'Metric': [
                'Best TF-IDF Configuration',
                'Best Model',
                'Best Parameters',
                'Initial Accuracy',
                'Final Tuned Accuracy',
                'Training Score',
                'Test Score',
                'Train-Test Gap',
                'Cross-validation Mean',
                'Cross-validation Std',
                'Number of Features',
                'Training Samples',
                'Test Samples',
                'Total Parameters Tried',
                'Training Time (seconds)'
            ],
            'Value': [
                best_result['TF-IDF Type'],
                best_result['Model'],
                str(grid_search.best_params_),
                f"{best_result['Accuracy']:.4f}",
                f"{final_accuracy:.4f}",
                f"{train_score:.4f}",
                f"{test_score:.4f}",
                f"{train_score - test_score:.4f}",
                f"{cv_scores.mean():.4f}" if 'cv_scores' in locals() else 'N/A',
                f"{cv_scores.std():.4f}" if 'cv_scores' in locals() else 'N/A',
                X_train.shape[1],
                X_train.shape[0],
                X_test.shape[0],
                len(grid_search.cv_results_['params']),
                f"{time.time() - start_time:.2f}"
            ]
        }

        pd.DataFrame(model_evaluation).to_csv('Output/model_evaluation_results.csv', index=False)

        # Save grid search results
        tuning_results = pd.DataFrame({
            'Parameter Combination': [str(params) for params in grid_search.cv_results_['params']],
            'Mean Test Score': grid_search.cv_results_['mean_test_score'],
            'Std Test Score': grid_search.cv_results_['std_test_score'],
            'Mean Train Score': grid_search.cv_results_['mean_train_score'] if 'mean_train_score' in grid_search.cv_results_ else None,
            'Rank': grid_search.cv_results_['rank_test_score'],
            'Time': grid_search.cv_results_['mean_fit_time']
        })

        # Sort by mean test score
        tuning_results = tuning_results.sort_values('Mean Test Score', ascending=False)

        # Save top 10 results
        tuning_results.head(10).to_csv('Output/hyperparameter_tuning_results.csv', index=False)

        # Print summary of best results
        print_timestamp("\nBest Parameter Combinations (Top 3):")
        for idx, row in tuning_results.head(3).iterrows():
            print_timestamp(f"\nRank {row['Rank']}:")
            print_timestamp(f"Parameters: {row['Parameter Combination']}")
            print_timestamp(f"Mean Test Score: {row['Mean Test Score']:.4f} (±{row['Std Test Score']:.4f})")
        
        # Print execution summary
        end_time = time.time()
        duration = round(end_time - start_time, 2)
        print_timestamp(f"\nEntire pipeline completed in {duration} seconds")
        print_timestamp("\nFiles generated:")
        print_timestamp("1. Output/2_preprocessed_text_sentiment_analysis_gopay.csv (Preprocessing results)")
        print_timestamp("2. Output/all_experiments_results.csv (All experiments results)")
        print_timestamp("3. Output/final_model.pkl (Best model)")
        print_timestamp("4. Output/final_tfidf.pkl (Best TF-IDF vectorizer)")
        print_timestamp("5. Output/model_comparison.png (Visualization)")
        print_timestamp("6. Output/model_metrics_comparison.png (Metrics visualization)")
        print_timestamp("7. Output/final_report.txt (Detailed report)")
        print_timestamp("8. Output/rf_parameter_analysis.png (Random Forest parameter analysis)")
        print_timestamp("9. Output/accuracy_heatmap.png (Model vs TF-IDF accuracy heatmap)")
        print_timestamp("10. Output/confusion_matrix.csv (Confusion matrix results)")
        print_timestamp("11. Output/model_evaluation_results.csv (Detailed evaluation metrics)")
        print_timestamp("12. Output/hyperparameter_tuning_results.csv (Grid search results)")
        
        print_timestamp("\nBEST MODEL SUMMARY:")
        print_timestamp(f"Best TF-IDF: {best_result['TF-IDF Type']}")
        print_timestamp(f"Best Model: {best_result['Model']}")
        print_timestamp(f"Best Accuracy: {best_result['Accuracy']:.4f}")
        if param_grid:
            print_timestamp(f"Final Tuned Accuracy: {final_accuracy:.4f}")
        
    except Exception as e:
        print_timestamp(f"Critical error in main process: {str(e)}")
        raise

if __name__ == "_main_":
    main()