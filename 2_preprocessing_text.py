import pandas as pd
import re
import string
import json  # Tambahkan impor json untuk membaca file slang
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

# Membuat stemmer dari Sastrawi
factory = StemmerFactory()
stemmer = factory.create_stemmer()

# Tokenisasi sederhana menggunakan spasi
def simple_tokenize(text):
    return text.split()

# Fungsi untuk normalisasi slang
def normalize_slang(text, slang_dict):
    words = simple_tokenize(text)  # Tokenisasi manual
    normalized_words = [slang_dict.get(word, word) for word in words]  # Ganti slang dengan kata yang benar
    return ' '.join(normalized_words)

# Fungsi untuk preprocessing teks
def preprocess_text(text, slang_dict, stemmer):
    text = text.lower()
    text = re.sub(r'@\w+\s*', '', text)  # Hapus mention
    text = re.sub(r'https?://\S+', '', text)  # Hapus link
    text = re.sub('\[.*?\]', ' ', text)
    text = re.sub('\(.*?\)', ' ', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), ' ', text)  # Hapus tanda baca
    text = re.sub('\w*\d\w*', ' ', text)
    text = re.sub('[‘’“”…♪♪]', '', text)  # Hapus karakter spesial
    text = re.sub('\n', ' ', text)
    text = re.sub('\xa0', ' ', text)
    text = re.sub('b ', ' ', text)
    text = re.sub('rt ', ' ', text)
    text = re.sub(r'\s+', ' ', text)  # Hapus spasi berlebih

    # Tokenisasi manual
    tokens = simple_tokenize(text)

    # Normalisasi slang
    tokens = [slang_dict.get(word, word) for word in tokens]

    # Stemming menggunakan Sastrawi
    tokens = [stemmer.stem(word) for word in tokens]

    # Gabungkan token kembali menjadi kalimat
    return ' '.join(tokens)

# Load slang dictionary dari file slang_abbrevations_words.txt
with open('Data/slang_abbrevations_words.txt', 'r') as file:
    slang_dict = json.load(file)  # Membaca file JSON

# Load data dari 'Output/1_extract_text.csv'
path = 'Output/1_extract_text.csv'
df = pd.read_csv(path)

# Terapkan preprocessing pada kolom 'review'
df['preprocessed_text'] = df['review'].apply(lambda x: preprocess_text(x, slang_dict, stemmer))

# Simpan hanya kolom 'preprocessed_text' ke file CSV baru
df[['preprocessed_text']].to_csv('Output/2_preprocessed_text_output.csv', index=False)

print("Preprocessing selesai, hasil kolom 'preprocessed_text' disimpan ke 'Output/2_preprocessed_text_output.csv'")
