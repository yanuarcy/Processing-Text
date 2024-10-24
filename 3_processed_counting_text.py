import pandas as pd
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from collections import Counter
import re

# Load data dari 'Output/2_preprocessed_text_output.csv'
df = pd.read_csv('Output/2_preprocessed_text_output.csv')  # Pastikan file ini sudah ada

# Inisialisasi stemmer dari Sastrawi
factory = StemmerFactory()
stemmer = factory.create_stemmer()

# List untuk menyimpan semua kata
all_words = []

# Fungsi untuk tokenisasi
def tokenize(text):
    # Menggunakan regex untuk memisahkan kata, cek jika text adalah string
    if isinstance(text, str):
        return re.findall(r'\w+', text)
    else:
        return []

# Iterasi melalui setiap baris dan tokenisasi teks
for index, row in df.iterrows():
    tokens = tokenize(row['preprocessed_text'])  # Tokenisasi menggunakan regex
    all_words.extend(tokens)  # Tambahkan semua token ke dalam satu list

# Hitung frekuensi kata menggunakan Counter
word_freq = Counter(all_words)

# Ambil kata-kata paling umum sebagai daftar tuple (kata, frekuensi)
most_common_words = word_freq.most_common()

# Ubah daftar tuple menjadi DataFrame pandas
df_word_freq = pd.DataFrame(most_common_words, columns=['Word', 'Frequency'])

# Simpan DataFrame ke file CSV
df_word_freq.to_csv('Output/3_processed_counting_text.csv', index=False)

# Cetak konfirmasi
print("Word frequencies saved to 'Output/3_processed_counting_text.csv'")