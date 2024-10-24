import pandas as pd
import nltk
# # Mendownload daftar kata yang ada (vocabulary)
nltk.download('stopwords')
nltk.download('punkt')
from nltk.corpus import stopwords
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
import string
from nltk.tokenize import word_tokenize
import re


path = 'Data/Data_Review_Aplikasi_BRIMO_1204210044.csv'
df = pd.read_csv(path)

print(df['review'])

df['review'].to_csv('Output/1_extract_text.csv', index=False)

print("Kolom 'review' berhasil disimpan ke 'Output/1_extract_text.csv'")