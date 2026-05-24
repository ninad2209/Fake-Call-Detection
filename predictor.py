import pandas as pd
import faker
import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.linear_model import LogisticRegression

nltk.download('stopwords', quiet=True)

port_stem = PorterStemmer()

def wordopt(Call_Text):
    Call_Text = str(Call_Text).lower()
    Call_Text = re.sub('\[.*?\]', '', Call_Text)
    Call_Text = re.sub("\\W", " ", Call_Text)
    Call_Text = re.sub('https?://\S+|www\.\S+', '', Call_Text)
    Call_Text = re.sub('<.*?>+', '', Call_Text)
    Call_Text = re.sub('[%s]' % re.escape(string.punctuation), '', Call_Text)
    Call_Text = re.sub('\n', '', Call_Text)
    Call_Text = re.sub('\w*\d\w*', '', Call_Text)
    # fix: actually apply stemming and remove stopwords
    words = Call_Text.split()
    Call_Text = ' '.join([port_stem.stem(w) for w in words if w not in stopwords.words('english')])
    return Call_Text

def output(n):
    if n == 0:
        return "Fake Call"
    elif n == 1:
        return "Not a Fake Call"

# ── GENERATE + TRAIN ──────────────────────────────────────
print("Generating dataset...")
fake = faker.Faker()
num_rows = 4000
data = {
    "Call_Text": [fake.sentence() for _ in range(num_rows)],
    "Label": ["1" if i % 3 == 0 else "0" for i in range(num_rows)]
}
df = pd.DataFrame(data)

print("Preprocessing...")
df['Call_Text'] = df['Call_Text'].apply(wordopt)

X = df['Call_Text'].values
Y = df['Label'].values

vectorizer = TfidfVectorizer()
vectorizer.fit(X)
X = vectorizer.transform(X)

X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.4, stratify=Y, random_state=2
)

print("Training Logistic Regression...")
model = LogisticRegression()
model.fit(X_train, Y_train)

print("Training SVM...")
classifier = svm.SVC(kernel='linear', probability=True)
classifier.fit(X_train, Y_train)

print("Models ready!")