import csv
from nltk.tokenize import RegexpTokenizer
from sklearn.multiclass import OneVsOneClassifier
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import classification_report
from sklearn import metrics
from sklearn import svm


happy_emoji = [':‑)', ':)', ':-]', ':]',
               ':-3', ':3', ':->', ':>',
               '8-)', '8)', ':-}', ':}',
               ':o)', ':c)', ':^)', '=]',
               '=)', ':‑D', ':D', '8‑D',
               '8D', 'x‑D', 'xD', 'X‑D',
               'XD', '=D', '=3', 'B^D', ':-))']

sad_emoji = [':‑(', ':(', ':‑c', ':c',
             ':‑<', ':<', ':‑[', ':[',
             ':-||', '>:[', ':{', ':@', '>:(']

negative_words = ["doesn't", "isn't", "wasn't", "shouldn't", "wouldn't", "couldn't", "won't", "can't", "don't"]

stop_words = set(stopwords.words('english'))


tokenizer = RegexpTokenizer(r'\w+')
lemmatizer = WordNetLemmatizer()


def preprocess_data(fileName, windowSize = 9, Testing = False):
    with open(fileName, newline='') as inputFile:
        fparser = csv.DictReader(inputFile)
        tokens = []
        asps_tokens = []
        reviews = []
        asp_term = []
        labels = []
        aspects = []
        ids = []
        for row in fparser:
            properAspect = []
            for asp_temp in tokenizer.tokenize(row[' aspect_term']):
                properAspect.append(lemmatizer.lemmatize(asp_temp))

            aspects.append(properAspect)

            if not Testing:
                labels.append(row[' class'])
            else:
                ids.append(row['example_id'])

            tempStr = row[' text'].replace('[comma]', ',').lower()
            tempStr = tempStr.replace('_', ' ').lower()
            tempAspect = row[' aspect_term'].replace('[comma]', ',').lower()
            tempAspect = tempAspect.replace('_', ' ').lower()

            for emoji in happy_emoji:
                if emoji in tempStr:
                    tempStr = tempStr.replace(emoji, 'happy')

            for emoji in sad_emoji:
                if emoji in tempStr:
                    tempStr = tempStr.replace(emoji, 'sad')

            for word in negative_words:
                if word in tempStr:
                    tempStr = tempStr.replace(word, 'not')

            for token in tokenizer.tokenize(tempStr):
                if token not in stop_words or token == 'not':
                    tokens.append(lemmatizer.lemmatize(token))

            for token in tokenizer.tokenize(tempAspect):
                if token not in stop_words or token == 'not':
                    asps_tokens.append(lemmatizer.lemmatize(token))

            reviews.append(tokens)
            asp_term.append(asps_tokens)
            tokens = []
            asps_tokens = []

        processedList = []
        for tk in reviews:
            processedList.append(' '.join(tk))

        newAsps = []
        for tk in asp_term:
            newAsps.append(' '.join(tk))

        windowS = []

        for idx, asp_term in enumerate(newAsps):
            sentence = processedList[idx]
            lh, _, rh = sentence.partition(asp_term)
            window = ' '.join(lh.split()[-windowSize:] + [asp_term] + rh.split()[:windowSize])
            windowS.append(window)

        if not Testing:
            return windowS, labels
        else:
            return windowS, ids





trainData, labels = preprocess_data('Data_1_train.csv', windowSize=9, Testing=False)
testData, ids = preprocess_data('Data-1_test.csv', windowSize=9, Testing=True)

tfi = TfidfVectorizer(ngram_range=(1, 1))
tfidf_train = tfi.fit_transform(trainData)


tfidf_test = tfi.transform(testData)

SVMClassifier = svm.SVC(kernel='linear', C=1, random_state=0)

SVMClassifier.fit(tfidf_train, labels)
predictedClasses = SVMClassifier.predict(tfidf_test)


outfile = open(r'result1.txt','w+')
for i in range(len(ids)):
    print(ids[i], ';;', predictedClasses[i], sep='', file=outfile)
outfile.close()