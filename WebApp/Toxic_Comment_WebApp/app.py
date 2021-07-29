import os
import pandas as pd
import pickle
from flask import Flask, render_template, request
from tensorflow import keras
from tensorflow.keras.preprocessing.sequence import pad_sequences

app = Flask(__name__)

path_parent = os.getcwd()
path_req = path_parent + '/models'
labels = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
tokenizer = pickle.load(open("https://drive.google.com/file/d/1arErOS3DgwmeshJ1Ig30gXE7N1MCSbqw/view?usp=sharing"))
loaded_model_word2vec = keras.models.load_model(path_req + "/LSTM_word2vec")
loaded_model_Glove = keras.models.load_model(path_req + "/LSTM_Glove")

def lstm_word2vec(comment):
    data = [comment]
    comment_data = pd.DataFrame(data, columns = ['Comment'])
    list_tokenized_test = tokenizer.texts_to_sequences(comment_data['Comment'])
    X_te = pad_sequences(list_tokenized_test, maxlen=200)
    preds_test = loaded_model_word2vec.predict(X_te)
    return preds_test

def lstm_glove(comment):
    data = [comment]
    comment_data = pd.DataFrame(data, columns = ['Comment'])
    list_tokenized_test = tokenizer.texts_to_sequences(comment_data['Comment'])
    X_te = pad_sequences(list_tokenized_test, maxlen=200)
    preds_test = loaded_model_Glove.predict(X_te)
    return preds_test

def makeData(comment, i):
    if(i == 1):
        predictions = lstm_glove(comment)
    else:
        predictions = lstm_word2vec(comment)
    #print(predictions)
    sum = 0
    arr = ['{:f}'.format(item) for item in predictions[0]]
    #print(arr)
    for i in arr:
         sum += float(i)
    #print(sum)
    arr.append(str(max([0,(1 - sum)])))
    sum += float(arr[6])
    arr = [float((float(item)/sum)*100) for item in arr]
    #print(arr)
    return arr

@app.route('/')
def main():
    return render_template('home.html')

@app.route('/predict', methods=['POST'])
def custom():
    comment = request.form['task']
    arr = []
    if request.method == 'POST':
        if request.form['submit-button'] == 'glove':
            arr = makeData(comment, 1)
        elif request.form['submit-button'] == 'word2vec':
            print("word2vec found")
            arr = makeData(comment, 2)
        arr.append(comment)
        return render_template('index.html', data = arr)

if __name__ == "__main__":
    app.run(debug=True)