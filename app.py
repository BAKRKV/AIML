import pickle
from flask import Flask, render_template, request
import numpy as np

app = Flask(__name__)
model = pickle.load(open('RFR.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/view')
def view():
    return render_template('predict.html')

@app.route('/about')
def about():
    return render_template('aboutus.html')

@app.route('/value1')
def value1():
    return render_template('predict.html')


@app.route('/predict1', methods=['POST', 'DELETE', 'GET'])
def predict1():
    AA = request.form['a']
    AA1 = request.form['b']
    AA2 = request.form['c']
    AA3 = request.form['d']
    AA4 = request.form['e']
    AA5 = request.form['f']
    AA6 = request.form['g']
    AA7 = request.form['h']
    AA8 = request.form['i']
    AA9 = request.form['j']
    AA10 = request.form['k']
    AA11 = request.form['l']
    AA12 = request.form['m']
    AA13 = request.form['n']
    AA14 = request.form['o']
    AA15 = request.form['p']
    AA16 = request.form['q']
    AA17 = request.form['r']
    AA18 = request.form['s']
    AA19 = request.form['t']
    AA20 = request.form['u']
    AA21 = request.form['v']
    AA22 = request.form['w']
    AA23 = request.form['x']


    int_features = [AA,AA1,AA2,AA3,AA4,AA5,AA6,AA7,AA8,AA9,AA10,AA11,AA12,AA13,AA14,AA15,AA16,AA17,AA18,AA19,AA20,AA21,AA22,AA23]
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)

    output = round(prediction[0], 2)
    return render_template('value.html', predict_text='The Predicted Price is {}'.format(output))


if __name__ == '__main__':
    app.run()
