from flask import Flask, render_template, request
import joblib
import numpy as np

model = joblib.load('model.pkl')

app = Flask(__name__)

@app.route('/')
def main():
    return render_template('home.html')


@app.route('/predict', methods=['POST'])
def home():
    data1 = float(request.form['a'])
    data2 = float(request.form['b'])
    data3 = float(request.form['c'])
    data4 = float(request.form['d'])
    data5 = float(request.form['e'])
    data6 = float(request.form['f'])
    data7 = float(request.form['g'])
    data8 = float(request.form['h'])
    data9 = int(request.form['i'])
    data10 = int(request.form['j'])
    data11 = int(request.form['k'])
    data12 = int(request.form['l'])
    data13 = int(request.form['m'])
    data14 = float(request.form['n'])
    data15 = float(request.form['o'])
    arr = np.array([[data1, data2, data3, data4, data5, data6, data7, data8, data9, data10, data11, data12, data13, data14, data15]])
    pred = model.predict(arr)
    return render_template('after.html', data=np.round(pred[0]))

if __name__ == "__main__":
    app.run(debug=True)