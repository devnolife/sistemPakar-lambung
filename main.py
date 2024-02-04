# Machine learning classification
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
# For data manipulation
import pandas as pd
# To plot
import matplotlib.pyplot as plt
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split


df= pd.read_csv(r'./new_model.csv')

X = df.drop('Class', axis=1) #? ini untuk menghapus kolom class
y = df['Class'] #? ini untuk mengambil kolom class

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

model= LogisticRegression(max_iter=1500)

model.fit(X_train, y_train)




from flask import Flask, render_template, request

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/submit', methods=['POST'])
def submit():
    data = request.form


    testing = {
        "Bp" : [float(data["Bp"])], 
        "Sg" : [float(data["Sg"])], 
        "Al" : [float(data["Al"])], 
        "Su" : [float(data["Su"])], 
        "Rbc" : [float(data["Rbc"])], 
        "Bu" : [float(data["Bu"])], 
        "Sc" : [float(data["Sc"])], 
        "Sod" : [float(data["Sod"])], 
        "Pot" : [float(data["Pot"])], 
        "Hemo" : [float(data["Hemo"])], 
        "Wbcc" : [float(data["Wbcc"])], 
        "Rbcc" : [float(data["Rbcc"])], 
        "Htn" : [float(data["Htn"])]
        }
    testing = pd.DataFrame(testing)

    predict_test = model.predict(testing)
    # Process the data (in this case, just printing it)
    print(f"Received data: {data}")
    hasil = ""
    if predict_test[0] == 0 : 
        hasil = "Tidak Memiliki"
    else : 
        hasil = "Memiliki"

    return f"Anda {hasil} penyakit ginjal kronis"



if __name__ == '__main__':
    app.run(debug=True)

