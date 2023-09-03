import numpy as np
from flask import Flask,request,jsonify,render_template
import pickle

app=Flask(__name__)
model=pickle.load(open('crop.pkl','rb'))


@app.route('/')
def home():
    return render_template('index1.html')  #html page here


@app.route('/predict1',methods=['POST'])
def predict1():
    
    int_features=[[int(x) for x in request.form.values()]]
    print(int_features)
    final_features=np.array(int_features)  
    prediction=model.predict(final_features)
    prediction=prediction.tolist()
    
    data = {
            "prediction" : prediction
        }
    return jsonify(data)
    
    # return render_template('index.html',prediction_text='Loan eligiblity is {}'.format(output))



if __name__=="__main__":
    app.run(debug=True)
    