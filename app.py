from flask import Flask, render_template, request
import pickle
import numpy as np
from sklearn.preprocessing import StandardScaler
app = Flask(__name__,template_folder='./templates',static_folder='./static')
model=pickle.load(open('modelws.pkl','rb'))

@app.route('/',methods=['GET'])
def home():
     return render_template('index.html')
@app.route("/predict", methods=['POST'])
def predict():
        features=[float(x) for x in request.form.values()]
        final_features=[np.array(features)]
        prediction=model.predict(final_features)[0]
        print(prediction)
        if prediction==1:
                return render_template('result1.html')
        else:
                return render_template('result2.html')
if __name__=="__main__":
    app.run(debug=True)