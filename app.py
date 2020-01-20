from flask import Flask, request, jsonify
import json
import pickle
import numpy as np

app = Flask(__name__)

__locations = None
__model = None
__data_columns= None


@app.route('/get_location_names', methods=['GET'])
def get_location_names():
    response = jsonify({
        'locations': get_location_names()
    })
    response.headers.add('Access-Control-Allow-Origin', '*')

    return response

@app.route('/predict_home_price', methods=['POST'])
def predict_home_price():
    total_sqft = float(request.form['total_sqft'])
    location = request.form['location']
    bhk = int(request.form['bhk'])
    bath = int(request.form['bath'])
    print('total_sqft', total_sqft)
    print('location', location)
    print('bhk', bhk)
    print('bath', bath)

    response = jsonify({
        'estimated_price': get_estimate_price(location,bhk,bath,total_sqft)
    })
    response.headers.add('Access-Control-Allow-Origin', '*')

    return response

def get_location_names():
    return __locations



def get_estimate_price(location,bhk,bath,sqft):
    try:
        location_index = __data_columns.index(location.lower())
    except:
        location_index = -1

    x = np.zeros(len(__data_columns))
    x[0] = sqft
    x[1]= bath
    x[2] = bhk

    if location_index >=0:
        x[location_index] = 1

    return round(__model.predict([x])[0], 2)


def load_saved_artifacts():
    print("loading saved artifacts.....START")
    global __data_columns
    global __locations

    with open("./common.json","r") as f:
        __data_columns = json.load(f)['data_columns']
        __locations = __data_columns[3:]  # first 3 columns are sqft, bath, bhk

    global __model
    if __model is None:
        with open('./banglore_home_price_model.pkl', 'rb') as f:
            __model = pickle.load(f)
    print("loading saved artifacts...done")

def get_data_columns():
    return __data_columns


if __name__ == "__main__":
    app.run()
