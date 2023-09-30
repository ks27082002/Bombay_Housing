from flask import Flask, request, jsonify
import util

app = Flask(__name__)

@app.route('/get_location_names', methods=['GET'])
def get_location_names():
    response = jsonify({
        'locations': util.get_location_names()
    })
    response.headers.add('Access-Control-Allow-Origin', '*')

    return response
@app.route('/predict_home_price', methods=['GET', 'POST'])
def predict_home_price():
    total_area = float(request.form['total_area'])
    region = request.form['region']
    bhk = int(request.form['bhk'])
    # bath = int(request.form['bath'])

    response = jsonify({
        'estimated_price': util.get_estimated_price(region, total_area, bhk)
    })
    response.headers.add('Access-Control-Allow-Origin', '*')

    return response

if __name__ == "__main__":
    print("Starting Python Flask Server For Home Price Prediction...")
    util.load_saved_artifacts()
    app.run()