import pickle
import json
import numpy as np

__locations = None
__data_columns = None
__model = None

def get_estimated_price(region,area,bhk):
    try:
        loc_index = __data_columns.index(region.lower())
    except:
        loc_index = -1

    x = np.zeros(len(__data_columns))
    x[0] = bhk
    x[1] = area

    if loc_index>=0:
        x[loc_index] = 1

    return round(__model.predict([x])[0],2)
def load_saved_artifacts():
    print("loading saved artifacts...start")
    global __data_columns
    global __locations

    with open("./artifacts/columns.json", "r") as f:
        __data_columns = json.load(f)['data_columns']
        __locations = __data_columns[2:]  # first 3 columns are sqft, bath, bhk

    global __model
    if __model is None:
        with open('./artifacts/bombay_home_prices_model.pickle', 'rb') as f:
            __model = pickle.load(f)
    print("loading saved artifacts...done")



def get_location_names():
    return __locations

if __name__ == '__main__':
    load_saved_artifacts()
    print(get_location_names())
    print(get_estimated_price('Virar',1000, 2))
    print(get_estimated_price('Andheri West',1000, 2))
    print(get_estimated_price('Vashi',1000, 2)) # other location
    print(get_estimated_price('Ambernath East', 1000, 2))  # other location
