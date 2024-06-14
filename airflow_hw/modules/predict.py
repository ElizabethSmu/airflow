import dill
from datetime import datetime
import pandas as pd
import os
import json

path = os.environ.get('PROJECT_PATH', '../')

def predict():
    with open(f'{path}/data/models/cars_pipe_{datetime.now().strftime("%Y%m%d%H%M")}.pkl', 'rb') as file:
        model = dill.load(file)

    json_dir = f'{path}/data/test'
    output_csv = f'{path}/data/predictions/pred{datetime.now().strftime("%Y%m%d%H%M")}.csv'
    json_files = [f for f in os.listdir(json_dir) if f.endswith('.json')]
    results_df = pd.DataFrame(columns=['file', 'prediction'])

    for json_file in json_files:
        file_path = os.path.join(json_dir, json_file)

        with open(file_path, 'r') as f:
            data = json.load(f)
        df = pd.DataFrame([data])
        prediction = model.predict(df)

        results_df = results_df._append({'file': json_file, 'prediction': prediction[0]}, ignore_index=True)

    results_df.to_csv(output_csv, index=False)

if __name__ == '__main__':
    predict()
