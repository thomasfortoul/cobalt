import datetime
import os
import logging
import joblib
import numpy as np
import pandas as pd
import requests
import sys

def getModels():
    url = 'https://weave.rs/getModels.php'
    params = {
    }
    try:
        # Send a GET request to the URL with parameters
        response = requests.get(url, params=params)

        # Check the response status code
        if response.status_code == requests.codes.ok:
            lines = response.text.strip().split('\n')
            return lines

        else:
            logging.error(f'getModels failed with status code: {response.status_code}')
            return False
    except requests.exceptions.RequestException as e:
        logging.error(f'getModels error. {e}')
        return False

def getAssetPrices(asset_id):
    url = 'https://weave.rs/getAssetPrices.php'
    params = {
        'assetid': asset_id,
    }
    try:
        # Send a GET request to the URL with parameters
        response = requests.get(url, params=params)
        counter_1 = 0
        empty_df = pd.DataFrame({'Empty': []})
        # Check the response status code
        if response.status_code == requests.codes.ok:
            logging.info(f'getAssetPrices connection established, extracting values.')
            # Split the response content by lines
            lines = response.text.strip().split('\n')

            # Initialize previous batch ID and next batch ID variables
            new_data = np.empty(6, dtype=str)
            total_array = np.array([new_data])
            final_blocks = []

            counter = 0
            prev_date = None
            num_missing = 0
            prices = []
            next_date = datetime.datetime.now()

            difference = datetime.timedelta(minutes=30)  # 30 minutes

            # Iterate over the lines in reverse order with a limit
            for line in lines:
                recent_missing = 0
                # Split the line by tabs
                values = line.split('\t')

                if(len(values) != 7): # Check number of values of input.
                    logging.error(f'getAssetPrices error, {len(values)} values per line, expected 7.')
                    return empty_df

            # Extract values.
                try:
                    date = values[1]
                    date = datetime.datetime.strptime(date, "%Y%m%d%H%M") # making datetime object
                    price = float(values[2])
                    short = float(values[3])
                    long = float(values[4])
                    none = float(values[5])
                    block = int(values[6])

                except Exception as e:
                    logging.error(f'getAssetPrices error, extracting values. Exception details: {e}')
                    return empty_df

                if(counter != 0 and prev_block != block):
                    logging.info(f'New block: {block}. Verifying previous block data and starting new dataframe.')
                    prev_block_df = checkBlock(total_array, prices, num_missing)
                    final_blocks.append(prev_block_df)

                    new_data = np.empty(6, dtype=str) #reset arrays for new block
                    total_array = np.array([new_data])
                    prev_block = block
                    counter = 0
                    num_missing = 0

            #Check range of hello values.
                hello_values = [short, long, none]

                # previous date is the record before.
                # 'new date' is always 30 minutes ahead to the 'date'
                if ((date.minute != 15) and (date.minute != 45)):
                    logging.info(f'getAssetPrices. Duplicating previous value. Date at HH:{date.minute}, expected either HH:15 or HH:45.')
                    total_array = np.r_[total_array, [prev_data]]
                    counter += 1
                    continue

                if (price == 0.0):
                    logging.error(
                        f'getAssetPrices error, price out of range: {price}. Expected > 0. Date: {date}. Taking previous price but current hello values.')
                    price = prev_price

                # Extract the price
                prices.append(float(price))

                for hello_value in hello_values:
                    if((hello_value < 0.0) or (hello_value > 20.0)):
                        logging.error(f'getAssetPrices error, Hello values out of range: {hello_value}. Expected between 0, 20. {date}. Duplicating previous value.')
                        total_array = np.r_[total_array, [prev_data]]
                        counter += 1
                        continue

           #     hello = np.sign(float(short) / (float(short) + float(none)) - 0.26)

                # if the previous date does not match the new date, we have a gap in the data
                if (counter != 0):  # continue until data is caught up, keep taking last value.
                    if(next_date != date):
                        while (next_date != date):
                            if (prev_block != block):
                                logging.info(
                                    f'New block: {block}. Verifying previous block data and starting new dataframe.')
                                prev_block_df = checkBlock(total_array, prices, num_missing)
                                final_block_df = final_block_df._append(prev_block_df, ignore_index=True)

                                new_data = np.empty(4, dtype=str)  # reset arrays for new block
                                total_array = np.array([new_data])
                                prev_block = block
                                counter = 0
                                num_missing = 0
                                break

                            logging.info(f'getAssetPrices missing value. Duplicating value at {prev_date}.')

                            # append previous data (most recent)
                            new_data = np.array([prev_price, next_date.strftime("%d/%m/%y %H:%M:%S"), short, long, none, prev_block])
                            total_array = np.r_[total_array, [new_data]]
                            next_date = next_date + difference

                            num_missing += 1
                            recent_missing += 1
                            counter += 1

                            counter_1 += 1
                            continue


                    new_data = np.array([price, date.strftime("%d/%m/%y %H:%M:%S"), short, long, none, block])
                    total_array = np.r_[total_array, [new_data]]

                next_date = date + difference

                prev_date = date
                prev_price = price
             #   prev_hello = hello
                prev_block = block
                prev_data = new_data
                counter += 1  # found the line, so increase counter.

            last_date = next_date - difference
            total_array[-1][1] = last_date.strftime("%d/%m/%y %H:%M:%S")
            logging.info(f'getAssetPrices all blocks finished. Last date = {last_date.strftime("%d/%m/%y %H:%M:%S")}')

            prev_block_df = checkBlock(total_array, prices, num_missing)
            final_blocks.append(prev_block_df)

            for index in range(len(final_blocks)):
                final_block = final_blocks[index]
                final_block['Short'] = final_block['Short'].astype(float)
                final_block['Long'] = final_block['Long'].astype(float)
                final_block['None'] = final_block['None'].astype(float)
                final_block['Hello'] = np.sign(final_block['Short'] / (final_block['Short'] + final_block['None']))

            return final_blocks
        else:
            logging.error(f'getAssetPrice failed with status code: {response.status_code}')
            return empty_df

    except requests.exceptions.RequestException as e:
        logging.error(f'getAssetPrice error, {e}.')
        return empty_df


def get_median_hello(input_file, median, max_median):
    # Find averages and initialize arrays.
    average_none = input_file.loc[:, 'None'].mean()
    average_hello = input_file.loc[:, 'Short'].mean() / (input_file.loc[:, 'Short'].mean() + average_none)
    logging.info(f'Average None = {average_none}, Average Hello = {average_hello}.')
    hello_values = [0]*(max_median-1)
    hello_n_values = [0]*(max_median-1)

    # Get initial subsets to get median
    short_subset = input_file['Short'].iloc[max_median-median:max_median].tolist()
    none_subset = input_file['None'].iloc[max_median-median:max_median].tolist()

    # First calculation.
    short_median = np.median(short_subset)
    none_median = np.median(none_subset)
    hello_value = short_median / (short_median + none_median) - average_hello
    hello_n_value = none_median - average_none
    hello_values.append(np.sign(hello_value))
    hello_n_values.append(np.sign(hello_n_value))

    for i in range(len(input_file) - max_median):
        # Resize subsets to always include 'median' values. Remove first row and add next row
        index_to_replace = i % median
        short_subset[index_to_replace] = input_file['Short'].iloc[max_median + i]
        none_subset[index_to_replace] = input_file['None'].iloc[max_median + i]

        # Get median of each subset.
        short_median = np.median(short_subset)
        none_median = np.median(none_subset)

        # Calculate hello values.
        hello_value = short_median / (short_median + none_median) - average_hello
        hello_n_value = none_median - average_none

        # Append values
        hello_values.append(np.sign(hello_value))
        hello_n_values.append(np.sign(hello_n_value))

    input_file['Hello'] = pd.DataFrame({'Hello': hello_values})
    input_file['Hello_None'] = pd.DataFrame({'Hello': hello_n_values})

    return input_file


def checkBlock(total_array, prices, num_missing):
    empty_df = pd.DataFrame({'Empty': []})

    # final check for remaining price values. (upper and lower bounds of previous 1 week average)
    stdev = np.std(prices)
    mean = np.mean(prices)
    lower = mean - 6 * stdev
    upper = mean + 6 * stdev
    for cur_price in prices:
        if ((cur_price > upper) or (cur_price < lower)):
            logging.error(f'getAssetPrices error. Price out of range: {cur_price}, expected between {lower}, {upper}.')
            return empty_df

    output = pd.DataFrame(data=total_array, columns=['Price', 'Date', 'Short', 'Long', 'None', 'Block']).tail(-1).reset_index(drop=True)  # remove first empty row
    if (num_missing == 0):  # if no missing values.
        logging.info(f'No getAssetPrices missing values. Successful execution of getAssetPrices.')
        return output

    logging.info(f'getAssetPrices {num_missing} missing values. Successful execution of getAssetPrices.')
    return output


def getIndices(step, start, nombres_moyennes_mobiles):
    indices = []

    for i in range(nombres_moyennes_mobiles):
        indices.append(start + i * step)
    return indices


def get_moyenne_mobiles(input_file, buffer_size, max_buffer, indices):
    logging.info('Calculating moyennes mobiles.')
    hello = input_file['Hello'].to_frame().astype(float)
    none = input_file['Hello_None'].to_frame().astype(float)

    data = {}

    counter = 0
    for i in range(len(indices)):  # 0-3
        index = indices[i]# 1, 8, 15, 22
        tmp_sum = 0

        num_records = index*48
        last_n_records = hello.tail(num_records)

        if(i == len(indices)-1):
            last_n_records = none.tail(num_records)
            data['Hello_None'] = last_n_records.sum().iloc[0]

        else:
            data[str(indices[i] - 1)] = last_n_records.sum().iloc[0]

    logging.info('Moyennes mobiles finished calculation.')
    return pd.DataFrame([data])

def getPrediction(testing_data, model_filename):
    model = joblib.load(model_filename)
    logging.info(f'Model loaded.')
    prediction = model.predict(testing_data)
    probability = np.round(np.max(model.predict_proba(testing_data)), 2)
    return [prediction, probability]

def getRobots(server_name):
    url = 'https://weave.rs/getRobots.php'
    params = {
        'server': server_name,  # Specify the desired limit here
    }

    try:
        # Send a GET request to the URL with parameters
        response = requests.get(url, params=params)

        # Check the response status code
        if response.status_code == requests.codes.ok:
            # Split the response content by lines
            values = response.text.strip().split('\t')
            logging.info(f'getRobots connection successful, status code: {response.status_code}.')

            if(len(values) != 1):
                logging.error(f'getRobots error, {len(values)} values per line, expected 1. Response: {response.text}')
                return False

            try:
                robot_id = int(values[0])

            except Exception as e:
                logging.error(f'getRobots error, variable assignment. Exception details: {e}')
                return False

            logging.info(f'getRobots data extraction successful. RobotId: {robot_id}.')
            return robot_id

        else:
            logging.error(f'getRobots failed with status code: {response.status_code}, Response: {response.text}')
            return False

    except requests.exceptions.RequestException as e:
        logging.error(f'getRobots error: {e}')
        return False


def main_func(argv):
    # Logging Configuration
    logname = 'Oracle_Log_File.log'

    log_dir = os.path.join(os.path.normpath(os.getcwd() + os.sep), 'Oracle_Logs')
    log_fname = os.path.join(log_dir, logname)

    log_dir_exists = os.path.exists(log_dir)
    if (not log_dir_exists):  # make model if does not work
        os.mkdir('Oracle_Logs')

    logging.basicConfig(handlers=[logging.FileHandler(log_fname), logging.StreamHandler()], encoding='utf-8',
                        format=f'%(asctime)s :%(levelname)s : %(message)s : Line %(lineno)d',
                        datefmt='%d/%m/%Y %I:%M %p', level=logging.INFO)

    logging.info('Log file configured. Verifying inputs - server name.')

    try:
        server_name = argv[1]

    except Exception as e:
        logging.error(f'Input error when extracting server name, finishing program.')
        return
    logging.info(f'Server name: {server_name}, calling getRobots.')

    robot_id = getRobots(server_name)
    if (robot_id == False):
        logging.error(f'getRobots failed, exiting program.')

    logging.info(f'getRobots finished, robot_id = {robot_id}. Calling getModels.')

    models = getModels()
    if (models == False):
        logging.error(f'getModels failed. Exiting program.')

    logging.info(f'getModels finished. Iterating through models.')
    counter = 0
    for model in models:
        try:
            values = model.split('\t')
            model_id = values[0]
            asset_id = values[1]
            moyennes_mobiles = int(values[2])
            start = int(values[3])
            step = int(values[4])
            target = int(values[5])
            model_filename = values[6]
            threshold = float(values[7])
            median = int(values[8])
            none_week = int(values[9])

        except Exception as e:
            logging.error(f'Unsuccessful assignment of parameters from getAssetPrices. Skipping this model.')
            continue

        logging.info(
            f'Successful getModels, extracted all relevant values. Model {model_id}. Calculating limit and max_buffer.')

        # Calculate maximum buffer and limit for feature engineering
        buffer_size = 48
        max_buffer = start + step * moyennes_mobiles

        logging.info(f'Calling getAssetPrices.')

        # Feature engineering and extract batch ID
        output = getAssetPrices(asset_id)

        if len(output) == 0:
            logging.error(f'getAssetPrices failed, input file cannot be configured, going to next model')
        else:
            for out in output:
                if (out.empty):
                    logging.error(f'getAssetPrices failed, input file cannot be configured, going to next model')
        output = output[-1]
        logging.info(
            f'Successful getAssetPrices. Number of records extracted: {len(output)}. Model {model_id}. Extracting batchId.')
        try:
            batch_id = output['Date'].iloc[-1]
            new_id = datetime.datetime.strptime(batch_id, "%d/%m/%y %H:%M:%S")  # making datetime object
            batch_id = new_id.strftime("%Y%m%d%H%M")

        except Exception as e:
            logging.error(f'BatchId extraction unsuccessful. Model: {model_id}. Skipping this model.')
            continue

        logging.info(f'Successful batch Id extraction. Batch Id = {batch_id}')

        try:
            logging.info(f'Details: Start {start}, Step {step}, Number of Hello Features: {moyennes_mobiles}')
            output['Short'] = pd.to_numeric(output['Short'])
            output['None'] = pd.to_numeric(output['None'])
            output['Long'] = pd.to_numeric(output['Long'])
            tmp_file = get_median_hello(output, median, median)
            tmp_file = tmp_file.tail(-median).reset_index(drop=True)  # Truncate for maximum median.

            tmp_file.to_csv('tmp_file.csv')

            indices = getIndices(step, start, moyennes_mobiles)
            indices.append(none_week * 7)
            sentiments = get_moyenne_mobiles(tmp_file, 48, max_buffer, indices)
            print(sentiments)

        except Exception as e:
            logging.error(f'Unsuccessful feature engineering. Model: {model_id}. Skipping this model.')
            continue

        logging.info(f'Successful feature engineering. Model {model_id}. Downloading model file: {model_filename}.')

        # Download model
        model_dir = os.path.join(os.path.normpath(os.getcwd() + os.sep), 'Models')
        model_fname = os.path.join(model_dir, model_filename)

        model_exists = os.path.exists(model_fname)
        model_dir_exists = os.path.exists(model_dir)

        if (not model_dir_exists):  # make model directory if does not exist yet.
            logging.info(f'Missing model directory, creating directory.')
            os.mkdir('Models')

        if (not model_exists):  # if file does not exist, downlaoad it.
            logging.info(f'Missing model file, {model_filename}. Fetching model with getModelFile. Model {model_id}.')
            try:
               getModelFile(model_id, model_fname)
            except Exception as e:
               logging.error(
                   f'Unsuccesful model downloading with getModelFile, model {model_id}. Skipping this model.')
               continue

        # Get predictions and probability
        logging.info(f'Successful access to model file. Fetching predictions. Model {model_id}')

        try:
            predictions = getPrediction(sentiments, model_fname)
        except Exception as e:
            logging.error(f'Unsuccessful getPrediction. Model: {model_id}. Skipping this model. {e}')
            continue

        try:
            prediction = predictions[0]
            probability = predictions[1]
        except Exception as e:
            logging.error(f'getPrediction return value error. Model {model_id}. Skipping this model.')
            continue

        # Check probability and prediction values.
        if ((probability < 0.0) or (probability > 1.0)):
            logging.error(
                f'getPrediction error. Probability: {probability}. Expected between 0, 1. Skipping this model.')
            continue

        if ((prediction != 1.00) and (prediction != -1.00)):
            logging.error(
                f'getPrediction error. Prediction: {prediction}. Expected either -1.00 or 1.00. Skipping this model.')
            continue

        # Post predictions and probability
        logging.info(f'Successful getPrediction. Posting predictions. Model {model_id}')
        # post_prediction = postPrediction(prediction[0], probability, batch_id, model_id, robot_id)
        # if (post_prediction == False):
        #    logging.error(f'postPrediction error. Continuing to next model..')
        #    continue



def getModelFile(model_id, model_filename):
    url = 'https://weave.rs/getModelFile.php'
    params = {
        'modelid': model_id
    }
    try:
        # Send a GET request to the URL with parameters
        response = requests.get(url, params=params)

        # Check the response status code
        if response.status_code == requests.codes.ok:

            open(model_filename, 'wb').write(response.content)

            return model_filename

        else:
            logging.error(f'getModelFile failed with status code: {response.status_code}')
            return False

    except requests.exceptions.RequestException as e:
        logging.error(f'getModelFile error. {e}')
        return False

def postPrediction(prediction, probability, batch_id, model_id, robot_id):
    # Define the URL
    url = 'https://weave.rs/postPrediction.php'

    # Define the parameters
    params = {
        'modelid': model_id,
        'batchid': batch_id,
        'robotid' : robot_id,
        'prediction': prediction,
        'probability': probability,
    }
    try:
    # Send the POST request
        response = requests.post(url, params=params)
        if response.status_code == requests.codes.ok:
            logging.info(f'Successful postPrediction, Model Id: {model_id}, BatchId:  {batch_id}, Prediction: {prediction}, Probability: {probability}')
            return True
        else:
            logging.error(f'postPrediction failed with status code: {response.status_code}')
            return False

    except requests.exceptions.RequestException as e:
        logging.error(f'postPredictions error: {e}.')
        return False

if __name__ == '__main__':
    #main_func(sys.argv)
    main_func([0,"Oracle_1"])
    #prediction.py model_id robot_id

