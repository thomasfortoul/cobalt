import datetime
import time
import os
import logging
import numpy as np
import pandas as pd
import requests
import sys
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib
from statistics import mean, covariance, variance
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier


# Sommes commes features - avec medians.

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
            prev_block_df = pd.DataFrame(data=total_array, columns=['Price', 'Date', 'Short', 'Long', 'None', 'Block']) # past input files.
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


def getFeatureRow(input_file, buffer_size, max_buffer, hello_prix):
    column = input_file[hello_prix].to_frame().astype(float)

    current_row = max_buffer * buffer_size - 1  # 30*48-1 -> index of start point
    s_array = np.empty(max_buffer, dtype=float)
    a_array = np.empty(max_buffer, dtype=float)
    for i in range(max_buffer):  # 0 - 29
        tmp_sum = 0
        for k in range(buffer_size * (i + 1)):  # 0 - (48*max_buffer-1)
            tmp_sum += column.iloc[current_row - k] # going backwards in records, from current to 0 for last iteration

        s_array[i] = tmp_sum        # [sum-1*48, sum-2*48, ..., sum-29*48, sum-30*48]
        a_array[i] = s_array[i] / (buffer_size * (i + 1))

    total_array = np.array([a_array])
    return pd.DataFrame(data=total_array)


def getIndices(step, start, nombres_moyennes_mobiles):
    indices = np.empty(nombres_moyennes_mobiles, dtype=int)

    for i in range(nombres_moyennes_mobiles):
        indices[i] = start + i * step - 1
    return indices

def getFeatures(moyennes_storage, indices):
    moyenne_1 = 0
    tmp_list_2 = []
    num = 0
    for i in range(len(indices) - 1):  # iterate by number of moyennes (3-7, 3-11, 3-15, 7-11, 7-15, 11-15)
        tmp_list = []

        for k in range(len(indices) - 1 - i):
            column_list = []
            num += 1
            for index in range(len(moyennes_storage)):
                diff = moyennes_storage.iloc[index][indices[moyenne_1] - 1] - moyennes_storage.iloc[index][
                    indices[moyenne_1 + k + 1] - 1]

                column_list.append(np.sign(diff))

            new_pd = pd.DataFrame(column_list, columns=[num])

            tmp_list.append(new_pd)

        moyenne_1 += 1
        tmp_pd = pd.concat(tmp_list, axis=1)
        tmp_list_2.append(tmp_pd)
    final_pd = pd.concat(tmp_list_2, axis=1)
    final_pd.columns = final_pd.columns.astype(str)
    return final_pd

def getModels():
    url = 'https://weave.rs/getModels.php'
    params = {
    }
    try:
        # Send a GET request to the URL with parameters
        response = requests.get(url, params=params)

        # Check the response status code
        if response.status_code == requests.codes.ok:
            logging.info(f'getModels connection succesful, status code: {response.status_code}.')

            models = response.text.strip().split('\n')

            logging.info(f'Extracting values from getModels.')

            for model in models:
                try:
                    values = model.split('\t')
                    model_id = values[0]
                    asset_id = values[1]
                    moyennes_mobiles = int(values[2])
                    step = int(values[3])
                    start = int(values[4])
                    target = int(values[5])
                    model_filename = values[6]

                except Exception as e:
                    logging.error(f'Unsuccessful assignment of parameters from getModels '
                                  f'return list (moyennes mobiles, .., target, filename).')
                    return False

                logging.info(f'Successful getModels, returning values to original function.')

            return models

        else:
            logging.error(f'getModels failed with status code: {response.status_code}, {response.text}')
            return False

    except requests.exceptions.RequestException as e:
        logging.error(f'getModels error: {e}.')
        return False


# Adds the market open, adjusted price columns to the original input file
def finishInputFile(list_blocks): #price, date, hello, market open
    final_blocks = []

    for df in list_blocks:
        market_open = []
        adjusted_price = []
        counter = 1
        prev_price = 0
         #iterate through all price values to make the market open column (based on previous repeated prices)
        # then iterate through new market open column and fill in repeated prices with the new opening price - adjusted price
        for index, row in df.iterrows():
            price = row['Price']
            current = datetime.datetime.strptime(row['Date'], "%d/%m/%y %H:%M:%S")

    # if previous price is same as new price OR that the market day is actually closed -> since when bridging holes in data, price changes in non-market horus.
            if prev_price == price or current.weekday() > 5 or \
                    (current.hour < 9) or current.hour >= 16 or (current.hour == 9 and current.minute <= 30): # repeated price, market is not open.
                market_open.append(0)
                counter += 1

            else:
                market_open.append(1)
                new_list = [price] * counter
                adjusted_price.extend(new_list)
                counter = 1  # reset the counter

            prev_price = price # lagging price.

        if(counter > 1): # ended on a market closed
            market_open.append(1)
            new_list = [price] * counter
            adjusted_price.extend(new_list)


        new_df = pd.DataFrame({'Adjusted Price' : adjusted_price, 'Market Open' : market_open})
        df_list = [df, new_df] #concatenate adjusted price and original df.
        final_df = pd.concat(df_list, axis=1).dropna() # remove last NaN values.

        final_blocks.append(final_df)

    return final_blocks
""" past_monday = [0]
    first_open = [0]
    new_price = []
    prev_price = df.iloc[0]['Price']

    # making market open column.
    for index, row in df.iterrows():
        date = datetime.datetime.strptime(row['Date'], "%d/%m/%y %H:%M:%S")
        price = row['Price']

        if (date.hour == 9 and date.minute > 30):
            past_monday.append(price)
            first_open.append(price)

        else:
            past_monday.append(past_monday[index])
            first_open.append(first_open[index])

    next_monday = past_monday[336:]
    next_open = first_open[48:]

    next_open = next_open[0:len(next_monday)]
    new_market_open = df['Market Open'][0:len(next_monday)]
    new_date = df['Date'][0:len(next_monday)]
    new_price = df['Price'][0:len(next_monday)]

    new_df = pd.DataFrame(data = {'Next Monday': next_monday, 'Next Open': next_open, 'Market Open': new_market_open,
                                  'Date':new_date, 'Price' : new_price})"""

# '12','13','14','15','16','17','23','24','25','26','27','34','35','36','37','45','46','47','56','57','67','h12','h13','h14','h15','h16','h17','h23','h24','h25','h26','h27','h34','h35','h36','h37','h45','h46','h47','h56','h57','h67'
def read_Input_File(filename):
    df = pd.read_csv(filename)
    df = df.dropna(how='all')
    return df


# Returns training or testing set
def Extract_Testing_Training_Set(inputs, block_size, offset, training_set_tf):
    new_set = inputs.copy()
    new_set = inputs.drop(columns=['Market Open', 'Date', 'Adjusted Price', 'Price', 'Block'])
    comparator = 1

    if (training_set_tf):
        for index, row in new_set.iterrows():
            if ((index + offset) % (2 * block_size) // block_size == comparator):
                new_set = new_set.drop(index)  # remove all rows that correspond to the regimen (1/0)
        return new_set

    else:
        return new_set


# fits model and returns the model's predictions
def get_Predictions_Random(model_type, input_file):
    clf = model_type()
    input_file = input_file.applymap(str)
    target = input_file['Target'].to_frame()
    inputs = input_file.drop(columns='Target')

    X_train, X_test, Y_train, Y_test = train_test_split(inputs, target, test_size=0.8)

    # Train the model on the data
    clf.fit(X_train, Y_train.values.ravel())

    # Predict labels of unseen (test) data
    predictions = clf.predict(X_test)

    return accuracy_score(Y_test, predictions)


# fits model and returns the model's predictions
def get_Predictions_Random_2(model_type, input_file):
    clf = model_type()
    input_file = input_file.applymap(str)
    target = input_file['Target'].to_frame()
    inputs = input_file.drop(columns='Target')

    X_train, X_test, Y_train, Y_test = train_test_split(inputs, target, test_size=0.9)

    # Train the model on the data
    clf.fit(X_train, Y_train.values.ravel())

    # Predict labels of unseen (test) data
    predictions = clf.predict(X_test)

    return accuracy_score(Y_test, predictions)


# fits model and returns the model's predictions
def get_Prediction_Probabilities(model, testing_set):
    testing_inputs = testing_set.drop(columns='Target')

    probabilities = model.predict_proba(testing_inputs)

    return pd.DataFrame(probabilities, columns=['Probability: -1', 'Probability: 1'])


# fits model and returns the model's predictions
def get_Model(model_type, training_set, testing_set):
    model = model_type(random_state=2, max_depth = 3, max_features = 1.0)
    #model = model_type(random_state=1, max_iter=2000)
    training_target = training_set['Target']
    training_inputs = training_set.drop(columns='Target')
    trained_model = model.fit(training_inputs, training_target.values.ravel())

    return trained_model


# fits model and returns the model's predictions
def get_Predictions(model, testing_set):
    testing_inputs = testing_set.drop(columns='Target')
    predictions = model.predict(testing_inputs)

    return pd.DataFrame(predictions, columns=['Predictions'])


# returns the accuracy of the model as a decimal
def score_Predictions(final_file, predictions):
    testing_set = []
    predictions = []
    for index, row in final_file.iterrows():  # take only for regime = 0
        if (final_file['Régime'].iloc[index] == 0):
            testing_set.append(final_file['Target'].iloc[index])  # target value
            predictions.append(final_file['Predictions'].iloc[index])  # prediction

    t = {'Testing': testing_set}
    p = {'Predictions': predictions}
    testing_target = pd.DataFrame(data=t)
    final_predictions = pd.DataFrame(data=p)

    accuracy = accuracy_score(testing_target, final_predictions)
    return accuracy


# give block size and interval,
def get_Offsets(block_size, num_of_interval):
    # the function returns all possible offsets including 0 and the full block size
    offsets = []
    for i in range(num_of_interval - 1):
        offset = block_size // num_of_interval

        offsets.append(block_size - offset * (i + 1))
    offsets.append(0)
    return offsets[::-1]


# Makes the regime column based on block size and offset
def make_Regime(inputs, block_size, offset):
    regime = []
    for index, row in inputs.iterrows():
        if ((index + offset) % (2 * block_size) // block_size == 0):
            regime.append(1)  # remove all rows that correspond to the regimen (1/0)
        else:
            regime.append(0)

    r = {'Régime': regime}
    new_set = pd.DataFrame(data=r)
    df_list = [new_set, inputs]

    new_set = pd.concat(df_list, axis=1)

    # new_set = new_set.join(inputs[['Date','Price','Target','Adjusted Price', 'Market Open']])
    # file with: actual target, actual price, adjusted price, date, market open, régime.
    return new_set


# produces a file with actual target, actual price, adjusted price, date, market open, régime, input to backtest.
def get_Backtest_File(regimen_file, prediction, probabilities):
    data_frames = [regimen_file, prediction, probabilities]
    regimen_file = pd.concat(data_frames, axis=1)
    return regimen_file


# Returns the percentage growth given two float inputs
def percentage_Growth(current_price, previous_price):
    return (float(current_price) - float(previous_price)) / (float(previous_price))


def backtest(backtest_file, short, long, transaction_cost, threshold, download_model, backtest_filename, share):
    backtest_file['Predictions'] = backtest_file['Predictions'].astype(float)
    backtest_file['Régime'] = backtest_file['Régime'].astype(float)
    backtest_file['Market Open'] = backtest_file['Market Open'].astype(float)
    backtest_file = backtest_file.applymap(str)

    backtest_file['Price'] = backtest_file['Price'].astype(float)
    backtest_file['Adjusted Price'] = backtest_file['Adjusted Price'].astype(float)  # convert prices to floats
    backtest_file['Probability: -1'] = backtest_file['Probability: -1'].astype(float)
    backtest_file['Probability: 1'] = backtest_file['Probability: 1'].astype(float)
    backtest_file['Block'] = backtest_file['Block'].astype(int) # convert blocks to integers.

    # Separate backtest file into block dataframes for discontinuous backtesting.
    backtest_blocks = []
    unique_blocks = backtest_file['Block'].unique() # get unique blocks.
    for unique_block in unique_blocks:
        backtest_blocks.append(backtest_file[backtest_file['Block'] == unique_block])

    count = 0
    for block in backtest_blocks:
        count += 1

# initialize counters, backtest value = 100, cost - same for all block dataframes.
    num_total = 0
    num_long = 0
    num_short = 0
    percent_long_array = [0]
    percent_short_array = [0]
    percent_total_array = [0]

    cost = 1 - transaction_cost / 100
    backtest = 100.0
    growth = 0.0

    backtest_col = []
    percent_growth_col = []
    previous_price_col = []
    current_pos_col = []
    share_stored = 1 - share

    backtest_file_subset = backtest_file[["Target", "Price", "Date", "Block", "Adjusted Price", "Market Open", "Predictions", "Probability: -1", "Probability: 1"]]
    backtest_file_subset = backtest_file_subset._append(backtest_file_subset.iloc[-1])
    old_date = backtest_file_subset['Date'].iloc[-1]
    old_date = pd.to_datetime(old_date, format='%d/%m/%y %H:%M:%S')
    backtest_file_subset.loc[-1, ("Date")] = (old_date + datetime.timedelta(minutes=30)).strftime("%d/%m/%y %H:%M:%S")
    backtest_file_subset = backtest_file_subset.reset_index(drop=True)

    for backtest_block in backtest_blocks:
         # reset previous price, current price and positions to start 'new' backtest (in new block dataframe)
        previous_price = backtest_file['Price'].iloc[0]
        current_price = previous_price

        previous_pos = '0.0'
        current_pos = '0.0'

        for index in range(len(backtest_block)):

            current_pred = backtest_block['Predictions'].iloc[index]  # current prediction 1.0/-1.0 strings!
            probability = max(backtest_block['Probability: 1'].iloc[index], backtest_block['Probability: -1'].iloc[index])

            if (backtest_block['Market Open'].iloc[index] == '1.0'):  # if market is open
                if (probability >= threshold):  # if probability is within threshold range, can execute trades.

                    if (previous_pos == '0.0'):  # first time in unseen regime when market open, take position but no trade
                        current_price = backtest_block['Price'].iloc[index]  # update bought price
                        current_pos = current_pred

                    else:  # was in position short or long
                        if (previous_pos != current_pred):  # change in position, prediction not alligned with position.
                            current_price = backtest_block['Price'].iloc[index]  # current market price
                            growth = percentage_Growth(current_price, previous_price)

                            if (previous_pos == '1.0'):  # long position
                                if (long): # if can execute long trades
                                    num_total += 1
                                    num_long += 1
                                    percent_total_array.append(growth)
                                    percent_long_array.append(growth)
                                    backtest = (backtest * share_stored) + backtest * share * (1 + growth) * cost

                                current_pos = '-1.0'

                            else:  # was in short position
                                if (short):  # execute only if short boolean is true.
                                    num_total += 1
                                    num_short += 1
                                    percent_total_array.append(-growth)
                                    percent_short_array.append(-growth)
                                    backtest = (backtest * share_stored) + backtest * share * (1 - growth) * cost
                                current_pos = '1.0'

            backtest_col.append(backtest)
            percent_growth_col.append(growth)
            previous_price_col.append(previous_price)
            current_pos_col.append(current_pos)

            growth = percentage_Growth(current_price, previous_price)
            previous_pos = current_pos  # follow up previous values and reset growth
            previous_price = current_price

        current_price = backtest_block['Price'].iloc[index]  # update bought price
        previous_price = backtest_block['Price'].iloc[index-1]  # update bought price
        growth = percentage_Growth(current_price, previous_price)

        if (previous_pos == '1.0'):  # long position
            if (long):
                num_total += 1
                num_long += 1
                percent_total_array.append(growth)
                percent_long_array.append(growth)
                backtest = (backtest * share_stored) + backtest * share * (1 + growth) * cost

        else:  # was in short position
            if (short):  # execute only if short boolean is true.
                num_total += 1
                num_short += 1
                percent_total_array.append(-growth)
                percent_short_array.append(-growth)
                backtest = (backtest * share_stored) + backtest * share * (1 - growth) * cost

    backtest_extra_details = pd.DataFrame({'Backtest': backtest_col, 'Current Position': current_pos_col,
                                           'Previous Price': previous_price_col,
                                           'Percentage Growth': percent_growth_col})

    backtest_col.append(backtest)
    percent_growth_col.append(growth)
    previous_price_col.append(previous_price)
    current_pos_col.append(current_pos)

    backtest_extra_details = pd.DataFrame({'Backtest': backtest_col, 'Current Position': current_pos_col,
                                           'Previous Price': previous_price_col, 'Percentage Growth': percent_growth_col}).reset_index(drop=True)

    backtest_df = pd.concat([backtest_file_subset, backtest_extra_details], axis=1)

    [benchmark_max_drawdown, max_drawdown, annualized_return, annualized_std, sharpe_ratio, benchmark_annualized_return,
            benchmark_annualized_std, benchmark_sharpe_ratio, alpha, beta, annualized_excess_return,
            annualized_excess_return_std, backtest_df, monthly_backtest] = calculateBacktestStats(backtest_df, 0.001)

    if(download_model):
        logging.info(f'Downloading backtest file and monthly returns for chosen model to: {backtest_filename}, '
                     f'{backtest_filename}-monthly,  {backtest_filename}-benchmark.')
        backtest_df.to_csv(f'{backtest_filename}.csv')
        monthly_backtest.to_csv(f'{backtest_filename}-monthly.csv')

        benchmark_df = pd.DataFrame({'Benchmark Maximum Drawdown': [benchmark_max_drawdown],
                                     'Benchmark Annualized Return': [benchmark_annualized_return],
                                     'Benchmark Annualized Stdev': [benchmark_annualized_std],
                                     'Benchmark Sharpe Ratio': [benchmark_sharpe_ratio],
                                     'Maximum Drawdown': [max_drawdown],
                                     'Annualized Return': [annualized_return],
                                     'Annualized Stdev': [annualized_std],
                                     'Annualized Sharpe Ratio': [sharpe_ratio],
                                     'Annualized Excess Return': [annualized_excess_return],
                                     'Annualized Excess Return Stdev': [annualized_excess_return_std],
                                     'Alpha': [alpha], 'Beta': [beta]})

        benchmark_df.to_csv(f'{backtest_filename}-benchmark.csv')

    return [backtest - 100, num_short, num_long, num_total, mean(percent_long_array) * 100, mean(percent_short_array)*100,
            mean(percent_total_array)*100, max_drawdown, annualized_return, annualized_std,
            sharpe_ratio, alpha, beta]

def calculateBacktestStats(backtest_df, risk_free_rate):
    backtest_df['Date'] = pd.to_datetime(backtest_df['Date'], format='%d/%m/%y %H:%M:%S')

    backtest_df.set_index('Date', inplace=True) # make date the index to group by month, isolate only relevant columns
    monthly_backtest = backtest_df[['Price', 'Block', 'Backtest']].groupby(pd.Grouper(freq="M")).first().dropna()

    # Calculate the annualized monthly return, annualized standard deviation (price and backtest)
    monthly_backtest['Monthly Returns'] = monthly_backtest['Backtest'].pct_change()
    monthly_backtest['Benchmark Monthly Returns'] = monthly_backtest['Price'].pct_change()

    annualized_return = (1 + monthly_backtest['Monthly Returns'].dropna().mean()) ** 12 - 1
    annualized_std = monthly_backtest['Monthly Returns'].dropna().std() * np.sqrt(12)
    benchmark_annualized_return = (1 + monthly_backtest['Benchmark Monthly Returns'].dropna().mean()) ** 12 - 1
    benchmark_annualized_std = monthly_backtest['Benchmark Monthly Returns'].dropna().std() * np.sqrt(12)

    monthly_backtest['Excess Monthly Returns'] = monthly_backtest['Monthly Returns'] - monthly_backtest['Benchmark Monthly Returns']
    annualized_excess_return = (1 + monthly_backtest['Excess Monthly Returns'].dropna().mean()) ** 12 - 1
    annualized_excess_return_std = monthly_backtest['Excess Monthly Returns'].dropna().std() * np.sqrt(12)

    #Calculate sharpe ratios.
    sharpe_ratio = 0.0
    benchmark_sharpe_ratio = 0.0
    if(annualized_std != 0.0):
        sharpe_ratio = (annualized_return - risk_free_rate) / annualized_std
    if (benchmark_annualized_std != 0.0):
        benchmark_sharpe_ratio = (benchmark_annualized_return - risk_free_rate) / benchmark_annualized_std

    # Calculate MDD - Backtest
    backtest_df['Running Max'] = backtest_df['Backtest'].cummax()
    backtest_df['Drawdown'] = 100*(backtest_df['Running Max'] - backtest_df['Backtest'])/ backtest_df['Running Max']
    max_drawdown = backtest_df['Drawdown'].max()

    # Calculate MDD - Benchmark
    backtest_df['Benchmark Running Max'] = backtest_df['Price'].cummax()
    backtest_df['Benchmark Drawdown'] = 100 * (backtest_df['Benchmark Running Max'] - backtest_df['Price']) / backtest_df['Benchmark Running Max']
    benchmark_max_drawdown = backtest_df['Benchmark Drawdown'].max()

    # Find linear regression portfolio and benchmark returns.
    valid_indices = monthly_backtest['Monthly Returns'].notnull()

    beta_portfolio = monthly_backtest['Monthly Returns'][valid_indices]
    beta_benchmark = monthly_backtest['Benchmark Monthly Returns'][valid_indices]

    beta = 0.0
    if(variance(beta_benchmark) != 0):
        beta = covariance(beta_portfolio, beta_benchmark)/variance(beta_benchmark)
    alpha = annualized_return - risk_free_rate - beta * (benchmark_annualized_return - risk_free_rate)

    return [benchmark_max_drawdown, max_drawdown, annualized_return*100, annualized_std, sharpe_ratio, benchmark_annualized_return*100,
            benchmark_annualized_std, benchmark_sharpe_ratio, alpha, beta, annualized_excess_return*100,
            annualized_excess_return_std, backtest_df, monthly_backtest]

def all_backtests(input_file, short, long, seen, unseen, block_size, offset, predictions, transaction_cost,
                  probabilities, threshold, training_set, download_model, backtest_filename, share):
    # Testing: Seen/Unseen Data
    if (seen & unseen):  # TEST ALL DATA
        regime = make_Regime(input_file, len(input_file), len(input_file))  # regime is always 0, execute always

    elif (seen & ~unseen):  # TEST ONLY SEEN
        regime = make_Regime(input_file, block_size, block_size)  # invert blocks

    elif (~seen & unseen):  # TEST ONLY UNSEEN
        regime = make_Regime(input_file, block_size, offset)  # regular

    else:  # test nothing
        return ([0, 0])
    # Backtesting and scoring.
    final_file = get_Backtest_File(regime, predictions, probabilities)

    final_file_index = final_file.index

    accuracy = score_Predictions(final_file, predictions)


    [backtest_value, num_short, num_long, num_total, percent_long, percent_short, percent_total, max_drawdown,
     annualized_return, annualized_std, sharpe_ratio, alpha, beta] \
        = backtest(final_file, short, long, transaction_cost, threshold, download_model, backtest_filename, share)

    return pd.Series(
        [accuracy, backtest_value, num_short, num_long, num_total, percent_long, percent_short, percent_total,
         max_drawdown, annualized_return, annualized_std, sharpe_ratio, alpha, beta])

def get_moyennes_mobiles_indices(step, start, nombres_moyennes_mobiles):
    indices = np.empty(nombres_moyennes_mobiles, dtype=int)

    for i in range(nombres_moyennes_mobiles):
        indices[i] = start + i * step - 1
    return indices


def calculate_Sentiments(moyennes_storage, indices):
    moyenne_1 = 0
    tmp_list_2 = []
    total_list = []
    num = 0
    tmp_list = []

    for i in range(len(indices) - 1):  # iterate by number of moyennes (3-7, 3-11, 3-15, 7-11, 7-15, 11-15)
        tmp_list = []
        for k in range(len(indices) - 1 - i):
            column_list = []
            num += 1
            for index in range(len(moyennes_storage)):
                diff = moyennes_storage.iloc[index][indices[moyenne_1]] - moyennes_storage.iloc[index][
                    indices[moyenne_1 + k + 1]]

                column_list.append(np.sign(diff))

            new_pd = pd.DataFrame(column_list, columns=[num])

            tmp_list.append(new_pd)

        moyenne_1 += 1
        tmp_pd = pd.concat(tmp_list, axis=1)
        tmp_list_2.append(tmp_pd)
    final_pd = pd.concat(tmp_list_2, axis=1)
    final_pd.columns = final_pd.columns.astype(str)
    return final_pd


def calculate_Max_Step(step, start, nombre_moyennes_mobiles):
    return 48 * (start + (nombre_moyennes_mobiles * step))


def get_moyenne_mobiles(input_file, buffer_size, max_buffer, none_weeks):
    logging.info('Calculating moyennes mobiles.')
    hello = input_file['Hello'].to_frame().astype(float)
    none = input_file['Hello_None'].to_frame().astype(float)

    current_row = max_buffer * buffer_size - 1  # 30*48-1 -> index of start point
    s_array = np.empty(max_buffer, dtype=float)
    s_array_none = np.empty(len(none_weeks), dtype=float)
    counter = 0
    for i in range(max_buffer):  # 0 - 29
        tmp_sum = 0
        if (i/7 in none_weeks):
            tmp_sum_2 = 0
            for k in range(buffer_size * (i + 1)):  # 0 - (48*max_buffer-1)
                tmp_sum += hello.iloc[current_row - k]  # going backwards in records, from current to 0 for last iteration
                tmp_sum_2 += none.iloc[current_row - k]
            s_array[i] = tmp_sum  # [sum-1*48, sum-2*48, ..., sum-29*48, sum-30*48]
            s_array_none[counter] = tmp_sum_2
            counter += 1

        else:
            for k in range(buffer_size * (i + 1)):  # 0 - (48*max_buffer-1)
                tmp_sum += hello.iloc[current_row - k]  # going backwards in records, from current to 0 for last iteration

            s_array[i] = tmp_sum  # [sum-1*48, sum-2*48, ..., sum-29*48, sum-30*48]

    total_array = np.array([s_array])
    total_array_none = np.array([s_array_none])
    # update sum array after each index iteration
    for index in range(len(hello) - current_row):
        counter = 0
        for i in range(max_buffer):  # 0 - 29
            if (i/7 in none_weeks):
                s_array_none[counter] = s_array_none[counter] + none.iloc[current_row + index] - none.iloc[
                    current_row + index - (i + 1) * buffer_size + 1]
                counter += 1
            s_array[i] = s_array[i] + hello.iloc[current_row + index] - hello.iloc[
                current_row + index - (i + 1) * buffer_size + 1]  # remove oldest record, add most recent

        total_array = np.r_[total_array, [s_array]]
        total_array_none = np.r_[total_array_none, [s_array_none]]

    logging.info('Moyennes mobiles finished calculation.')
    return pd.DataFrame(data=total_array), pd.DataFrame(data=total_array_none)

def calculate_sentiment_lag(input_file, buffer_size, max_buffer):
    return max_buffer * buffer_size - 1  # to remove from top


def calculate_target_lag(input_file, target_size, max_target):
    return target_size * max_target - 1  # to remove from bottom


def get_targets(input_file, target_size, max_target):
    logging.info('Calculating all targets. Iterating through adjusted price column.')
    column = input_file['Adjusted Price'].to_frame().astype(float)

    t_array = np.empty(max_target, dtype=float)
    total_array = np.array([t_array])
    cols = []
    for i in range(max_target):
        cols.append('Target')

    prev_sign = 1
    for index in range(len(column) - max_target * target_size):
        for i in range(max_target):  # 0 - 9 targets +1-> 1-10
            sign = np.sign(float(column.iloc[index + (i + 1) * target_size]) - float(column.iloc[index]))
            t_array[i] = sign

            if (sign == 0):
                t_array[i] = prev_sign

            else:
                prev_sign = sign

        total_array = np.r_[total_array, [t_array]]

    final = pd.DataFrame(data=total_array, columns=cols).tail(-1)  # remove first row.
    logging.info(f'Target file configured. Max target = {max_target}.')
    return final


# Tests block size and offset size variations
# master_Function('filename', target_size, max_target, buffer_size, max_buffer, ['Hello', 'Price'],
# [number of moyennes mobiles], [start 1, start 2], [step 1, step 2], [block_size_1, block_size_2], num_intervals,
# [seen True, False], [Unseen True, False], [model_1, model_2])

def master_Function(input_files, target_size, max_target, target_list, buffer_size, max_buffer, hello_prix,
                    moyennes_mobiles, starts, steps, blocks, num_of_intervals, seen_input, unseen_input, models,
                    training_sizes, transaction_costs, thresholds, download_model, model_filename, medians, none_weeks,
                    shares):
    num_iterations = len(thresholds) * len(transaction_costs) * len(training_sizes) * len(target_list) * len(
        hello_prix) * len(moyennes_mobiles) * len(starts) * len(steps) * len(blocks) * (num_of_intervals) * len(
        seen_input) * len(unseen_input) * len(models) * len(medians) * len(none_weeks)

    num_combinations = len(moyennes_mobiles) * len(starts) * len(steps)

    total_data = pd.DataFrame(columns=['Model', 'None Week', 'Median',
                                       'Training Size', 'Moyennes Mobiles', 'Start', 'Step', 'Target',
                                       'Accuracy', 'Threshold', 'Backtest', 'Transaction Cost Percentage',
                                       'Short Trades', 'Long Trades',
                                       'Total Trades', 'Avg Percentage Short', 'Avg Percentage Long',
                                       'Avg Percentage Total', 'Maximum Drawdown', 'Annualized Return',
                                       'Annualized Stdev', 'Alpha', 'Beta'])

    data_1 = pd.DataFrame(columns=['Model', 'None Week', 'Median',
                                       'Training Size', 'Moyennes Mobiles', 'Start', 'Step', 'Target',
                                       'Accuracy', 'Threshold', 'Backtest', 'Transaction Cost Percentage',
                                       'Short Trades', 'Long Trades',
                                       'Total Trades', 'Avg Percentage Short', 'Avg Percentage Long',
                                       'Avg Percentage Total', 'Maximum Drawdown', 'Annualized Return',
                                       'Annualized Stdev', 'Alpha', 'Beta'])

    logging.info(f'Number of iterations (number of backtests): {num_iterations}.')
    logging.info(f'Number of combinations per median (moyennes, starts, steps): {num_combinations}.')
    logging.info(f'Number of medians: {len(medians)}.')
    logging.info(f'Number of none weeks: {len(none_weeks)}.')

    # initialize all arrays.
    none_week_array = []
    threshold_array = []
    nombres_moyennes_mobiles_array = []
    demarrage_array = []
    step_array = []
    target_array = []
    model_name = []
    accuracy_array = []
    backtest_array = []
    training_size_array = []
    num_short_array = []
    num_long_array = []
    num_total_array = []
    percent_long_array = []
    percent_short_array = []
    percent_total_array = []
    transaction_cost_array = []
    mdd_array = []
    return_array = []
    std_array = []
    sharpe_array = []
    alpha_array = []
    beta_array = []
    medians_array = []
    shares_array = []

    num = []
    numb = 0

    block_targets = []
    block_moyennes = []
    block_inputs = []

    input_block_df = input_files[0] # targets and total_block_df need 1 iteration.
    logging.info(f'Getting targets and total data.')
    for block_num in range(len(input_block_df)):  # iterate through the block dataframes, make separate targets/moyennes.
        logging.info(f'Block number: {block_num}')
        current_block_df = input_block_df[block_num]

        top_lag = calculate_sentiment_lag(current_block_df, buffer_size, max_buffer)
        bottom_lag = calculate_target_lag(current_block_df, target_size, max_target)

        # Calculate target and moyennes mobiles storages.
        targets = get_targets(current_block_df, target_size, max_target)

        # fix sizing.
        current_block_df = current_block_df.tail(-top_lag)
        current_block_df = current_block_df.head(-bottom_lag)

        targets = targets.tail(-top_lag + 1)

        current_block_df = current_block_df.reset_index(drop=True)
        current_block_df = current_block_df.drop('Hello', axis=1)  # no need for hello anymore.
        current_block_df = current_block_df.drop('Hello_None', axis=1)  # no need for hello_none anymore.
        targets = targets.reset_index(drop=True)

        block_targets.append(targets)  # get list of target and moyennes df per block.
        logging.info(
            f'Target and feature files configured for block dataframe number {block_num} Going to next block.')

        # rejoin all targets and moyennes mobiles to extract columns index.
        if (block_num == 0):
            total_target = targets
            total_block_df = current_block_df
        else:
            total_target = total_target._append(targets).reset_index(drop=True)  # append all block dataframes
            total_block_df = total_block_df._append(current_block_df).reset_index(drop=True)

    total_block_df = total_block_df.drop(columns=['Short', 'Long', 'None'])
    logging.info(f'Target and main input file configured. Generating feature files. (Median Hello and Median Hello None)')

    median_sentiments = []
    median_sentiments_none = []
    for median_index in range(len(medians)):
        input_block_df = input_files[median_index]
        median = medians[median_index]
        logging.info(f'Median {median}: Calculating target and sentiment files.')
        for block_num in range(len(input_block_df)):  # iterate through the block dataframes, make separate targets/moyennes.
            logging.info(f'Block number: {block_num}')
            current_block_df = input_block_df[block_num]

            # Calculate target and moyennes mobiles storages.
            moyennes_storage, moyennes_storage_none = get_moyenne_mobiles(current_block_df, buffer_size, max_buffer, none_weeks)

            # fix sizing.
            moyennes_storage = moyennes_storage.head(-bottom_lag - 1)
            moyennes_storage_none = moyennes_storage_none.head(-bottom_lag - 1)
            moyennes_storage = moyennes_storage.reset_index(drop=True)
            moyennes_storage_none = moyennes_storage_none.reset_index(drop=True)

            block_moyennes.append(moyennes_storage)
            logging.info(
                f'Target and feature files configured for block dataframe number {block_num} Going to next block.')

            # rejoin all targets and moyennes mobiles to extract columns index.
            if (block_num == 0):
                total_moyennes = moyennes_storage
                total_moyennes_none = moyennes_storage_none
            else:
                total_moyennes = total_moyennes._append(moyennes_storage).reset_index(drop=True)
                total_moyennes_none = total_moyennes_none._append(moyennes_storage).reset_index(drop=True)

        median_sentiments.append(total_moyennes)
        median_sentiments_none.append(total_moyennes_none)

# build data frame for all training data, per median (10 medians, 10 dataframes for sums)

    # then, per combination, average/start/step/none/target, train a model based on median -> then training size
    # train 50 models per combination.
    logging.info(f'Median {median}: Target and feature files configured, iterating through features.')

    # loop through all combinations.
    for moyenne_mobile in moyennes_mobiles:
        for start in starts:
            if (moyenne_mobile == 2):
                step1 = 24
            elif (moyenne_mobile == 3):
                step1 = 12
            elif (moyenne_mobile == 4):
                step1 = 8
            else:
                step1 = int(30 / (moyenne_mobile))

            for step in steps:
                #step = step1
                logging.info(f'Getting features for: Median: {median}, Moyennes Mobiles: {moyenne_mobile},  '
                             f'Start: {start},  Step: {step}.')
                indices = get_moyennes_mobiles_indices(step, start, moyenne_mobile)
                sentiments = total_moyennes.iloc[:, indices]
                sentiments.columns = sentiments.columns.astype(str)

                for t in target_list:
                    target = total_target.iloc[:, t - 1] # extract target columns by index - 1 large df

                    for none_counter in range(len(none_weeks)):
                        none_week = none_weeks[none_counter]

                        logging.info(f'Testing new combination: None : {none_week}, Target: {t}, '
                                     f'Average: {moyenne_mobile}, Start: {start},  Step: {step}.')

                        for median_index in range(len(medians)):
                            # START OF TESTING FOR UNIQUE COMBINATION

                            combination_invalid = False
                            median = medians[median_index]

                            #Get none column for this median
                            total_moyennes_none = median_sentiments_none[median_index] #index by median
                            none_col = total_moyennes_none.iloc[:, none_counter] # index by none
                            none_column = pd.DataFrame({'Hello_None': none_col})

                            #Get sentiments for this median
                            total_moyennes = median_sentiments[median_index]
                            sentiments = total_moyennes.iloc[:, indices]
                            sentiments.columns = sentiments.columns.astype(str)

                            # Make final dataframe (continuous) for model training.
                            data_frames = [target, total_block_df, sentiments, none_column]
                            input_file = pd.concat(data_frames, axis=1)
#                            input_file.to_csv('SP500-rfc-data.csv')

                            # START OF TESTING FOR UNIQUE COMBINATION
                            for model in models:  # iterate by model
                                if combination_invalid:
                                    break
                                for block in blocks:
                                    if combination_invalid:
                                        break
                                    offsets = get_Offsets(block, num_of_intervals)
                                    for offset in offsets:
                                        if combination_invalid:
                                            break
                                        for training_size in training_sizes:
                                            if combination_invalid:
                                                break

                                            for share in shares:
                                                if combination_invalid:
                                                    break

                                                for i in range(25): # 50 different testing sets.
                                                    # Training, testing sets, prediction calculations
                                                    testing_set = Extract_Testing_Training_Set(input_file, block, offset, False)
                                                    training_set = testing_set.sample(frac=training_size, random_state=i)

                                                    trained_model = get_Model(model, training_set, testing_set)
                                                    predictions = get_Predictions(trained_model, testing_set)
                                                    probabilities = get_Prediction_Probabilities(trained_model, testing_set)

                                                    for transaction_cost in transaction_costs:
                                                        if combination_invalid:
                                                            break
                                                        for threshold in thresholds:
                                                            # backtesting.
                                                            backtest_basefilename = f'{model_filename}-backtest'
                                                            backtest_filename = os.path.join(
                                                                os.path.join(os.path.join(os.path.normpath(os.getcwd() + os.sep),
                                                                                          'BacktestFiles'), backtest_basefilename))
                                                            res = all_backtests(input_file, True, True, True, True, block, offset,
                                                                                predictions, transaction_cost, probabilities,
                                                                                threshold, training_set, download_model,
                                                                                backtest_filename, share)

                                                            if download_model:
                                                                logging.info(
                                                                    f'Downloading model: {model_filename}.  None : {none_week}, '
                                                                    f'Target: {t}, Average: {moyenne_mobile}, '
                                                                    f'Start: {start},  Step: {step}.')
                                                                new_filename = os.path.join(
                                                                    os.path.join(
                                                                        os.path.join(os.path.normpath(os.getcwd() + os.sep),
                                                                                     'ModelFiles'), model_filename))

                                                                joblib.dump(trained_model, new_filename)
                                                                break

                                                            else:
                                                                if res[1] <= 0:  # if backtest <= 0, skip this combination.
                                                                    logging.info(
                                                                        f'Backtest = {round(res[1], 3)}. Continuing to next combination.')
                                                                    combination_invalid = True

                                                                    break


                                                            # append all relevant values for final table.
                                                            none_week_array.append(none_week)
                                                            medians_array.append(median)
                                                            threshold_array.append(threshold)
                                                            nombres_moyennes_mobiles_array.append(moyenne_mobile)
                                                            demarrage_array.append(start)
                                                            step_array.append(step)
                                                            target_array.append(t)
                                                            model_name.append(model)
                                                            accuracy_array.append(res[0])
                                                            backtest_array.append(res[1])
                                                            num_short_array.append(res[2])
                                                            num_long_array.append(res[3])
                                                            num_total_array.append(res[4])
                                                            percent_long_array.append(res[5])
                                                            percent_short_array.append(res[6])
                                                            percent_total_array.append(res[7])
                                                            mdd_array.append(res[8])
                                                            return_array.append(res[9])
                                                            std_array.append(res[10])
                                                            sharpe_array.append(res[11])
                                                            alpha_array.append(res[12])
                                                            beta_array.append(res[13])
                                                            shares_array.append(share)

                                                            training_size_array.append(training_size)
                                                            transaction_cost_array.append(transaction_cost)

                                                            num.append(numb)
                                                            numb += 1  # row number.

                            data = {'Model': model_name,
                                    'Share': shares_array,
                                    'None Week': none_week_array,
                                    'Median': medians_array,
                                    'Training Size': training_size_array,
                                    'Moyennes Mobiles': nombres_moyennes_mobiles_array,
                                    'Start': demarrage_array,
                                    'Step': step_array, 'Target': target_array,
                                    'Accuracy': accuracy_array, 'Threshold': threshold_array,
                                    'Backtest': backtest_array,
                                    'Transaction Cost Percentage': transaction_cost_array,
                                    'Short Trades': num_short_array,
                                    'Long Trades': num_long_array,
                                    'Total Trades': num_total_array,
                                    'Avg Percentage Short': percent_short_array,
                                    'Avg Percentage Long': percent_long_array,
                                    'Avg Percentage Total': percent_total_array,
                                    'Maximum Drawdown': mdd_array,
                                    'Annualized Return': return_array,
                                    'Annualized Stdev': std_array, 'Alpha': alpha_array,
                                    'Beta': beta_array}

                            data = pd.DataFrame(data, index = num)

                            logging.info(f'Number of completed iterations: {numb}.')

                            #Numb = number of iteration per unique combination (a combination is the unique combination of none_week,
                            #    moyennes, start, step, target, none

                            logging.info(f'Combination: {model_filename}.  None : {none_week}, Target: {t}, '
                                         f'Average: {moyenne_mobile}, Start: {start},  Step: {step}.')
                            total_data = total_data._append(data, ignore_index=True)

                            # reset all arrays.
                            none_week_array = []
                            medians_array = []
                            threshold_array = []
                            nombres_moyennes_mobiles_array = []
                            demarrage_array = []
                            step_array = []
                            target_array = []
                            model_name = []
                            accuracy_array = []
                            backtest_array = []
                            training_size_array = []
                            num_short_array = []
                            num_long_array = []
                            num_total_array = []
                            percent_long_array = []
                            percent_short_array = []
                            percent_total_array = []
                            transaction_cost_array = []
                            mdd_array = []
                            return_array = []
                            std_array = []
                            sharpe_array = []
                            alpha_array = []
                            beta_array = []
                            shares_array = []

                            num = []
                            numb = 0
    logging.info(f'Finished all iterations, returning final dataframe.')
    return total_data


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

def selectModel(model_filename, summary_filename):
    training_details = read_Input_File(os.path.join(os.path.join(os.path.normpath(os.getcwd() + os.sep), 'BacktestFiles'), model_filename))

    new = training_details.groupby(['Median','None Week','Moyennes Mobiles', 'Start', 'Step', 'Target']). \
        agg(
        {'Median': 'first','None Week': 'first', 'Moyennes Mobiles': 'first', 'Start': 'first', 'Step': 'first', 'Target': 'first', 'Backtest': ['mean', 'std', 'min'],
         'Total Trades': ['mean'], 'Avg Percentage Total': ['mean'], 'Maximum Drawdown': ['mean'], 'Annualized Return': ['mean'],
         'Annualized Stdev': ['mean'], 'Alpha': ['mean'], 'Beta': ['mean']})

    sharpe = []

    for index, row in new.iterrows():
        if(row['Backtest']['std'] == 0 or row['Maximum Drawdown']['mean'] == 0):
            continue

        ratio = row['Backtest']['mean']*row['Backtest']['min']/(row['Backtest']['std']*row['Maximum Drawdown']['mean'])
        sharpe.append(ratio)

    s = {'Sharpe Ratio': sharpe}
    sharpe_df = pd.DataFrame(data=s)
    df_list = [new.reset_index(drop=True), sharpe_df.reset_index(drop=True)]

    final_df = pd.concat(df_list, axis=1)
    final_df = final_df.sort_values(by='Sharpe Ratio', ascending=False)

    new_filename = os.path.join(os.path.join(os.path.normpath(os.getcwd() + os.sep), 'BacktestFiles'), summary_filename)

    final_df.to_csv(new_filename)

    combination = final_df.iloc[0, 0:6]  # best combination
    return combination

def postModelFile(model_filename, model_id):
    # Define the URL
    url = 'https://weave.rs/postModelFile.php'

    # Define the parameters
    params = {
        'modelid': model_id
    }

    file_path = os.path.join(os.path.join(os.path.normpath(os.getcwd() + os.sep), 'ModelFiles'),model_filename)
    files = {'file': open(file_path, 'rb')}

    try:
        response = requests.post(url, files=files, params=params)
        if response.status_code == requests.codes.ok:
            logging.info(f"postModelFile uploaded successfully. Response: {response.text}. "
                        f"Status code: {response.status_code}")
            return True

        else:
            logging.error(f"postModelFile connection failed. Status code: {response.status_code}. {response.text}")
            return False

    except requests.exceptions.RequestException as err:
        logging.error(f"postModelFile error occurred: {err}")
        return False


def postModel(model_id, combination, threshold):
    # Define the URL
    url = 'https://weave.rs/postModel.php'

    # Define the parameters
    params = {
        'modelid': model_id,
        'median': int(combination[0]),
        'none': int(combination[1]),
        'average': int(combination[2]),
        'start' :int(combination[3]),
        'step': int(combination[4]),
        'target': int(combination[5]),
        'threshold': threshold
    }

    try:
        response = requests.post(url, params=params)
        if response.status_code == requests.codes.ok:
            logging.info(f"postModels uploaded successfully. Response: {response.text}. "
                        f"Status code: {response.status_code}")
            return True

        else:
            logging.error(f"postModels connection failed. Status code: {response.status_code}. {response.text}")
            return False

    except requests.exceptions.RequestException as err:
        logging.error(f"postModels error occurred: {err}")
        return False

def trainModel(input_files, input_basefilename, training_size, threshold, medians, none_weeks):
    # Filename value extraction
    value_list = input_basefilename.split('-')

    model_name = value_list[0]
    cur_time = value_list[1][0:-4] # remove .csv
    model_filename = f'{model_name}-{cur_time}'

    logging.info(f'Iterating through combinations of training features. Calling master_Function.')
    start_time = time.time()

    # def master_Function(filename, target_size, max_target, target_list, buffer_size, max_buffer, hello_prix,
    #                     moyennes_mobiles, starts, steps, blocks, num_of_intervals, seen_input, unseen_input, models,
    #                     training_sizes, transaction_costs, thresholds, download_model, model_filename, medians):

    # Iterate through all combinations to find optimal.
    #
    # training_details = master_Function(input_files, 48, 4, [2, 3], 48, 25, ['Hello'], [5],
    #                                [1], [3, 4], [1000], 1,
    #                                [False], [True], [MLPClassifier], [training_size], [0.2], [0.8, 0.85, 0.9], False,
    #                                model_filename, [medians], [none_weeks], [shares])

#Per combination: 24 (3 trainings, 2 transaction costs, 3 thresholds) 5*2*4*4*4 = 768 combinations = 11520 total
    training_details = master_Function(input_files, 48, 7, [5,6], 48, 42, ['Hello'], [1], [25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40],
                                       [5], [1000], 1,
                                    [False], [True], [RandomForestClassifier], [0.15], [0.2], [0.5],
                                      False, model_filename, medians, none_weeks, [1])

    end_time = time.time()

    elapsed = time.strftime("%Hh%Mm%Ss", time.gmtime(end_time - start_time))

    logging.info(f'Training over all features finished. Elapsed time: {elapsed}.')

    testing_filename = f'{model_name}-tests2-RandomForest-{cur_time}.csv'
    new_filename = os.path.join(os.path.join(os.path.normpath(os.getcwd() + os.sep), 'BacktestFiles'), testing_filename)
    training_details.to_csv(new_filename)

    logging.info(f'Test details saved in {testing_filename}. Selecting strongest model.')

    # Model Selection
    summary_filename = f'{model_name}-summary2-{cur_time}.csv'

    combination = selectModel(testing_filename, summary_filename)

    logging.info(f'Model selection finished. Summary file: {summary_filename}.')
    logging.info(f'Optimal combination: {combination}. Re-training model for download.')


    median = int(combination[0])
    none_week = int(combination[1])
    moyennes = int(combination[2])
    start = int(combination[3])
    step = int(combination[4])
    target = int(combination[5])

    max_moyennes = start + step*moyennes + 1

    # Download model
    final_model = master_Function(input_files, 48, target+1, [target], 48, max_moyennes, ['Hello'],
                                  [moyennes], [start], [step],
                                  [1000], 1, [False], [True], [RandomForestClassifier], [training_size], [0.2], [threshold], True,
                                  model_filename, [median], [none_week], [1])

    # Model Selection
    final_model_filename = f'{model_name}-final-model-test-{cur_time}.csv'
    final_filename = os.path.join(os.path.join(os.path.normpath(os.getcwd() + os.sep), 'BacktestFiles'), final_model_filename)
    final_model.to_csv(final_filename)

    logging.info(f'Model downloaded, filename: {final_filename}')

    logging.info(f'Training finished, exiting program.')

    logging.info(f'Test results and model: {summary_filename}, {testing_filename}, {model_filename}, {final_filename}.')

    return model_filename, combination


def prepareLogging():
    # Logging Configuration
    logname = 'Trainer_Log_File.log'

    log_dir0 = os.path.join(os.path.normpath(os.getcwd() + os.sep), 'TrainerLogs')
    log_dir1 = os.path.join(os.path.normpath(os.getcwd() + os.sep), 'ModelFiles')
    log_dir2 = os.path.join(os.path.normpath(os.getcwd() + os.sep), 'BacktestFiles')
    log_dir3 = os.path.join(os.path.normpath(os.getcwd() + os.sep), 'InputFiles')
    log_fname = os.path.join(log_dir0, logname)

    logdirs=[log_dir0, log_dir1, log_dir2, log_dir3]

    for logdir in logdirs:
        log_dir_exists = os.path.exists(logdir)
        if (not log_dir_exists):  # make directory if it doesn't exits
            os.mkdir(logdir.split('/')[-1])

    logging.basicConfig(handlers=[logging.FileHandler(log_fname), logging.StreamHandler()], encoding='utf-8',
                        format=f'%(asctime)s :%(levelname)s : %(message)s : Line %(lineno)d',
                        datefmt='%d/%m/%Y %I:%M %p', level=logging.INFO)


def get_median_hello(input_file, median, max_median):
    # Find averages and initialize arrays.
    average_none = input_file.loc[:, 'None'].mean()
    average_hello = input_file.loc[:, 'Short'].mean() / (input_file.loc[:, 'Short'].mean() + average_none)
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

def configureInputFile(asset_id, model_name, medians):
    output = getAssetPrices(asset_id)
    if len(output) == 0:
        logging.error(f'getAssetPrices failed, input file cannot be configured, going to next model')
    else:
        for out in output:
            if(out.empty):
                logging.error(f'getAssetPrices failed, input file cannot be configured, going to next model')

    logging.info(f'getAssetPrices finished, configuring additional input file columns (Market Open, Adjusted Price).')

    input_file = finishInputFile(output)  # without median calculations, remove first m lines.

    input_filenames = []
    input_files = []
    #Get time.
    now = datetime.datetime.now().strftime("%Y%m%d")

    input_basefilename = f'{model_name}-{now}.csv'

    for median in medians:
        median_files = []
        median_input_filenames = []
        for block_num in range(len(input_file)):
            input_filename = f'{model_name}-input-{median}-{block_num}-{now}.csv'
            new_filename = os.path.join(os.path.join(os.path.normpath(os.getcwd() + os.sep), 'InputFiles'), input_filename)

            tmp_file = input_file[block_num]
            tmp_file['Short'] = pd.to_numeric(tmp_file['Short'])
            tmp_file['None'] = pd.to_numeric(tmp_file['None'])
            tmp_file['Long'] = pd.to_numeric(tmp_file['Long'])
            tmp_file = get_median_hello(tmp_file, median, max(medians))
            tmp_file = tmp_file.tail(-max(medians)).reset_index(drop=True) # Truncate for maximum median.

            median_files.append(tmp_file)
            median_input_filenames.append(input_filename)
            logging.info(f'Input file, {input_filename} for model training finished.')

        input_files.append(median_files)
        input_filenames.append(median_input_filenames) # add the input filename, one per median.

        logging.info(f'Input files for median: {median} finished.')
    logging.info(f'All input files finished: Medians: {medians}.')

    return input_files, input_basefilename, input_filenames

def mainFunction(argv):
    prepareLogging()
    logging.info('Log file and folders configured. Verifying inputs - server name.')
    threshold = 0.5
    training_size = 0.15
    medians = [96, 144]
    none_weeks = [4]

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
    if(models == False):
        logging.error(f'getModels unsuccessful, cannot continue program, exiting.')
        return

    logging.info(f'getModels successful, iterating through models.')
    counter_1 = 0
    for model in models:
        counter_1 += 1
        if(counter_1 == 2):
            continue
        values = model.split('\t')
        model_id = values[0]
        asset_id = values[1]
        model_name = values[6]

        logging.info(f'Model: {model_id}, Asset Id: {asset_id}, configuring input file.')
        input_files, input_basefilename, input_filenames = configureInputFile(asset_id, model_name, medians)

        if (len(input_files) == 0):  # checking we have atleast 1 block dataframe
            logging.error(f'configureInputFile failed, going to next model.')
        else:
            for median_array in input_files:
                for block_input in median_array:
                    if block_input.empty:  # Checking dataframes per block is not empty
                        logging.error(f'configureInputFile failed, going to next model.')

        logging.info(f' ------- Model: {model_id} - {model_name}, Asset ID: {asset_id}, configuring training models.  ------- ')
        model_filename, combination = trainModel(input_files, input_basefilename, training_size, threshold, medians, none_weeks)

        logging.info(f'Model: {model_id} - {model_name}, Asset ID: {asset_id}, finished model training, posting details.')
        # if(not postModel(model_id, combination, threshold)):
        #    logging.error(f'postModel error. Going to next model ID.')
        #
        # if(not postModelFile(model_filename, model_id)):
        #     logging.error(f'postModelFile error. Going to next model ID.')

        logging.info(f' ========= Model: {model_id} - {model_name}, Asset ID: {asset_id}, finished. Next model.  ========= ')

    logging.info(f' ========= All model training and posting finished. Trainer Program Finished.  ========= ')

if __name__ == '__main__':
    #mainFunction(sys.argv)
    #getAssetPrices('14')
    #selectModel('HELLO_SP500_1-tests-20230613.csv', 'summary.csv')
    mainFunction([0,'Trainer_1'])
    #postModel('HELLO-SP500_20230612', 1, [7,3,4,5])
