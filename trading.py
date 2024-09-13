import logging
import requests
import datetime
from ibapi.client import EClient
from ibapi.wrapper import EWrapper
from ibapi.contract import Contract
from ibapi.order import Order
import threading
import time
import pandas as pd
import os
import sys
import subprocess

class TradingApp(EWrapper, EClient):
    def __init__(self):
        EClient.__init__(self, self)
        self.stoploss_to_close = False
        self.short_stoploss = False
        self.long_stoploss = False
        self.shortprice= 0.0
        self.longprice = 0.0
        self.dailyPnL = 0.0
        self.unrealizedPnL = 0.0
        self.realizedPnL = 0.0
        self.valuePnL = 0.0
        self.avg_cost = 1.0
        self.market_price = 0.0
        self.asset_id = ''
        self.connected = False
        self.stoploss_error = ''
        self.thread = None
        self.open = None
        self.symbol = ''
        self.quantity = 0
        self.account_number = ''
        self.connection_details = ["127.0.0.1", 4002, 1] #["127.0.0.1", 7497, 2]  - IP, Socket, ClientId
        self.logger = None
        self.models = []
        self.account = ''      #DU... or U...
        self.predictions = ['', '', '', '', ''] # 'predictionid'  'batchid'  'modelid'  prediction probability
        self.accountmodels = [0, '', 0, 1.0, '', '', '', '', 0.0, 0.0]
        self.event = '' #accountmodelid, modelid, active, threshold, shortassetid, longassetid, shortsymbol, longsymbol, stoploss, share
        self.price = 0.0
        self.operation = [0, 0, '', 0, '', '', 0, '']#[None, None, None, None, None, None, None, None]# [0, 0, '', 0, '', '', 0, ''] # accountmodelid,predictionid,batchid,robotid,type,symbol,quantity,status,event
        self.robot_id = 0
        self.wait_iterations = 30
        self.waiting_time = 1
        self.order_waiting_time = 10
        self.status = 0
        self.order_status = 0
        self.account_updates = pd.DataFrame(columns=["Key", "Value", "Currency", "AccountName"])
        self.account_portfolio = pd.DataFrame(columns=["Symbol", "SecType", "Exchange", "Position", "MarketPrice",
                                                     "MarketValue", "AverageCost", "UnrealizedPNL", "RealizedPNL",
                                                     "AccountName"])
        self.contract_details = pd.DataFrame(columns=["ReqId", "Symbol", "secType", "Currency", "Exchange", "Hours"])
        self.order_df = pd.DataFrame(columns=['PermId', 'ClientId', 'OrderId',
                                              'Account', 'Symbol', 'SecType',
                                              'Exchange', 'Action', 'OrderType',
                                              'TotalQty', 'CashQty', 'LmtPrice',
                                              'AuxPrice', 'Status'])
    def reset_contract_details(self):
        self.contract_details = pd.DataFrame(columns=["ReqId", "Symbol", "secType", "Currency", "Exchange", "Hours"])

    def reset_order_df(self):
        self.order_df = pd.DataFrame(columns=['PermId', 'ClientId', 'OrderId',
                                              'Account', 'Symbol', 'SecType',
                                              'Exchange', 'Action', 'OrderType',
                                              'TotalQty', 'CashQty', 'LmtPrice',
                                              'AuxPrice', 'Status'])

    def reset_account_updates(self):
        self.account_updates = pd.DataFrame(columns=["Key", "Value", "Currency", "AccountName"])
        self.account_portfolio = pd.DataFrame(columns=["Symbol", "SecType", "Exchange", "Position", "MarketPrice",
                                                       "MarketValue", "AverageCost", "UnrealizedPNL", "RealizedPNL",
                                                       "AccountName"])

    def contractDetails(self, reqId, contractDetails):
        dictionary = {"ReqId": reqId, "Symbol": contractDetails.contract.symbol, "secType": contractDetails.contract.secType,
                      "Currency": contractDetails.contract.currency, "Exchange": contractDetails.contract.exchange,
                      "Hours": contractDetails.liquidHours}
        self.contract_details = self.contract_details._append(dictionary, ignore_index=True)

    def contractDetailsEnd(self, reqId):
        super().contractDetailsEnd(reqId)
        self.status = 1

    def updatePortfolio(self, contract, position, marketPrice, marketValue, averageCost, unrealizedPNL,
                        realizedPNL, accountName):
        super().updatePortfolio(contract, position, marketPrice, marketValue, averageCost, unrealizedPNL,
                                realizedPNL, accountName)
        dictionary = {"Symbol" : contract.symbol, "SecType": contract.secType, "Exchange":
        contract.exchange, "Position": position, "MarketPrice": marketPrice,
        "MarketValue": marketValue, "AverageCost": averageCost,
        "UnrealizedPNL" : unrealizedPNL, "RealizedPNL": realizedPNL, "AccountName": accountName}

        self.account_portfolio = self.account_portfolio._append(dictionary, ignore_index=True)

    def updateAccountValue(self, key: str, val: str, currency: str, accountName: str):
        super().updateAccountValue(key, val, currency, accountName)
        dictionary = {"Key": key, "Value": val, "Currency": currency, "AccountName": accountName}
        self.account_updates = self.account_updates._append(dictionary, ignore_index=True)

    def accountDownloadEnd(self, accountName: str):
        super().accountDownloadEnd(accountName)
        self.status = 1

    def error(self, reqId, errorCode, errorString, advancedOrderRejectJson):
        if(errorCode == 1100 or errorCode == 1300 or errorCode == 504 or errorCode == 502 or errorCode == 503):
            self.event = "IB_CONNECTION"
            logOperation(self)
            self.logger.error("Error {} {} {}".format(reqId, errorCode, errorString))
            raise Exception("Error {} {} {}".format(reqId, errorCode, errorString))

        if (errorCode == 501):
            self.logger.error(f'Connection established, error code: {errorCode}')
        logging.info("Error {} {} {}".format(reqId, errorCode, errorString))

    def tickPrice(self, reqId, tickType, price, attrib):
        super().tickPrice(reqId, tickType, price, attrib)
        # print("TickPrice. TickerId:", reqId, "tickType:", tickType,
        # "Price:", price, "CanAutoExecute:", attrib.canAutoExecute,
        # "PastLimit:", attrib.pastLimit, end = ' ')
        self.status = 1

    def tickSize(self, reqId, tickType, size):
        super().tickSize(reqId, tickType, size)
        # print("TickSize. TickerId:", reqId, "TickType:", tickType, "Size: ", size)
        self.status = 1

    def position(self, account, contract, position, avgCost):
        super().position(account, contract, position, avgCost)
        dictionary = {"Account": account, "Symbol": contract.symbol, "SecType": contract.secType,
                      "Currency": contract.currency, "conId": contract.conId,
                      "Position": position, "Avg cost": avgCost}
       # self.pos_df = self.pos_df._append(dictionary, ignore_index=True)
        print(dictionary)

    def positionEnd(self):
        super().positionEnd()
        self.status = 1

    def openOrder(self, orderId, contract, order, orderState):
        super().openOrder(orderId, contract, order, orderState)
        dictionary = {"PermId": order.permId, "ClientId": order.clientId, "OrderId": orderId,
                      "Account": order.account, "Symbol": contract.symbol, "SecType": contract.secType,
                      "Exchange": contract.exchange, "Action": order.action, "OrderType": order.orderType,
                      "TotalQty": order.totalQuantity, "CashQty": order.cashQty,
                      "LmtPrice": order.lmtPrice, "AuxPrice": order.auxPrice, "Status": orderState.status}
        self.order_df = self.order_df._append(dictionary, ignore_index=True)

    def nextValidId(self, orderId):
        super().nextValidId(orderId)
        self.nextValidOrderId = orderId

    def pnlSingle(self, reqId, pos, dailyPnL, unrealizedPnL, realizedPnL, value):
        super().pnlSingle(reqId, pos, dailyPnL, unrealizedPnL, realizedPnL, value)
        dictionary = {"ReqId": reqId, "Position": pos, "unrealizedPnL": unrealizedPnL,
                      "realizedPnL": realizedPnL, "dailyPnL": dailyPnL, "Value": value}
        self.position_pnl = self.position_pnl._append(dictionary, ignore_index=True)
        self.status = 1

    def orderStatus(self, orderId, status, filled, remaining, avgFillPrice, permId,
                    parentId, lastFillPrice, clientId, whyHeld, mktCapPrice):
        super().orderStatus(orderId, status, filled, remaining, avgFillPrice, permId, parentId,
                            lastFillPrice, clientId, whyHeld, mktCapPrice)
        if status == 'ApiPending':
            self.order_status = 'ApiPending'
        elif status == 'PendingSubmit':
            self.order_status = 'PendingSubmit'
        elif status == 'PendingCancel':
            self.order_status = 'PendingCancel'
        elif status == 'PreSubmitted':
            self.order_status = 'PreSubmitted'
        elif status == 'Submitted':
            self.order_status = 'Submitted'
        elif status == 'ApiCancelled':
            self.order_status = 'ApiCancelled'
        elif status == 'Cancelled':
            self.order_status = 'Cancelled'
        elif status == 'Filled':
            self.order_status = 'Filled'
        elif status == 'Inactive':
            self.order_status = 'Inactive'
        else:
            # Handle unknown status
            self.order_status = 'Unknown'

def websocket_con(app):
    app.run()

# creating object of the Contract class - will be used as a parameter for other function calls
def getContract(symbol, sec_type, currency, exchange):
    contract = Contract()
    contract.symbol = symbol
    contract.secType = sec_type
    contract.currency = currency
    contract.exchange = exchange
    return contract


def marketOrder(direction, quantity):
    order = Order()
    order.action = direction
    order.orderType = "MKT"
    order.totalQuantity = quantity
    return order


# creating object of the limit order class - will be used as a parameter for other function calls
def limitOrder(direction, quantity, lmt_price):
    order = Order()
    order.action = direction
    order.orderType = "LMT"
    order.totalQuantity = quantity
    order.lmtPrice = lmt_price
    return order

def getAssetPrices(app):
    url = 'https://weave.rs/getAssetPrices.php'
    params = {
        'limit': 1,  # most recent price only
        'assetid': app.asset_id
    }
    try:
        # Send a GET request to the URL with parameters
        response = requests.get(url, params=params)

        # Check the response status code
        if response.status_code == requests.codes.ok:
            app.logger.info(f'getAssetPrices connection successful, status code: {response.status_code},')
            # Split the response content by lines
            lines = response.text.strip().split('\n')
            line = lines[0]
            values = line.split('\t')

            if (len(values) != 6):
                app.logger.error(f'getAssetPrices error, {len(values)} values per line, expected 6.')
                return False

            try:
                price = float(values[2])

                if(price <= 0):
                    app.logger.error(f'getAssetPrices error, price = {price}, expected > 0.')
                    return False

            except Exception as e:
                app.logger.error(f'getAssetPrices error, price. Exception details: {e}')
                return False

            app.logger.info(f'getAssetPrices price extraction successful. Price: {price}.')

            app.price = price
            return True

        else:
            app.logger.error(f'getAssetPrice failed with status code: {response.status_code}, Response: {response.text}')
            return False

    except requests.exceptions.RequestException as e:
        app.logger.error(f'getAssetPrice error: {e}')
        return False

def getAccount(app):
    url = 'https://weave.rs/getAccount.php'
    params = {
        'account': app.account_number
    }
    try:
        # Send a GET request to the URL with parameters
        response = requests.get(url, params=params)

        # Check the response status code
        if response.status_code == requests.codes.ok:
            app.logger.info(f'getAccount connection successful, status code: {response.status_code},')
            # Split the response content by lines

            try:
                values = response.text.strip()

                app.logger.info(f'getAccount data extraction successful: {values}.')

                if (len(values) > 0):
                    return (int(values))

                return 0

            except Exception as e:
                app.logger.error(f'getAccount failed with error: {e}, Response: {response.text}.')
                return False
        else:
            app.logger.error(f'getAccount failed with status code: {response.status_code}, Response: {response.text}.')
            return False

    except requests.exceptions.RequestException as e:
        app.logger.error(f'getAccount error: {e}')
        return False


def getAccountModels(app):
    url = 'https://weave.rs/getAccountModels.php'
    params = {
        'account': app.account_number
    }

    try:
        # Send a GET request to the URL with parameters
        response = requests.get(url, params=params)

        # Check the response status code
        if response.status_code == requests.codes.ok:
            # Split the response content by lines
            app.models = response.text.strip().split('\n')
            app.logger.info(f'getAccountModels connection successful, status code: {response.status_code}.')
            return True

        else:
            app.logger.error(f'getAccountModels failed with status code: {response.status_code}. Response: {response.text}')
            return False

    except requests.exceptions.RequestException as e:
        app.logger.error(f'getAccountModels error: {e}')
        return False

def getPrediction(app):
    url = 'https://weave.rs/getPrediction.php'
    params = {
        'account': app.account_number,
        'accountmodelid': app.accountmodels[0]
    }

    try:
        # Send a GET request to the URL with parameters
        response = requests.get(url, params=params)

        # Check the response status code
        if response.status_code == requests.codes.ok:
            # Split the response content by lines
            values = response.text.strip().split('\t')
            app.logger.info(f'getPrediction connection successful, status code: {response.status_code}.')

            if(len(values) != 5):
                app.logger.error(f'getPredictions error, {len(values)} values per line, expected 5. Response: {response.text}')
                return False

            try:
                prediction_id = values[0]
                batch_id = values[1]
                modelid = values[2]
                prediction = float(values[3])
                probability = float(values[4])

            except Exception as e:
                app.logger.error(f'getPrediction error, variable assignment. Exception details: {e}')
                return False

            app.predictions = [prediction_id, batch_id, modelid, prediction, probability]
            app.logger.info(f'getPrediction data extraction successful. Predictions: {app.predictions}.')
            return True

        else:
            app.logger.error(f'getPrediction failed with status code: {response.status_code}. Response: {response.text}')
            return False

    except requests.exceptions.RequestException as e:
        app.logger.error(f'getPrediction error: {e}')
        app.event = 'HELLO_PREDICTION'
        app.operation = [app.accountmodels[0], 0, '', app.robot_id, '', '', 0, '']
        logOperation(app)
        return False


def getRobots(app):
    url = 'https://weave.rs/getRobots.php'
    params = {
        'server': app.server,  # Specify the desired limit here
    }

    try:
        # Send a GET request to the URL with parameters
        response = requests.get(url, params=params)

        # Check the response status code
        if response.status_code == requests.codes.ok:
            # Split the response content by lines
            values = response.text.strip().split('\t')
            app.logger.info(f'getRobots connection successful, status code: {response.status_code}.')

            if(len(values) != 4):
                app.logger.error(f'getRobots error, {len(values)} values per line, expected 4. Response: {response.text}')
                return False

            try:
                robot_id = int(values[0])
                asset_id = values[1]
                symbol = values[2]
                account_number = values[3]

            except Exception as e:
                app.logger.error(f'getRobots error, variable assignment. Exception details: {e}')
                return False

            app.robot_id = robot_id
            app.account_number = account_number
            app.logger.info(f'getRobots data extraction successful. RobotId: {app.robot_id}, Account Number: {app.account_number}.')
            return True

        else:
            app.logger.error(f'getRobots failed with status code: {response.status_code}, Response: {response.text}')
            return False

    except requests.exceptions.RequestException as e:
        app.logger.error(f'getRobots error: {e}')
        return False

def checkStatus(function, location, app): # total of 30 seconds wait-time, if the status returns false
    function = function.upper()
    location = location.upper()
    for i in range(app.wait_iterations):  #we will check what happened in further detail in separate functions
        if app.status == 1:
            app.logger.info(f'Callback finished. {function} - {location}. Status = {app.status}.')
            app.status = 0 # reset app status value.
            return True

        time.sleep(app.waiting_time)

    app.logger.error(f'{function} - {location} timeout error. Status = {app.status}.')
    app.event = f"IB_{function}_{location}"
    logOperation(app)

    return False

def checkOrderStatus(function, location, app):
    function = function.upper()
    location = location.upper()
    for i in range(app.wait_iterations):
        if app.order_status == 'Filled':
            app.logger.info(f'Order placed. {function} - {location}. Order status = {app.order_status}.')
            app.operation[7] = app.order_status
            logOperation(app)
            app.status = 0
            app.order_status = 0  # reset app order status value.
            return True

        app.logger.info(f'Order not filled, current status: {app.order_status}. Iteration: {i+1}/{app.wait_iterations}.')
        time.sleep(app.order_waiting_time)

    # Order was not filled, log error and operation

    app.logger.error(f'{function} - {location} timeout error. Order status = {app.order_status}.')
    app.operation[7] = app.order_status
    app.event = f"IB_{function}_{location}"
    logOperation(app)
    app.order_status = 0  # reset values.
    app.status = 0
    return False


def checkPositionAgainstPrediction(app):
    shortsymbol = app.accountmodels[6]
    longsymbol = app.accountmodels[7]

    prediction = app.predictions[3]

    for index, row in app.account_portfolio.iterrows():  # iterate through positions
        if row['Symbol'] == shortsymbol or row['Symbol'] == longsymbol:
            cur_pos = row['Position']
            symbol = row['Symbol']
            if cur_pos > 0.0:  # if we have bought this stock, we have a position.
                app.logger.info(f'Account {app.accountmodels[0]}, Current position: {cur_pos}, Symbol: {symbol}')
                if prediction == -1.0:
                    if symbol == longsymbol:
                        app.logger.info(f"Incorrect position, should be in short position.")
                        return True

                else:
                    if symbol == shortsymbol:
                        app.logger.info(f"Incorrect position, should be in long position.")
                        return True
    return False

def compareTradingHours(hours):
    hours_list = hours.split(';')
    now = datetime.datetime.now()
    open = now
    close = now

    for iteration in hours_list:  # look through pairs of open/close (since overlap with bitcoin  - always open)

        new_list = iteration.split('-')
        if(new_list[0][9:] == 'CLOSED'):
            continue
        open = datetime.datetime.strptime(new_list[0], "%Y%m%d:%H%M")
        close = datetime.datetime.strptime(new_list[1], "%Y%m%d:%H%M")
        if(now > open) and (now < close):
            return [True, open, close]  # market is open

    return [False, open, close] # market is closed

def checkEmptyPosition(app):
    shortsymbol = app.accountmodels[6]
    longsymbol = app.accountmodels[7]

    for index, row in app.account_portfolio.iterrows():  # iterate through positions (only check out of long/short positions)
        if (row['Symbol'] == shortsymbol or row['Symbol'] == longsymbol):
            if (row['Position'] > 0.0):  # if we have bought this stock, we have a position.
                symbol = row['Symbol']
                quantity = row['Position']
                app.logger.info(f'Account {app.accountmodels[0]}. Active position: Symbol: {symbol}, Position Quantity: {quantity}.')
                return False

    app.logger.info(f'Account {app.accountmodels[0]}. No active positions.')
    return True

def assignPrice(app):
    tmp_event = app.event # save previous event.
    app.operation = [app.accountmodels[0], app.predictions[0], app.predictions[1], app.robot_id, '', app.symbol, 0, '']
    app.event = 'HELLO_GET_ASSET_PRICES'
    getAssetPrices_success = getAssetPrices(app)

    if (not getAssetPrices_success):
        app.logger.error(f'getAssetPrices error. Skipping this model.')
        logOperation(app)
        return False
    app.event = tmp_event
    return True

def openPosition(app):
    symbol = app.symbol

    predictions = app.predictions
    accountmodels = app.accountmodels

    availableFunds = float(app.account_updates.loc[(app.account_updates['Key'] == "AvailableFunds")]['Value'])
    netLiquidation = float(app.account_updates.loc[(app.account_updates['Key'] == "NetLiquidation")]['Value'])

    share = accountmodels[9] - 0.01
    order_id = app.nextValidOrderId + 1

    app.price = app.longprice
    if symbol == app.accountmodels[4]:  # short
        app.price = app.shortprice

    amountAllocated = share * netLiquidation

    app.logger.info(f'Net liquidation: {netLiquidation}, Available Funds: {availableFunds}, Symbol: {symbol}, '
                    f'Share: {share}, Amount allocated: {amountAllocated}, Symbol Price: {app.price}, '
                    f'Shares: {int(amountAllocated / app.price)}')

    if(availableFunds > amountAllocated): # the account has sufficient funds, will have remaining cash.
        quantity = int(amountAllocated / app.price)
        app.logger.info(f'Sufficient funds to allocate {share*100}% of account to {symbol}. Number of shares to buy: {quantity}.')

    else: #if account has insufficient funds in cash to allocate x% to symbol, simply allocate what is left.
        quantity = int(availableFunds / app.price)
        app.logger.info(f'Insufficient funds to allocate {share * 100}% of account to {symbol}. Using avilable funds ({availableFunds}). '
                        f'Number of shares to buy: {quantity}.')

    action = 'BUY'
    app.operation = [accountmodels[0], predictions[0], predictions[1], app.robot_id, action, symbol, quantity, 'Incomplete']
    if (quantity == 0):
        app.logger.error(
            f'Open position order quantity = {quantity}, expected > 0. Account liquidation: {netLiquidation}, Share: {share}, Stock price: {app.price}.'
            f' Not placing order.')
        app.logger.error(f'Possible error source: insufficient funds, excessive share size, excessive stock price.')

        app.event = 'IB_MARKET_INSUFFICIENT_FUNDS'

        logOperation(app)

    order = marketOrder(action, quantity)
    order.eTradeOnly = False
    order.firmQuoteOnly = False

    app.logger.info(f'Placing order. Symbol: {symbol}, Quantity: {quantity}, Type: {action}, Account Model: {accountmodels[0]}. Price: {app.price}')
    app.placeOrder(order_id, getContract(symbol, "STK", "USD", "SMART"), order)

    if (not checkOrderStatus('placeOrder', 'openPosition', app)):
        return False

    return True



def closePosition(app):
    accountmodels = app.accountmodels
    predictions = app.predictions
    action = 'SELL'

    symbol = app.symbol
    app.price = app.longprice
    if symbol == app.accountmodels[4]: #short symbol
        app.price = app.shortprice

    app.operation = [accountmodels[0], predictions[0], predictions[1], app.robot_id, action, symbol, 0, '']

    app.logger.info(f'Iterating through portfolio to close: {symbol}.')

    for index, row in app.account_portfolio.iterrows():
        if(row['Position'] > 0.0):
            if (row['Symbol'] == symbol):
                order_id = app.nextValidOrderId
                quantity = int(row['Position'])

                app.logger.info(f'Current position: {symbol}, {quantity}.')

                if (quantity == 0):
                    app.logger.error(f'Close position order quantity = {quantity}, expected > 0. Position was {quantity}. Not placing order.')
                    break

                app.operation[6] = quantity

                order = marketOrder(action, quantity)
                order.eTradeOnly = False
                order.firmQuoteOnly = False

                app.logger.info(f'Placing order. Symbol: {symbol}, Quantity: {quantity}, Type: SELL, Account Model: {accountmodels[0]}')

                app.placeOrder(order_id, getContract(symbol, "STK", "USD", "SMART"), order)

                if (not checkOrderStatus('placeOrder'.upper(), 'closePosition'.upper(), app)):  # trade did not go through
                    return False

                return True

    app.logger.info(f'Position with symbol: {symbol} not found. Portfolio: \n {app.account_portfolio}.')
    return False


def postOperation(app):
    # Define the URL
    url = 'https://weave.rs/postOperation.php'
    operation = app.operation

    # Define the parameters
    params = {
        'account': app.account,
        'accountmodelid': operation[0],
        'predictionid': operation[1],
        'batchid': operation[2],
        'robotid': operation[3],
        'type': operation[4],
        'symbol': operation[5],
        'quantity': operation[6],
        'status': operation[7],
        'price': app.price,
        'event': app.event
    }
    try:
        # Send the POST request
        response = requests.post(url, params=params)
        if response.status_code == requests.codes.ok and response.text == '1':
            app.logger.info(f'Successful postOperation. Status code: {response.status_code}, Response: {response.text}')
            return True
        else:
            app.logger.info(f'postOperation failed with status code: {response.status_code}, Response: {response.text}')
            return False
    except requests.exceptions.RequestException as e:
        app.logger.error(f'postOperation error: {e}')
        return False

def postAccountPnL(app):
    # Define the URL
    url = 'https://weave.rs/postAccountPnL.php'

    # Define the parameters
    params = {
        'account': app.account_number,
        'batchid': app.operation[2],
        'unrealized': app.unrealizedPnL,
        'realized': app.realizedPnL
    }

    try:
        # Send the POST request
        response = requests.post(url, params=params)
        if response.status_code == requests.codes.ok and response.text == '1':
            app.logger.info(f'Successful postAccountPnL. Status code: {response.status_code}, Response: {response.text}')
            return True

        else:
            app.logger.info(f'postAccountPnL failed with status code: {response.status_code}, Response: {response.text}')
            return False

    except requests.exceptions.RequestException as e:
        app.logger.error(f'postAccountPnL error: {e}')
        return False


def postAccountModelPnL(app):
    # Define the URL
    url = 'https://weave.rs/postAccountModelPnL.php'
    # Define the parameters
    params = {
        'account': app.account_number,
        'accountmodelid': app.accountmodels[0],
        'batchid': app.operation[2],
        'symbol' : app.symbol,
        'quantity': app.quantity,
        'avgcost': app.avg_cost,
        'unrealized': app.unrealizedPnL,
        'realized': app.realizedPnL
    }

    try:
        # Send the POST request
        response = requests.post(url, params=params)
        if response.status_code == requests.codes.ok and response.text == '1':
            app.logger.info(f'Successful postAccountModelPnL. Status code: {response.status_code}, Response: {response.text}')
            return True

        else:
            app.logger.info(f'postAccountModelPnL failed with status code: {response.status_code}, Response: {response.text}')
            return False

    except requests.exceptions.RequestException as e:
        app.logger.error(f'postAccountModelPnL error: {e}')
        return False

def checkPnLStoploss(app):
    shortsymbol = app.accountmodels[6]
    longsymbol = app.accountmodels[7]
    stoploss = app.accountmodels[8]
    app.short_stoploss = False
    app.long_stoploss = False
    app.stoploss_to_close = False

    for index, row in app.account_portfolio.iterrows():
        if(row['Symbol'] == shortsymbol or row['Symbol'] == longsymbol):
            if float(row['Position']) > 0.0:
                app.symbol = row['Symbol']
                app.logger.info(f'Fetching data for {app.symbol}.')

                app.market_price = float(row['MarketPrice'])
                app.avg_cost = float(row['AverageCost'])
                app.unrealizedPnL = float(row['UnrealizedPNL'])
                app.realizedPnL = float(row['RealizedPNL'])
                app.quantity = float(row['Position'])
                percentagePnL = app.market_price / app.avg_cost - 1

                app.logger.info(f'Position with {app.symbol}: {app.quantity}. Realized PnL: {app.realizedPnL}. '
                                f'Unrealized PnL: {app.unrealizedPnL}. Percentage PnL = {percentagePnL}')

                app.logger.info(f'Calling postAccountModelPnL')
                if (not postAccountModelPnL(app)):
                    tmp_event = app.event  # save event for logging purposes
                    app.event = 'HELLO_POST_ACCOUNT_MODEL_PNL'
                    logOperation(app)
                    app.logger.error(f'postAccountModelPnL error.')
                    app.event = tmp_event

                app.logger.info(f'postAccountModelPnL finished for symbol: {app.symbol}, checking stoploss.')

                if (percentagePnL < stoploss):  #### pnl is less than
                    app.logger.info(f'{app.symbol} stoploss reached: PnL({percentagePnL}) < Stoploss({stoploss})')
                    app.stoploss_to_close = True
                    if app.symbol == shortsymbol:  # Changing stoploss variable for when to change close position.
                        app.short_stoploss = True
                    else:
                        app.long_stoploss = True

                    return True

                  # stoploss not reached, don't change anything.
                app.logger.info(f'{app.symbol} stoploss not reached: PnL({percentagePnL}) >= Stoploss({stoploss})')

                return True

    app.logger.info(f'{shortsymbol}, {longsymbol} not found in active positions.')
    return False

def logOperation(app):
    operation = app.operation
    event = app.event
    robot_id = app.robot_id
    if(not postOperation(app)): # (account,accountmodelid,predictionid,type,symbol,quantity,status)
        app.logger.error(f'Failed postOperation')
    if event[0:5] == 'ERROR' or event[0:5] == 'HELLO' or event[0:2] == 'IB':
        app.logger.error(f'Event = {event}. RobotId: {robot_id}. Account Model: {operation[0]}, '
                         f'Prediction Id: {operation[1]}, BatchId: {operation[2]}, RobotId: {operation[3]}, Type: {operation[4]}, '
                         f'Symbol: {operation[5]}, Quantity: {operation[6]}, Price: {app.price}, Status: {operation[7]}')
    else:
        app.logger.info(f'Event = {event}. RobotId: {robot_id}. Account Model: {operation[0]}, '
                        f'Prediction Id: {operation[1]}, BatchId: {operation[2]}, RobotId: {operation[3]}, Type: {operation[4]}, '
                        f'Symbol: {operation[5]}, Quantity: {operation[6]}, Price: {app.price}, Status: {operation[7]}')

def setup_logger(logger_name, log_file, formatter, level):
    l = logging.getLogger(logger_name)
    fileHandler = logging.FileHandler(log_file, mode='a')
    fileHandler.setFormatter(formatter)
    streamHandler = logging.StreamHandler()
    streamHandler.setFormatter(formatter)

    l.setLevel(level)
    l.addHandler(fileHandler)
    l.addHandler(streamHandler)

    return l

def preparelogger(app):
    logname = 'API_Log_File.log'
    logname_2 = 'Trader_Log_File.log'

    log_dir = os.path.join(os.path.normpath(os.getcwd() + os.sep), 'Trading_Logs')
    log_fname = os.path.join(log_dir, logname)
    log_fname_2 = os.path.join(log_dir, logname_2)

    log_dir_exists = os.path.exists(log_dir)
    if (not log_dir_exists):  # make model if does not work
        os.mkdir('Trading_Logs')

    logging.basicConfig(handlers=[logging.FileHandler(log_fname)], encoding='utf-8',
                        format=f'%(asctime)s : %(levelname)s : %(message)s : Line %(lineno)d',
                        datefmt='%d/%m/%Y %I:%M %p', level=logging.INFO)

    formatter = logging.Formatter(fmt='%(asctime)s %(levelname)s %(message)s', datefmt='%d/%m/%Y %I:%M %p')
    logger = setup_logger('app.logger', log_fname_2, formatter, 'INFO')

    app.logger = logger


def trading(argv):
    app = TradingApp()
    preparelogger(app)  # 'trading_log.log')
    app.logger.info(f'Logger initialized.')

# HELLO - CONFIG
    app.event = "HELLO_CONFIG"
    try:
        app.server = argv[1]

    except Exception as e:
        logOperation(app)
        app.logger.error(f'Input error, robot Id. Exiting program. Exception details:', exc_info = True)
        return

# HELLO - getRobots
    app.event = "HELLO_GET_ROBOTS"
    getRobots_success = getRobots(app) #raises exception if fails.

    if(not getRobots_success):
        logOperation(app)
        app.logger.error(f'getRobots failed. Exiting program.')
        return #exit program.

    app.logger.info(f'Successful getRobots. Calling getAccount. RobotId = {app.robot_id}')

# HELLO - getAccount --------------------------
    app.event = "HELLO_GET_ACCOUNT"
    app.operation = [0, 0, '', app.robot_id, '', '', '', 0, '']

    account_active = getAccount(app)  #  raises exception if failure.
    if (account_active == False):
        logOperation(app)
        app.logger.error(f'getAccount failed. Exiting program.')
        return

    app.logger.info(f'Successful getAccount. Checking model activity. Account active: {account_active}')

    if (account_active == 0):  # Model inactive
        app.logger.info(f'Account is inactive, exiting program.')
        app.event = "OPERATION_ACCOUNT_INACTIVE"
        logOperation(app)
        return # exit program

# HELLO - getAccountModels ------------------
    app.logger.info(f'Model is active, calling getAccountModels.')
    app.event = "HELLO_GET_ACCOUNT_MODELS"
    getAccountModels(app)   # since we cannot execute any trades without it.

    if(len(app.models) == 0): # No models in getAccountModels.
        app.logger.error(f'getAccountModels error, no models. Expected at least 1. Exiting program.')
        logOperation(app) #is this rly an error? empty gAM
        return #exit, no models.

    app.logger.info(f'Successful getAccountModels. {len(app.models)} models found. Iterating through models.')

    iterations = 1
    for model in app.models:
        app.logger.info(f'Resetting PnL, Positions, Account Summary, Contract Details. Model iteration: {iterations}')
        app.reset_contract_details()  #
        values = model.split('\t')
        iterations += 1

        if (len(values) != 10):
            app.logger.error(f'getAccountModels error, {len(values)} values per line, expected 10. Skipping this model.')
            logOperation(app) #app.event = "HELLO_GET_ACCOUNT_MODELS"
            continue

        try:  #accountmodelid, modelid, active, threshold, shortassetid, longassetid, shortsymbol, longsymbol, stoploss, share
            accountmodelid = values[0]
            modelid = values[1]
            active = int(values[2])
            threshold = float(values[3])
            shortassetid = values[4]
            longassetid = values[5]
            shortsymbol = values[6]
            longsymbol = values[7]
            stoploss = float(values[8])
            share = float(values[9])

        except Exception as e:
            app.logger(f'getAccountModels error, value extraction. Skipping this model.')
            logOperation(app)   #app.event = "HELLO_GET_ACCOUNT_MODELS"
            continue

        if shortsymbol == '' and longsymbol == '':  # if shortsymbol is empty.
            app.logger.error(f'Longsymbol and shortsymbol both empty. Skipping this model..')
            logOperation(app)
            continue

        app.accountmodels = [accountmodelid, modelid, active, threshold, shortassetid, longassetid, shortsymbol, longsymbol, stoploss, share]


        app.logger.info(f'getAccountModels data extraction successful. Checking account activity. AccountModels: {app.accountmodels}.')
        accountmodelid = app.accountmodels[0]

# OPERATION - account model active/inactive.
        app.event = "OPERATION_ACCOUNT_MODEL_INACTIVE"
        app.operation = [app.accountmodels[0], 0, '', app.robot_id, '', '', 0, '']
        if(app.accountmodels[2] == 0):
            app.logger.info(f'Model {app.accountmodels[1]} inactive, verifying positions.')
            if checkEmptyPosition(app):
                app.logger.info(f'No current positions. Skipping this model.')
                logOperation(app)

            else:  # Closing all trades if inactive model.
                app.event = "OPERATION_ACCOUNT_MODEL_INACTIVE_CLOSING"
                if (not closePosition(app)):
                    app.logger.error(f'Failed to close active positions, account model {accountmodelid}. Continuing to next model.')
                    continue

                app.logger.info(f'Successfully closed all positions for: Account Model {accountmodelid}, '
                                f'inactive model {app.accountmodels[1]}. Skipping to next model.')

            continue #regardless of whether active positions

        app.logger.info(f'Model {app.accountmodels[1]} active.')


# HELLO - getPrediction --------------------
        app.logger.info(f'Getting predictions. Account Model: {accountmodelid}, Model Id: {app.accountmodels[1]}.')
        app.event = 'HELLO_GET_PREDICTION'
        app.operation = [app.accountmodels[0], 0, '', app.robot_id, '', '', 0, '']

        getPrediction_success = getPrediction(app)
        if(not getPrediction_success):
            app.logger.error(f'getPrediction error. Skipping this model.')
            logOperation(app)
            continue #go to next model.

        predictions = app.predictions
        symbols = f'{app.accountmodels[6]}, {app.accountmodels[7]}'
        app.operation = [app.accountmodels[0], app.predictions[0], app.predictions[1], app.robot_id, '', symbols, 0, '']


# HELLO - getAssetPrices
        app.logger.info(f'Hello - getPrediction finished. Calling getAssetPrices.')
        app.event = 'HELLO_GET_ASSET_PRICES'

        for symbol in [app.accountmodels[6], app.accountmodels[7]]:
            if symbol != '':
                app.asset_id = longassetid
                if symbol == shortsymbol:
                    app.asset_id = shortassetid

                if (not getAssetPrices(app)):
                    app.logger.error(f'getAssetPrices error. Skipping symbol: {symbol}.')
                    logOperation(app)
                    return False

                app.longprice = app.price
                if symbol == shortsymbol:
                    app.shortprice = app.price

        app.logger.info(f'Hello section finished. getAssetPrices finished. Moving to IB section..')

# ==================================== IB SECTION ====================================

# IB - Connection
        app.event = "IB_CONNECTION"
        counter = 0
        while(not app.isConnected()):
            if(counter > 0):  # try disconnecting and reconnecting.
                app.disconnect()
                time.sleep(1)
                app.thread.join()

            try:
                app.connect(app.connection_details[0], app.connection_details[1], app.connection_details[2])
                # starting a separate daemon thread to execute the websocket connection

                con_thread = threading.Thread(target=websocket_con, args=[app], daemon=True)
                con_thread.start()
                app.thread = con_thread
                time.sleep(1)

            except Exception as e:
                app.logger.error(f"Connection Error: {e}. Launching IB script. "
                                 f"Connection Details: {app.connection_details}, Account Model: {accountmodelid}")

                ib_process = subprocess.run('start cscript "C:\Users\Administrator\Desktop\ib.vbs', check=True,
                                            capture_output=True, timeout=400)

                if ib_process.returncode != 0:
                    raise ib_process.CalledProcessError(f"IB Script error, exiting trading.py. Return code: {ib_process.returncode}."
                                                        f" Stdout: {ib_process.stdout}. Stderr: {ib_process.stderr}.")

                app.logger.info(f"IB script complete, connecting to gateway."
                                 f" Return code: {ib_process.returncode}. Stdout: {ib_process.stdout}.")
                app.connect(app.connection_details[0], app.connection_details[1], app.connection_details[2])
                # starting a separate daemon thread to execute the websocket connection

                con_thread = threading.Thread(target=websocket_con, args=[app], daemon=True)
                con_thread.start()
                app.thread = con_thread
                time.sleep(1)

            counter += 1

            if(counter >= 10):
                logOperation(app)
                app.disconnect()
                app.thread.join()
                break

        if (not app.isConnected()):
            app.logger.error(f"Connection Error. Skipping to next model. "
                             f"Connection Details: {app.connection_details}, Account Model: {accountmodelid}")
            logOperation(app)
            app.disconnect()
            app.thread.join()
            continue
        app.logger.info(f'App is connected: {app.isConnected()}.')
        app.logger.info(f"Connection established. Details: {app.connection_details}, Account Model: {accountmodelid}")

# IB - reqContractDetails
        index = 6
        if(shortsymbol == ''): # if shortsymbol is empty, take long symbol for market hours, assume both not ''
            index = 7
        app.logger.info(f'Fetching market hours. Symbol: {app.accountmodels[index]}')
        contract = getContract(app.accountmodels[index], "STK", "USD", "SMART")

        app.reqContractDetails(app.connection_details[2], contract)

        if(not checkStatus('reqContractDetails', 'Main', app)):
            app.logger.error(f'Failed to get market hours for account model {accountmodelid}. Skipping this model.')
            app.operation = [app.accountmodels[0], app.predictions[0], app.predictions[1], app.robot_id,
                          '', app.accountmodels[index], 0, '']
            logOperation(app)
            if (app.isConnected()):
                app.logger.info(f'Disconnecting from API.')

                app.disconnect()
                app.thread.join()
            continue

        hours = app.contract_details.iloc[0]['Hours']
        app.open = compareTradingHours(hours)

        app.logger.info(f"Contract details and market hours fetched.")

        app.logger.info(f"Market hour data success. Requesting PnL for postAccountPnL.")

# IB - reqAccountUpdates
        app.reset_account_updates()
        app.reqAccountUpdates(True, app.account_number) #Keep this subscription open until the end since it auto-updates with info.
        if (not checkStatus('reqAccountUpdates', 'main', app)):
            app.disconnect()
            app.thread.join()
            app.logger.info(f'reqAccountUpdates failed. Going to next model. Disconnecting from API.')
            continue

        app.logger.info(f'reqAccountUpdates callback finished. Cancelling subscription. Portfolio, updates: \n {app.account_portfolio} \n {app.account_updates}')
        app.reqAccountUpdates(False, app.account_number)

        app.realizedPnL = float(app.account_updates.loc[(app.account_updates['Key'] == "RealizedPnL") & (
                    app.account_updates['Currency'] == "BASE")]['Value'])

        app.unrealizedPnL = float(app.account_updates.loc[(app.account_updates['Key'] == "UnrealizedPnL") & (
                    app.account_updates['Currency'] == "BASE")]['Value'])

        if (not postAccountPnL(app)):
            app.logger.error(f'Failed postAccountPnL.')
            app.event = 'HELLO_POST_ACCOUNT_PNL'
            logOperation(app)

        app.logger.info(f'Cancelled position subscription. Calling checkPnLStopLoss, postAccountModelPnl.')
        if(not checkPnLStoploss(app)):
            app.logger.info(f'checkPnLStoploss not complete. Continuing, assuming stoploss not reached.')

        app.logger.info(f'checkPnLStopLoss finished, verifying market hours.')

#  IB - Operation Market Closed
        if (not app.open[0]): #true or false.
            app.event = "OPERATION_MARKET_CLOSED"
            logOperation(app)
            app.logger.info(f'Market currently closed. Going to next model. Time: {datetime.datetime.now().strftime("%y-%m-%d %H:%M:%S")}. '
                            f'Open from {app.open[1]} to {app.open[2]}.')
            if (app.isConnected()):
                app.logger.info(f'Disconnecting from API.')

                app.disconnect()
                app.thread.join()
            continue

        app.logger.info(f'Market currently OPEN, time: {datetime.datetime.now().strftime("%y-%m-%d %H:%M:%S")}. '
                        f'Open from {app.open[1]} to {app.open[2]}.')

# IB - stoploss
        app.event = "OPERATION_STOPLOSS"
        app.logger.info(f'Account Model: {accountmodelid}. If stoploss reached, closing positions.')

        if(app.short_stoploss):
            app.symbol = app.accountmodels[6]
            app.logger.info(f'Stoploss reached for {app.symbol}. Closing position. Account Model: {accountmodelid}')
        elif(app.long_stoploss):
            app.symbol = app.accountmodels[7]
            app.logger.info(f'Stoploss reached for {app.symbol}. Closing position. Account Model: {accountmodelid}')
        else:
            app.logger.info(f'Stoploss not reached.')

        if(app.stoploss_to_close):
            app.logger.info(f'Calling closePosition.')
            if (not closePosition(app)):
                app.logger.error(f'Stoploss closePosition failed. Account model {accountmodelid}. Continuing to next model.')
                if (app.isConnected()):
                    app.logger.info(f'Disconnecting from API.')

                    app.disconnect()
                    app.thread.join()
                continue

        app.logger.info(f'Stoploss check success. postAccountPnL, postAccountModelPnL finished. Checking prediction.')
        app.stoploss_to_close = False

# OPERATION - Prediction section----------------
        if(predictions[4] < threshold):
            app.logger.info(f'Prediction {predictions[0]} : Probability ({predictions[4]}), less than threshold '
                f'({threshold}). No operation, skip to next model.')
            app.event = f'OPERATION_PREDICTION_UNDER_THRESHOLD'
            logOperation(app)
            if (app.isConnected()):
                app.logger.info(f'Disconnecting from API.')

                app.disconnect()
                app.thread.join()
            continue

        app.logger.info(f'Prediction {predictions[0]} : Probability ({predictions[4]}), greather or equal to '
                         f'threshold ({threshold}). Checking positions.')

# Case 1: No current positions.
        if checkEmptyPosition(app):
            app.logger.info(f'Case 1: No current positions. Making first order. Account Model: {accountmodelid}')
            # Open new position
            app.event = "OPERATION_PREDICTION_INIT_SHORT"
            app.symbol = shortsymbol
            if(predictions[3] == 1.00):
                app.symbol = longsymbol
                app.event = "OPERATION_PREDICTION_INIT_LONG"

            if (app.symbol == ''):
                app.logger.info(f'Cash position. No operation.')
                app.operation = [app.accountmodels[0], app.predictions[0], app.predictions[1], app.robot_id,
                '', app.symbol , 0, '']
                app.event = "OPERATION_CASH"
                logOperation(app)

            else:
                app.logger.info(f'Symbol: {app.symbol}, not a cash position. Opening position.')
                if (not openPosition(app)):
                    app.logger.error(f'Failed to open position. Account model {accountmodelid}. Continuing to next model.')
                    if (app.isConnected()):
                        app.logger.info(f'Disconnecting from API.')

                        app.disconnect()
                        app.thread.join()
                    continue #order failed, go to next model.

# Case 2: Position not alligned with prediction
        elif checkPositionAgainstPrediction(app):
            app.logger.info(f'Case 2: Position not alligned with prediction. Account Model: {accountmodelid}')

            app.event = "OPERATION_PREDICTION_FROM_LONG_TO_SHORT"
            app.symbol = longsymbol #closing short
            if (predictions[3] == 1.00):
                app.event = "OPERATION_PREDICTION_FROM_SHORT_TO_LONG"
                app.symbol = shortsymbol  # closing short

            # Close current position
            if(not closePosition(app)):
                app.logger.error(f'Failed to close position. Account model {accountmodelid}. Continuing to next model.')
                if (app.isConnected()):
                    app.logger.info(f'Disconnecting from API.')

                    app.disconnect()
                    app.thread.join()
                continue #could not close position, go to next model...

            app.logger.info(f'Resetting account updates due to action taken. Calling reqAccountUpdates.')
            app.reset_account_updates()
            app.reqAccountUpdates(True, app.account_number)  # Keep this subscription open until the end since it auto-updates with info.
            if (not checkStatus('reqAccountUpdates', 'main', app)):
                app.disconnect()
                app.thread.join()
                app.logger.info(f'reqAccountUpdates failed. Going to next model. Disconnecting from API.')
                continue

            app.logger.info(
                f'reqAccountUpdates callback finished. Cancelling subscription. Portfolio, updates: \n {app.account_portfolio} \n {app.account_updates}')
            app.reqAccountUpdates(False, app.account_number)

            app.symbol = shortsymbol
            if (predictions[3] == 1.00):
                app.symbol = longsymbol

            if(app.symbol == ''):
                app.logger.info(f'Cash position. No operation.')
                app.operation = [app.accountmodels[0], app.predictions[0], app.predictions[1], app.robot_id,  '', app.symbol, 0, '']
                app.event = "OPERATION_CASH"
                logOperation(app)

            else: # if app.symbol is not empty, open the new position with corresponding symbol
                app.logger.info(f'Symbol: {app.symbol}, not a cash position. Opening position.')
                if(not openPosition(app)):
                    app.logger.error(f'Failed to open position. Account model {accountmodelid}. Continuing to next model.')
                    if (app.isConnected()):
                        app.logger.info(f'Disconnecting from API.')

                        app.disconnect()
                        app.thread.join()
                    continue #go to next model

#Case 3: Position alligned with prediction.
        else:
            app.logger.info(f'Case 3: Position alligned with prediction. No operation necessary. Account Model: {accountmodelid}')
            app.event = 'OPERATION_PREDICTION_NONE'
            logOperation(app)

        app.logger.info(f' ============== Program for account model: {accountmodelid} finished. Next model...  ============== ')

    if(app.isConnected()):
        app.logger.info(f'Disconnecting from API.')

        app.disconnect()
        app.thread.join()

    app.logger.info(f' ============== All models executed. Program finished. ============== ')

if (__name__ == '__main__'):
    trading(sys.argv)
    #trading([0,'Trader_1'])

##          python      trading.py   server_name