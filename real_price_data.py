import requests
import numpy as np 
import matplotlib.pyplot as plt

url = 'https://rest.coinapi.io/v1/exchangerate/ETH/USD/history?period_id=1DAY&time_start=2018-01-01T00:00:00&time_end=2021-07-02T00:00:00'
headers = {'X-CoinAPI-Key' : '0B114B16-1A97-428B-89C7-709F15BC54C3'}
response = requests.get(url, headers=headers)
response_json = response.json()

def makePriceArray(elem):
    return elem["rate_close"]

priceArray = list(map(makePriceArray, response_json))
print(len(priceArray))

# plt.plot(priceArray)
# plt.show()

dailyReturns = []

for i in range(len(priceArray)-1):
    dailyReturns.append(priceArray[i+1]/priceArray[i])

dailyReturns = np.array(dailyReturns)
dailyReturns = np.log(dailyReturns)
print(np.shape(dailyReturns))

# plt.plot(dailyReturns)
# plt.show()

h = np.histogram(dailyReturns)
plt.hist(h, bins='auto')
plt.show()




