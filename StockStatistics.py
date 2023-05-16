import yfinance as yf
import statistics
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import lognorm, expon
import matplotlib.colors as mcolors

def calculate_stock_statistics(symbol):
    # Retrieve historical stock data from Yahoo Finance
    stock_data = yf.download(symbol, start='2022-01-01', end='2022-12-31')
    
    # Extract the closing prices from the stock data
    closing_prices = stock_data['Close']
    
    # Calculate the average
    average = statistics.mean(closing_prices)
    
    # Calculate the median
    median = statistics.median(closing_prices)
    
    # Calculate the mode
    try:
        mode = statistics.mode(closing_prices)
    except statistics.StatisticsError:
        mode = "No unique mode"
    
    # Calculate the range
    price_range = max(closing_prices) - min(closing_prices)
    
    # Return the calculated statistics
    return average, median, mode, price_range

def plot_distributions(data):
    # Fit the data to a lognormal distribution
    log_shape, log_loc, log_scale = lognorm.fit(data, floc=0)
    
    # Fit the data to an exponential distribution
    exp_loc, exp_scale = expon.fit(data)
    
    # Generate values for the x-axis
    x = np.linspace(min(data), max(data), 100)
    
    # Calculate the lognormal and exponential PDFs using the fitted parameters
    log_pdf = lognorm.pdf(x, log_shape, log_loc, log_scale)
    exp_pdf = expon.pdf(x, exp_loc, exp_scale)
    
    # Plot the distributions
    plt.figure(figsize=(8, 6))
    plt.plot(x, log_pdf, color='lightgreen', label='Lognormal Distribution')
    plt.plot(x, exp_pdf, color=mcolors.CSS4_COLORS['mediumorchid'], label='Exponential Distribution')
    plt.xlabel('Values')
    plt.ylabel('Probability Density')
    plt.title('Lognormal and Exponential Distributions')
    plt.legend()
    plt.gca().set_facecolor('white')
    plt.show()

def main():
    stock_symbol = 'LZ'
    average, median, mode, price_range = calculate_stock_statistics(stock_symbol)

    print(f"Stock: {stock_symbol}")
    print(f"Average: {average}")
    print(f"Median: {median}")
    print(f"Mode: {mode}")
    print(f"Range: {price_range}")
    
    # Generate random data for the distributions
    np.random.seed(0)
    data = np.random.lognormal(mean=1, sigma=0.5, size=1000)
    
    # Plot the distributions
    plot_distributions(data)

if __name__ == '__main__':
    main()
