# Portfolio Analytics & Optimization Toolkit
# Built on top of the original stock tracker + correlation viewer projects
#Sneha Anudeep verma 

import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import minimize

stock_data_cache = {}

TICKERS = ["AAPL", "DIS", "NVDA", "JPM", "PG"] #the stock tickers be

#gets stock history for a 6 month period 
def get_stock_history(ticker, period="2y"):
    """Gets price history for a ticker, using a cached copy if already fetched."""
    if ticker in stock_data_cache:
        return stock_data_cache[ticker]
    stock = yf.Ticker(ticker)
    data = stock.history(period=period)
    stock_data_cache[ticker] = data
    return data


def get_stock_history_close(ticker, period="2y"):
    """Gets the Close price history of the chosen stock."""
    data = get_stock_history(ticker, period) #calls on history function 
    return data["Close"]#returns the close row of the data frame 


def get_daily_percent_change(ticker):
    """Calculates the day-over-day percent change (daily returns)."""
    return get_stock_history_close(ticker).pct_change() #uses the percent change function on the close row 


def get_combined_returns(tickers):
    """Combines daily returns for multiple tickers into one aligned DataFrame."""
    returns_dict = {ticker: get_daily_percent_change(ticker) for ticker in tickers} #get percent change for every ticker in the list 
    return pd.concat(returns_dict, axis=1, sort=False).dropna() #removes the NAN values and moves the values one row up 


def get_correlation_matrix(tickers):
    """Computes the correlation matrix of daily returns across multiple tickers."""
    return get_combined_returns(tickers).corr() #uses the corrolation function on the returns for the tickers in the list 


def plot_correlation_heatmap(tickers):
    corr = get_correlation_matrix(tickers) #fetches matrix 
    sns.heatmap(corr, annot=True, cmap="coolwarm", vmin=-1, vmax=1) #creates a heat map with the corrolation values
    plt.title("Stock Correlation Heatmap") #sets title for graph 
    plt.tight_layout() #makes it look neat 




def get_annualized_stats(tickers, risk_free_rate=0.04):
    """
    Returns (expected_returns, covariance_matrix), both annualized.
    expected_returns: average historical daily return * 252, per ticker
    covariance_matrix: how the stocks move together, annualized
    """
    returns = get_combined_returns(tickers) #combined daily returns is fetched 
    expected_returns = returns.mean() * 252 #multiplies by 252 to get annulizes value 
    cov_matrix = returns.cov() * 252 #covariance functino used 
    return expected_returns, cov_matrix #returns both so multiple functions dont need to be made 


def portfolio_performance(weights, expected_returns, cov_matrix, risk_free_rate=0.04):
    """Given a set of weights, compute portfolio return, volatility, and Sharpe ratio."""
    
    port_return = np.dot(weights, expected_returns) #uses the dot product to give you a predicted return
    
    #squarreroots the dot prodoct if the weights matrix flipped and the dor product of the covvarience and weights.
    
    #np.dot(cov_matrix, weights) returns an array -- so you have to .T the weights othereise it would break because of matrix multiplcation 
    port_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    
    sharpe = (port_return - risk_free_rate) / port_volatility #calculates sharpe value for whole portfolio (its an average basically)
    return port_return, port_volatility, sharpe


def negative_sharpe(weights, expected_returns, cov_matrix, risk_free_rate=0.04):
    """scipy.optimize only minimizes, so we minimize the NEGATIVE Sharpe ratio
    to effectively maximize it."""
    _, _, sharpe = portfolio_performance(weights, expected_returns, cov_matrix, risk_free_rate) #fetch sharpe
    
    return -sharpe #makes it negative to maximize for scipy


def optimize_portfolio(tickers, risk_free_rate=0.04):
    """
    Finds the portfolio weights that maximize the Sharpe ratio.
    Constraints: weights sum to 1 (fully invested), no shorting (each weight >= 0).
    """
    expected_returns, cov_matrix = get_annualized_stats(tickers, risk_free_rate)

    
    n = len(tickers)#amount of tickers 

    #makes the weights equal 
    initial_guess = np.array([1.0 / n] * n)

    # Constraint: weights must sum to 1 --- for the minimize funtion 
    constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1})

    # Bounds: each weight between 0 and 1 (no shorting, no leverage) -- for the minimize function 
    bounds = tuple((0, 1) for _ in range(n))

    #calling the minimize function 
    result = minimize(
        negative_sharpe, #function it is trying to minimize  
        initial_guess, #whhat we think the initial weights should be 
        args=(expected_returns, cov_matrix, risk_free_rate), #what the negative sharpe function needs 
        method='SLSQP', #method for solving -- this is good for contrainghts 
        bounds=bounds, #what our bounds are for our guesses 
        constraints=constraints #a rule that must be satisfied the most optimal solution 
    )

    optimal_weights = result.x #the optimal solution
    
    port_return, port_vol, sharpe = portfolio_performance(
        optimal_weights, expected_returns, cov_matrix, risk_free_rate
    ) #uses optimal solution, the cov_matrox and expected returns for the whole portfolio to calculate the performance 

    return optimal_weights, port_return, port_vol, sharpe


def display_optimal_portfolio(tickers, risk_free_rate=0.04):
    """Prints a clean summary of the optimal portfolio allocation."""
    
    weights, port_return, port_vol, sharpe = optimize_portfolio(tickers, risk_free_rate)

    #formatting 
    print("=" * 50)
    print("  Optimal Portfolio (Max Sharpe Ratio)")
    print("=" * 50)
    for ticker, weight in zip(tickers, weights):
        print(f"  {ticker}: {weight:.1%}")
    print("-" * 50)
    print(f"  Expected Annual Return: {port_return:.2%}")
    print(f"  Annual Volatility:      {port_vol:.2%}")
    print(f"  Sharpe Ratio:           {sharpe:.2f}")
    print("=" * 50)

    return weights

def compare_to_equal_weight(tickers, risk_free_rate=0.04):
    """Compares the optimized portfolio's Sharpe ratio against an equal-weighted baseline."""
    expected_returns, cov_matrix = get_annualized_stats(tickers, risk_free_rate)
    n = len(tickers)

    #get equal weights 
    equal_weights = np.array([1.0 / n] * n)
    eq_return, eq_vol, eq_sharpe = portfolio_performance(
        equal_weights, expected_returns, cov_matrix, risk_free_rate
    )

    # Optimized portfolio
    opt_weights, opt_return, opt_vol, opt_sharpe = optimize_portfolio(tickers, risk_free_rate)

    #calculate percent change 
    improvement_pct = ((opt_sharpe - eq_sharpe) / eq_sharpe) * 100

    print("=" * 50)
    print("  Equal-Weight vs. Optimized Comparison")
    print("=" * 50)
    print(f"  Equal-Weight Sharpe:  {eq_sharpe:.2f}")
    print(f"  Optimized Sharpe:     {opt_sharpe:.2f}")
    print(f"  Improvement:          {improvement_pct:.1f}%")
    print("=" * 50)

    return improvement_pct


compare_to_equal_weight(TICKERS)

def plot_efficient_frontier(tickers, num_portfolios=2000, risk_free_rate=0.04):
    """
    Simulates random portfolios to visualize the risk/return tradeoff,
    then highlights the optimal (max Sharpe) portfolio found by the optimizer.
    """
    expected_returns, cov_matrix = get_annualized_stats(tickers, risk_free_rate) #get stats
    n = len(tickers)

    #creating array with 3 rows and num_portfolios number of coloums 
    results = np.zeros((3, num_portfolios))

    #a for loop creating a scatter plot with random outcomes to see where the optimized portfolio falls  -- Monte Carlo simulation results!
    
    for i in range(num_portfolios):
        
        weights = np.random.random(n) #generates an array of n random numbers {1,2,3,4,
        #print(f"this is how weights works {weights}")
        weights /= np.sum(weights)
        port_return, port_vol, sharpe = portfolio_performance(
            weights, expected_returns, cov_matrix, risk_free_rate
        )
        results[0, i] = port_vol
        results[1, i] = port_return
        results[2, i] = sharpe

    # Plot random portfolios, colored by Sharpe ratio
    plt.figure(figsize=(10, 6))
    plt.scatter(results[0], results[1], c=results[2], cmap='viridis', alpha=0.5, s=10)
    plt.colorbar(label='Sharpe Ratio')

    # Highlight the optimizer's answer
    opt_weights, opt_return, opt_vol, opt_sharpe = optimize_portfolio(tickers, risk_free_rate)
    plt.scatter(opt_vol, opt_return, c='red', marker='*', s=400, label='Optimal Portfolio')

    plt.title('Efficient Frontier: Random Portfolios vs. Optimal')
    plt.xlabel('Annual Volatility (Risk)')
    plt.ylabel('Expected Annual Return')
    plt.legend()
    plt.tight_layout()
    plt.show()




print("\n--- Correlation Matrix ---")
print(get_correlation_matrix(TICKERS))
plot_correlation_heatmap(TICKERS)

print("\n--- Portfolio Optimization ---")
display_optimal_portfolio(TICKERS)
plot_efficient_frontier(TICKERS)

print("\n--- imporvement from Basic 20% weighted portfolio ---")
compare_to_equal_weight(TICKERS)

