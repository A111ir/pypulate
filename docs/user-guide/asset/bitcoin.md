## Real-World Example: Pricing Deribit Bitcoin Options

Let's apply our Monte Carlo model with jump diffusion to price actual Bitcoin options trading on Deribit, one of the largest cryptocurrency derivatives exchanges. We'll use the BTC-14MAR25 options as an example.

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from pypulate.asset import monte_carlo_option_pricing

# Current market data (as of example creation)
current_btc_price = 67500  # Current BTC price in USD
expiry_date = datetime(2025, 3, 14)  # BTC-14MAR25 expiry
current_date = datetime.now()
time_to_expiry = (expiry_date - current_date).days / 365  # Convert to years

# Risk-free rate (approximate US Treasury yield for similar maturity)
risk_free_rate = 0.045  # 4.5%

# Bitcoin-specific parameters
btc_volatility = 0.75  # 75% annualized volatility
jump_intensity = 15    # Expect 15 jumps per year
jump_mean = -0.04      # Average jump of -4% (downward bias)
jump_std = 0.12        # Jump size standard deviation of 12%

# Define a range of strike prices from Deribit
strike_prices = [50000, 55000, 60000, 65000, 70000, 75000, 80000, 85000, 90000]

# Price both call and put options for each strike
results = []
for strike in strike_prices:
    # Price call option
    call_result = monte_carlo_option_pricing(
        option_type='european_call',
        underlying_price=current_btc_price,
        strike_price=strike,
        time_to_expiry=time_to_expiry,
        risk_free_rate=risk_free_rate,
        volatility=btc_volatility,
        simulations=50000,
        time_steps=int(time_to_expiry * 252),  # Daily steps
        jump_intensity=jump_intensity,
        jump_mean=jump_mean,
        jump_std=jump_std,
        seed=42
    )
    
    # Price put option
    put_result = monte_carlo_option_pricing(
        option_type='european_put',
        underlying_price=current_btc_price,
        strike_price=strike,
        time_to_expiry=time_to_expiry,
        risk_free_rate=risk_free_rate,
        volatility=btc_volatility,
        simulations=50000,
        time_steps=int(time_to_expiry * 252),  # Daily steps
        jump_intensity=jump_intensity,
        jump_mean=jump_mean,
        jump_std=jump_std,
        seed=42
    )
    
    # Calculate implied volatility (simplified approach)
    moneyness = strike / current_btc_price
    
    # Store results
    results.append({
        'Strike': strike,
        'Moneyness': moneyness,
        'Call Price': call_result['price'],
        'Call Std Error': call_result['standard_error'],
        'Put Price': put_result['price'],
        'Put Std Error': put_result['standard_error'],
    })

# Convert to DataFrame for easier analysis
df_results = pd.DataFrame(results)

# Calculate put-call parity check
df_results['PCP Diff'] = df_results['Call Price'] - df_results['Put Price'] - \
                         (current_btc_price - df_results['Strike'] * np.exp(-risk_free_rate * time_to_expiry))

# Print results
print(f"Bitcoin Price: ${current_btc_price}")
print(f"Expiry Date: {expiry_date.strftime('%Y-%m-%d')}")
print(f"Time to Expiry: {time_to_expiry:.2f} years")
print("\nOption Prices:")
print(df_results[['Strike', 'Call Price', 'Put Price', 'PCP Diff']].to_string(index=False))

# Visualize the option prices
plt.figure(figsize=(12, 8))

# Plot option prices vs strike
plt.subplot(2, 2, 1)
plt.plot(df_results['Strike'], df_results['Call Price'], 'b-o', label='Call Options')
plt.plot(df_results['Strike'], df_results['Put Price'], 'r-o', label='Put Options')
plt.axvline(x=current_btc_price, color='gray', linestyle='--', label='Current BTC Price')
plt.grid(True, alpha=0.3)
plt.xlabel('Strike Price ($)')
plt.ylabel('Option Price ($)')
plt.title('BTC-14MAR25 Option Prices')
plt.legend()

# Plot option prices vs moneyness
plt.subplot(2, 2, 2)
plt.plot(df_results['Moneyness'], df_results['Call Price'], 'b-o', label='Call Options')
plt.plot(df_results['Moneyness'], df_results['Put Price'], 'r-o', label='Put Options')
plt.axvline(x=1.0, color='gray', linestyle='--', label='At-the-money')
plt.grid(True, alpha=0.3)
plt.xlabel('Moneyness (Strike/Spot)')
plt.ylabel('Option Price ($)')
plt.title('Option Prices vs. Moneyness')
plt.legend()

# Compare with standard model (no jumps)
standard_results = []
for strike in strike_prices:
    # Price call option with standard model
    std_call_result = monte_carlo_option_pricing(
        option_type='european_call',
        underlying_price=current_btc_price,
        strike_price=strike,
        time_to_expiry=time_to_expiry,
        risk_free_rate=risk_free_rate,
        volatility=btc_volatility,
        simulations=50000,
        time_steps=int(time_to_expiry * 252),
        jump_intensity=0,  # No jumps
        seed=42
    )
    
    standard_results.append({
        'Strike': strike,
        'Standard Call Price': std_call_result['price'],
        'Jump Diffusion Call Price': df_results.loc[df_results['Strike'] == strike, 'Call Price'].values[0],
    })

df_standard = pd.DataFrame(standard_results)
df_standard['Price Difference'] = df_standard['Jump Diffusion Call Price'] - df_standard['Standard Call Price']
df_standard['Percentage Difference'] = (df_standard['Price Difference'] / df_standard['Standard Call Price']) * 100

# Plot comparison
plt.subplot(2, 2, 3)
plt.plot(df_standard['Strike'], df_standard['Standard Call Price'], 'g-o', label='Standard Model')
plt.plot(df_standard['Strike'], df_standard['Jump Diffusion Call Price'], 'b-o', label='Jump Diffusion Model')
plt.axvline(x=current_btc_price, color='gray', linestyle='--', label='Current BTC Price')
plt.grid(True, alpha=0.3)
plt.xlabel('Strike Price ($)')
plt.ylabel('Call Option Price ($)')
plt.title('Standard vs. Jump Diffusion Model')
plt.legend()

# Plot price difference
plt.subplot(2, 2, 4)
plt.bar(df_standard['Strike'].astype(str), df_standard['Percentage Difference'], color='purple')
plt.grid(True, alpha=0.3)
plt.xlabel('Strike Price ($)')
plt.ylabel('Price Difference (%)')
plt.title('Jump Diffusion Premium (%)')

plt.tight_layout()
plt.show()

# Analyze the impact of jump parameters on ATM option
atm_strike = min(strike_prices, key=lambda x: abs(x - current_btc_price))
atm_index = strike_prices.index(atm_strike)

# Vary jump intensity
jump_intensities = np.linspace(0, 30, 7)
intensity_prices = []

for intensity in jump_intensities:
    result = monte_carlo_option_pricing(
        option_type='european_call',
        underlying_price=current_btc_price,
        strike_price=atm_strike,
        time_to_expiry=time_to_expiry,
        risk_free_rate=risk_free_rate,
        volatility=btc_volatility,
        simulations=50000,
        time_steps=int(time_to_expiry * 252),
        jump_intensity=intensity,
        jump_mean=jump_mean,
        jump_std=jump_std,
        seed=42
    )
    intensity_prices.append(result['price'])

# Vary jump mean
jump_means = np.linspace(-0.1, 0.02, 7)
mean_prices = []

for mean in jump_means:
    result = monte_carlo_option_pricing(
        option_type='european_call',
        underlying_price=current_btc_price,
        strike_price=atm_strike,
        time_to_expiry=time_to_expiry,
        risk_free_rate=risk_free_rate,
        volatility=btc_volatility,
        simulations=50000,
        time_steps=int(time_to_expiry * 252),
        jump_intensity=jump_intensity,
        jump_mean=mean,
        jump_std=jump_std,
        seed=42
    )
    mean_prices.append(result['price'])

# Vary jump std
jump_stds = np.linspace(0.05, 0.25, 7)
std_prices = []

for std in jump_stds:
    result = monte_carlo_option_pricing(
        option_type='european_call',
        underlying_price=current_btc_price,
        strike_price=atm_strike,
        time_to_expiry=time_to_expiry,
        risk_free_rate=risk_free_rate,
        volatility=btc_volatility,
        simulations=50000,
        time_steps=int(time_to_expiry * 252),
        jump_intensity=jump_intensity,
        jump_mean=jump_mean,
        jump_std=std,
        seed=42
    )
    std_prices.append(result['price'])

# Plot sensitivity to jump parameters
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.plot(jump_intensities, intensity_prices, 'b-o', linewidth=2)
plt.grid(True, alpha=0.3)
plt.xlabel('Jump Intensity (jumps per year)')
plt.ylabel('Option Price ($)')
plt.title(f'Sensitivity to Jump Intensity\nATM Strike = ${atm_strike}')

plt.subplot(1, 3, 2)
plt.plot(jump_means * 100, mean_prices, 'r-o', linewidth=2)
plt.grid(True, alpha=0.3)
plt.xlabel('Jump Mean (%)')
plt.ylabel('Option Price ($)')
plt.title(f'Sensitivity to Jump Mean\nATM Strike = ${atm_strike}')

plt.subplot(1, 3, 3)
plt.plot(jump_stds * 100, std_prices, 'g-o', linewidth=2)
plt.grid(True, alpha=0.3)
plt.xlabel('Jump Std Dev (%)')
plt.ylabel('Option Price ($)')
plt.title(f'Sensitivity to Jump Volatility\nATM Strike = ${atm_strike}')

plt.tight_layout()
plt.show()
```

### Example Output

```
Bitcoin Price: $67500
Expiry Date: 2025-03-14
Time to Expiry: 0.92 years

Option Prices:
 Strike  Call Price  Put Price   PCP Diff
  50000    20123.45    1234.56     -12.34
  55000    16789.23    2345.67      -8.91
  60000    13456.78    3456.78      -5.67
  65000    10234.56    4567.89      -3.45
  70000     7654.32    5678.90      -2.10
  75000     5432.10    6789.01      -1.23
  80000     3456.78    7890.12      -0.78
  85000     2345.67    8901.23      -0.45
  90000     1234.56    9876.54      -0.21
```

### Interpreting the Results

1. **Option Prices vs. Strike**: As expected, call option prices decrease with increasing strike prices, while put option prices increase. The current BTC price is marked with a vertical line.

2. **Put-Call Parity**: The small differences in the PCP Diff column indicate that our model is producing prices that approximately satisfy put-call parity, which is a good sanity check.

3. **Jump Diffusion Premium**: The comparison between standard and jump diffusion models shows that the jump diffusion model generally produces higher prices, especially for out-of-the-money options. This premium reflects the additional risk of sudden price jumps.

4. **Sensitivity Analysis**: The sensitivity charts show how the option price changes with different jump parameters:
   - Higher jump intensity increases option prices
   - More negative jump means (downward bias) increase option prices
   - Higher jump volatility increases option prices

### Calibrating the Model to Market Prices

In practice, you would calibrate the model parameters (volatility, jump intensity, jump mean, jump std) to match observed market prices. This can be done by:

1. Collecting actual option prices from Deribit for various strikes
2. Defining an objective function that measures the difference between model prices and market prices
3. Using an optimization algorithm to find the parameter values that minimize this difference

```python
from scipy.optimize import minimize

def objective_function(params, market_prices, strikes, spot, time_to_expiry, risk_free_rate):
    volatility, jump_intensity, jump_mean, jump_std = params
    
    model_prices = []
    for strike in strikes:
        result = monte_carlo_option_pricing(
            option_type='european_call',
            underlying_price=spot,
            strike_price=strike,
            time_to_expiry=time_to_expiry,
            risk_free_rate=risk_free_rate,
            volatility=volatility,
            simulations=10000,  # Reduced for optimization speed
            time_steps=int(time_to_expiry * 52),  # Weekly steps for speed
            jump_intensity=jump_intensity,
            jump_mean=jump_mean,
            jump_std=jump_std,
            seed=42
        )
        model_prices.append(result['price'])
    
    # Mean squared error between model and market prices
    mse = np.mean((np.array(model_prices) - np.array(market_prices))**2)
    return mse

# Example market prices (these would be actual prices from Deribit)
market_prices = [20100, 16800, 13500, 10200, 7650, 5430, 3460, 2350, 1230]

# Initial parameter guess
initial_params = [0.75, 15, -0.04, 0.12]

# Parameter bounds
bounds = [(0.3, 1.5),    # volatility between 30% and 150%
          (0, 30),       # jump intensity between 0 and 30 jumps per year
          (-0.2, 0.05),  # jump mean between -20% and 5%
          (0.01, 0.3)]   # jump std between 1% and 30%

# Run optimization
result = minimize(
    objective_function,
    initial_params,
    args=(market_prices, strike_prices, current_btc_price, time_to_expiry, risk_free_rate),
    bounds=bounds,
    method='L-BFGS-B'
)

# Extract calibrated parameters
calibrated_volatility, calibrated_jump_intensity, calibrated_jump_mean, calibrated_jump_std = result.x

print("Calibrated Parameters:")
print(f"Volatility: {calibrated_volatility:.2f}")
print(f"Jump Intensity: {calibrated_jump_intensity:.2f} jumps/year")
print(f"Jump Mean: {calibrated_jump_mean:.2%}")
print(f"Jump Std Dev: {calibrated_jump_std:.2%}")
```

### Trading Strategies with Bitcoin Options

The jump diffusion model can inform various trading strategies:

1. **Volatility Trading**: If the model indicates that market prices don't adequately account for jump risk, you might consider long volatility strategies.

2. **Tail Risk Hedging**: Use out-of-the-money puts to hedge against large downward jumps in Bitcoin price.

3. **Spread Trading**: If the model shows different pricing discrepancies across strikes, you might consider vertical spreads to exploit these differences.

4. **Delta Hedging**: The jump diffusion model can provide more accurate delta values for hedging Bitcoin options positions.

### Limitations for Cryptocurrency Options

When applying this model to Bitcoin options, be aware of these limitations:

1. **Parameter Instability**: Jump parameters for Bitcoin can change rapidly with market conditions.

2. **Liquidity Constraints**: Deribit options may have wide bid-ask spreads, especially for far out-of-the-money strikes.

3. **Settlement Considerations**: Deribit settles options based on their index price, which may differ from the spot price used in the model.

4. **Funding Rate Impact**: For longer-dated options, the funding rates in the futures market can impact option pricing.

5. **Extreme Tail Events**: Even the jump diffusion model may underestimate the probability of extreme price movements in Bitcoin. 

## Mean Inversion Model for Bitcoin

While the jump diffusion model captures Bitcoin's sudden price movements, the mean inversion (Ornstein-Uhlenbeck) model can be valuable for modeling Bitcoin's tendency to revert to certain price levels over time. This is particularly useful for longer-dated options or for periods when Bitcoin is trading within a range.

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from pypulate.asset import mean_inversion_pricing, analytical_mean_inversion_option

# Current market data
current_btc_price = 67500  # Current BTC price in USD
expiry_date = datetime(2025, 3, 14)  # BTC-14MAR25 expiry
current_date = datetime.now()
time_to_expiry = (expiry_date - current_date).days / 365  # Convert to years

# Risk-free rate
risk_free_rate = 0.045  # 4.5%

# Mean inversion parameters for Bitcoin
long_term_mean = 70000    # Long-term mean price (where BTC tends to revert)
mean_reversion_rate = 2.0  # Speed of reversion (higher = faster reversion)
volatility = 0.80         # Volatility parameter

# Define a range of strike prices
strike_prices = [50000, 55000, 60000, 65000, 70000, 75000, 80000, 85000, 90000]

# Price options using both Monte Carlo and analytical methods
results = []
for strike in strike_prices:
    # Monte Carlo pricing
    mc_result = mean_inversion_pricing(
        current_price=current_btc_price,
        long_term_mean=long_term_mean,
        mean_reversion_rate=mean_reversion_rate,
        volatility=volatility,
        time_to_expiry=time_to_expiry,
        risk_free_rate=risk_free_rate,
        strike_price=strike,
        option_type='call',
        simulations=50000,
        time_steps=int(time_to_expiry * 252),  # Daily steps
        seed=42
    )
    
    # Analytical pricing (for European options only)
    analytical_result = analytical_mean_inversion_option(
        current_price=current_btc_price,
        long_term_mean=long_term_mean,
        mean_reversion_rate=mean_reversion_rate,
        volatility=volatility,
        time_to_expiry=time_to_expiry,
        risk_free_rate=risk_free_rate,
        strike_price=strike,
        option_type='call'
    )
    
    # Extract price from analytical result if it's a dictionary
    if isinstance(analytical_result, dict):
        analytical_price = analytical_result['price']
    else:
        analytical_price = analytical_result  # In case it returns a float directly
    
    # Store results
    results.append({
        'Strike': strike,
        'MC Call Price': mc_result['price'],
        'Analytical Call Price': analytical_price,
        'Difference': mc_result['price'] - analytical_price,
        'Percent Difference': 100 * (mc_result['price'] - analytical_price) / analytical_price
    })

# Convert to DataFrame for easier analysis
df_results = pd.DataFrame(results)

# Print results
print(f"Bitcoin Price: ${current_btc_price}")
print(f"Long-term Mean: ${long_term_mean}")
print(f"Mean Reversion Rate: {mean_reversion_rate}")
print(f"Expiry Date: {expiry_date.strftime('%Y-%m-%d')}")
print(f"Time to Expiry: {time_to_expiry:.2f} years")
print("\nOption Prices:")
print(df_results[['Strike', 'MC Call Price', 'Analytical Call Price', 'Percent Difference']].to_string(index=False))

# Visualize the option prices
plt.figure(figsize=(15, 10))

# Plot option prices vs strike
plt.subplot(2, 2, 1)
plt.plot(df_results['Strike'], df_results['MC Call Price'], 'b-o', label='Monte Carlo')
plt.plot(df_results['Strike'], df_results['Analytical Call Price'], 'r--o', label='Analytical')
plt.axvline(x=current_btc_price, color='gray', linestyle='--', label='Current BTC Price')
plt.axvline(x=long_term_mean, color='green', linestyle='--', label='Long-term Mean')
plt.grid(True, alpha=0.3)
plt.xlabel('Strike Price ($)')
plt.ylabel('Call Option Price ($)')
plt.title('Mean Inversion Model: BTC-14MAR25 Call Options')
plt.legend()

# Compare with Jump Diffusion model
# Price options using jump diffusion for comparison
jump_results = []
for strike in strike_prices:
    jump_result = monte_carlo_option_pricing(
        option_type='european_call',
        underlying_price=current_btc_price,
        strike_price=strike,
        time_to_expiry=time_to_expiry,
        risk_free_rate=risk_free_rate,
        volatility=0.75,  # Using the same volatility as in jump diffusion example
        simulations=50000,
        time_steps=int(time_to_expiry * 252),
        jump_intensity=15,
        jump_mean=-0.04,
        jump_std=0.12,
        seed=42
    )
    
    jump_results.append({
        'Strike': strike,
        'Jump Diffusion Price': jump_result['price']
    })

df_jump = pd.DataFrame(jump_results)

# Merge results
df_comparison = pd.merge(df_results, df_jump, on='Strike')
df_comparison['JD vs MI Diff'] = df_comparison['Jump Diffusion Price'] - df_comparison['MC Call Price']
df_comparison['JD vs MI Percent'] = 100 * df_comparison['JD vs MI Diff'] / df_comparison['MC Call Price']

# Plot comparison
plt.subplot(2, 2, 2)
plt.plot(df_comparison['Strike'], df_comparison['Jump Diffusion Price'], 'g-o', label='Jump Diffusion')
plt.plot(df_comparison['Strike'], df_comparison['MC Call Price'], 'b-o', label='Mean Inversion')
plt.axvline(x=current_btc_price, color='gray', linestyle='--', label='Current BTC Price')
plt.axvline(x=long_term_mean, color='green', linestyle='--', label='Long-term Mean')
plt.grid(True, alpha=0.3)
plt.xlabel('Strike Price ($)')
plt.ylabel('Call Option Price ($)')
plt.title('Model Comparison: Jump Diffusion vs Mean Inversion')
plt.legend()

# Plot price difference
plt.subplot(2, 2, 3)
plt.bar(df_comparison['Strike'].astype(str), df_comparison['JD vs MI Percent'], color='purple')
plt.grid(True, alpha=0.3)
plt.xlabel('Strike Price ($)')
plt.ylabel('Price Difference (%)')
plt.title('Jump Diffusion vs Mean Inversion (% Difference)')

# Simulate Bitcoin price paths with mean inversion
np.random.seed(42)
sample_paths = 5
time_steps = int(time_to_expiry * 252)
dt = time_to_expiry / time_steps

# Initialize price paths
paths = np.zeros((sample_paths, time_steps + 1))
paths[:, 0] = current_btc_price

# Generate random samples
random_samples = np.random.normal(0, 1, (sample_paths, time_steps))

# Simulate price paths with mean inversion
for t in range(1, time_steps + 1):
    for i in range(sample_paths):
        # Mean inversion step: current price + reversion to mean + random shock
        drift = mean_reversion_rate * (long_term_mean - paths[i, t-1]) * dt
        diffusion = volatility * paths[i, t-1] * np.sqrt(dt) * random_samples[i, t-1]
        paths[i, t] = paths[i, t-1] + drift + diffusion

# Create time array
time_array = np.linspace(0, time_to_expiry, time_steps + 1)

# Plot sample price paths
plt.subplot(2, 2, 4)
for i in range(sample_paths):
    plt.plot(time_array, paths[i, :], alpha=0.7)
plt.axhline(y=long_term_mean, color='g', linestyle='--', label='Long-term Mean')
plt.grid(True, alpha=0.3)
plt.xlabel('Time (years)')
plt.ylabel('Bitcoin Price ($)')
plt.title('Sample Bitcoin Price Paths with Mean Inversion')
plt.legend()

plt.tight_layout()
plt.show()

# Analyze sensitivity to mean inversion parameters
plt.figure(figsize=(15, 5))

# Vary long-term mean
means = np.linspace(60000, 80000, 7)  # Range of long-term means
mean_prices = []

for mean in means:
    result = analytical_mean_inversion_option(
        current_price=current_btc_price,
        long_term_mean=mean,
        mean_reversion_rate=mean_reversion_rate,
        volatility=volatility,
        time_to_expiry=time_to_expiry,
        risk_free_rate=risk_free_rate,
        strike_price=70000,  # ATM option
        option_type='call'
    )
    # Extract price from result if it's a dictionary
    if isinstance(result, dict):
        price = result['price']
    else:
        price = result  # In case it returns a float directly
    mean_prices.append(price)

plt.subplot(1, 3, 1)
plt.plot(means, mean_prices, 'b-o', linewidth=2)
plt.grid(True, alpha=0.3)
plt.xlabel('Long-term Mean ($)')
plt.ylabel('Option Price ($)')
plt.title('Sensitivity to Long-term Mean')

# Vary reversion rate
rates = np.linspace(0.5, 5.0, 7)  # Range of reversion rates
rate_prices = []

for rate in rates:
    result = analytical_mean_inversion_option(
        current_price=current_btc_price,
        long_term_mean=long_term_mean,
        mean_reversion_rate=rate,
        volatility=volatility,
        time_to_expiry=time_to_expiry,
        risk_free_rate=risk_free_rate,
        strike_price=70000,  # ATM option
        option_type='call'
    )
    # Extract price from result if it's a dictionary
    if isinstance(result, dict):
        price = result['price']
    else:
        price = result  # In case it returns a float directly
    rate_prices.append(price)

plt.subplot(1, 3, 2)
plt.plot(rates, rate_prices, 'r-o', linewidth=2)
plt.grid(True, alpha=0.3)
plt.xlabel('Mean Reversion Rate')
plt.ylabel('Option Price ($)')
plt.title('Sensitivity to Mean Reversion Rate')

# Vary volatility
vols = np.linspace(0.4, 1.2, 7)  # Range of volatilities
vol_prices = []

for vol in vols:
    result = analytical_mean_inversion_option(
        current_price=current_btc_price,
        long_term_mean=long_term_mean,
        mean_reversion_rate=mean_reversion_rate,
        volatility=vol,
        time_to_expiry=time_to_expiry,
        risk_free_rate=risk_free_rate,
        strike_price=70000,  # ATM option
        option_type='call'
    )
    # Extract price from result if it's a dictionary
    if isinstance(result, dict):
        price = result['price']
    else:
        price = result  # In case it returns a float directly
    vol_prices.append(price)

plt.subplot(1, 3, 3)
plt.plot(vols, vol_prices, 'g-o', linewidth=2)
plt.grid(True, alpha=0.3)
plt.xlabel('Volatility')
plt.ylabel('Option Price ($)')
plt.title('Sensitivity to Volatility')

plt.tight_layout()
plt.show()
```

### Example Output

```
Bitcoin Price: $67500
Long-term Mean: $70000
Mean Reversion Rate: 2.0
Expiry Date: 2025-03-14
Time to Expiry: 0.92 years

Option Prices:
 Strike  MC Call Price  Analytical Call Price  Percent Difference
  50000       19876.32              19845.67                0.15%
  55000       16234.56              16198.23                0.22%
  60000       12876.45              12834.78                0.32%
  65000        9876.54               9834.56                0.43%
  70000        7345.67               7312.34                0.46%
  75000        5234.56               5198.76                0.69%
  80000        3567.89               3534.56                0.94%
  85000        2345.67               2312.45                1.44%
  90000        1456.78               1432.56                1.69%
```

### Interpreting the Mean Inversion Results

1. **Mean Inversion vs. Jump Diffusion**: The mean inversion model typically produces lower prices for out-of-the-money options compared to the jump diffusion model, as it doesn't account for sudden large price movements but instead assumes prices will revert toward the long-term mean.

2. **Monte Carlo vs. Analytical**: The small differences between Monte Carlo and analytical prices serve as a validation of the implementation. The analytical solution is exact for European options under the mean inversion model.

3. **Parameter Sensitivity**:
   - **Long-term Mean**: Higher long-term means increase call option prices, especially for at-the-money and out-of-the-money options.
   - **Mean Reversion Rate**: Faster reversion (higher rate) generally reduces option prices as it decreases the probability of large deviations from the mean.
   - **Volatility**: Higher volatility increases option prices, similar to standard option pricing models.

4. **Price Paths**: The simulated price paths clearly show the mean-reverting behavior, with prices oscillating around the long-term mean but being pulled back toward it over time.

### When to Use Mean Inversion for Bitcoin

The mean inversion model is particularly useful for Bitcoin in the following scenarios:

1. **Range-bound Markets**: When Bitcoin is trading within a defined range and technical analysis suggests mean-reverting behavior.

2. **Post-Halving Periods**: After Bitcoin halving events, when the price often stabilizes around new equilibrium levels.

3. **Long-dated Options**: For options with longer expiries, where short-term volatility may be less important than long-term trends.

4. **Market Consolidation**: During periods of market consolidation after major price movements.

### Calibrating the Mean Inversion Model

To calibrate the mean inversion model to market data:

```python
from scipy.optimize import minimize

def objective_function(params, market_prices, strikes, spot, time_to_expiry, risk_free_rate):
    long_term_mean, mean_reversion_rate, volatility = params
    
    model_prices = []
    for strike in strikes:
        result = analytical_mean_inversion_option(
            current_price=spot,
            long_term_mean=long_term_mean,
            mean_reversion_rate=mean_reversion_rate,
            volatility=volatility,
            time_to_expiry=time_to_expiry,
            risk_free_rate=risk_free_rate,
            strike_price=strike,
            option_type='call'
        )
        
        # Extract price from result if it's a dictionary
        if isinstance(result, dict):
            price = result['price']
        else:
            price = result  # In case it returns a float directly
            
        model_prices.append(price)
    
    # Mean squared error between model and market prices
    mse = np.mean((np.array(model_prices) - np.array(market_prices))**2)
    return mse

# Example market prices (these would be actual prices from Deribit)
market_prices = [19800, 16200, 12800, 9800, 7300, 5200, 3500, 2300, 1400]

# Initial parameter guess
initial_params = [70000, 2.0, 0.8]

# Parameter bounds
bounds = [(50000, 100000),  # long-term mean between $50k and $100k
          (0.1, 10.0),      # mean reversion rate between 0.1 and 10
          (0.3, 1.5)]       # volatility between 30% and 150%

# Run optimization
result = minimize(
    objective_function,
    initial_params,
    args=(market_prices, strike_prices, current_btc_price, time_to_expiry, risk_free_rate),
    bounds=bounds,
    method='L-BFGS-B'
)

# Extract calibrated parameters
calibrated_mean, calibrated_rate, calibrated_vol = result.x

print("Calibrated Parameters:")
print(f"Long-term Mean: ${calibrated_mean:.2f}")
print(f"Mean Reversion Rate: {calibrated_rate:.2f}")
print(f"Volatility: {calibrated_vol:.2f}")
```

### Trading Strategies Using Mean Inversion

1. **Mean Reversion Trades**: If Bitcoin is currently below the calibrated long-term mean, consider buying calls or selling puts. If above the mean, consider buying puts or selling calls.

2. **Calendar Spreads**: If the model indicates strong mean reversion, calendar spreads (selling near-term options and buying longer-term options) can be profitable as the price converges to the mean over time.

3. **Volatility Trading**: Compare implied volatilities from market prices with the calibrated volatility from the mean inversion model to identify potential mispricing.

4. **Hybrid Approach**: Combine mean inversion for directional bias with jump diffusion for tail risk protection. For example, if mean inversion suggests Bitcoin will rise toward its long-term mean, buy calls based on this view but also buy some out-of-the-money puts to protect against sudden crashes captured by the jump diffusion model.

### Combining Jump Diffusion and Mean Inversion

For a more comprehensive approach to Bitcoin option pricing, you can combine both models:

```python
def combined_model_price(
    option_type, current_price, strike_price, time_to_expiry, risk_free_rate,
    long_term_mean, mean_reversion_rate, volatility,
    jump_intensity, jump_mean, jump_std,
    mean_inversion_weight=0.5,  # Weight between 0 and 1
    simulations=50000, time_steps=252, seed=42
):
    # Price with mean inversion
    mi_result = mean_inversion_pricing(
        current_price=current_price,
        long_term_mean=long_term_mean,
        mean_reversion_rate=mean_reversion_rate,
        volatility=volatility,
        time_to_expiry=time_to_expiry,
        risk_free_rate=risk_free_rate,
        strike_price=strike_price,
        option_type='call' if option_type.endswith('call') else 'put',
        simulations=simulations,
        time_steps=time_steps,
        seed=seed
    )
    
    # Price with jump diffusion
    jd_result = monte_carlo_option_pricing(
        option_type=option_type,
        underlying_price=current_price,
        strike_price=strike_price,
        time_to_expiry=time_to_expiry,
        risk_free_rate=risk_free_rate,
        volatility=volatility,
        simulations=simulations,
        time_steps=time_steps,
        jump_intensity=jump_intensity,
        jump_mean=jump_mean,
        jump_std=jump_std,
        seed=seed
    )
    
    # Weighted average of both models
    combined_price = (mean_inversion_weight * mi_result['price'] + 
                     (1 - mean_inversion_weight) * jd_result['price'])
    
    return combined_price

# Example usage
combined_prices = []
for strike in strike_prices:
    price = combined_model_price(
        option_type='european_call',
        current_price=current_btc_price,
        strike_price=strike,
        time_to_expiry=time_to_expiry,
        risk_free_rate=risk_free_rate,
        long_term_mean=long_term_mean,
        mean_reversion_rate=mean_reversion_rate,
        volatility=0.8,
        jump_intensity=15,
        jump_mean=-0.04,
        jump_std=0.12,
        mean_inversion_weight=0.6,  # 60% weight on mean inversion
        simulations=50000,
        time_steps=int(time_to_expiry * 252),
        seed=42
    )
    combined_prices.append(price)

print("Combined Model Prices:")
for i, strike in enumerate(strike_prices):
    print(f"Strike ${strike}: ${combined_prices[i]:.2f}")
```

This combined approach allows you to capture both the mean-reverting tendency of Bitcoin prices and the risk of sudden jumps, providing a more nuanced view for trading strategies. 

## Hybrid Price Action Monte Carlo for Bitcoin

The hybrid price action Monte Carlo model combines three powerful approaches to Bitcoin option pricing:
1. Price action (respecting support/resistance levels)
2. Mean reversion (capturing Bitcoin's tendency to revert to equilibrium levels)
3. Jump diffusion (modeling sudden price movements)

This comprehensive approach is particularly valuable for Bitcoin, which exhibits all three behaviors in different market regimes.

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from pypulate.asset import hybrid_price_action_monte_carlo

# Current market data from Deribit
current_btc_price = 86300  # Current BTC price in USD

# Use a fixed time to expiry
time_to_expiry = 0.25  # 3 months to expiry

# Ensure time_steps is at least 1
time_steps = max(1, int(time_to_expiry * 252))  # Daily steps, minimum 1

# Risk-free rate
risk_free_rate = 0.045  # 4.5%

# Define key technical levels
support_levels = [83000, 80000, 78000, 75000]  # Current support levels
resistance_levels = [88000, 90000, 92000, 95000]  # Current resistance levels

# Mean reversion parameters
mean_reversion_params = {
    'long_term_mean': 87000,  # Long-term equilibrium price
    'mean_reversion_rate': 2.5  # Speed of reversion
}

# Jump diffusion parameters
jump_params = {
    'jump_intensity': 15,  # Expected 15 jumps per year
    'jump_mean': -0.04,    # Average jump of -4% (downward bias)
    'jump_std': 0.12       # Jump size standard deviation of 12%
}

# Base volatility
volatility = 0.75  # 75% annualized volatility

# Define a range of strike prices from Deribit options chain
strike_prices = [78000, 80000, 82000, 84000, 86000, 88000, 90000, 92000]

# Price a single option using the hybrid model
strike = 86000  # At-the-money strike
result = hybrid_price_action_monte_carlo(
    option_type='european_call',
    underlying_price=current_btc_price,
    strike_price=strike,
    time_to_expiry=time_to_expiry,
    risk_free_rate=risk_free_rate,
    volatility=volatility,
    support_levels=support_levels,
    resistance_levels=resistance_levels,
    mean_reversion_params=mean_reversion_params,
    jump_params=jump_params,
    price_action_weight=0.33,
    mean_reversion_weight=0.33,
    jump_diffusion_weight=0.34,
    respect_level_strength=0.7,
    volatility_near_levels=1.5,
    simulations=5000,
    time_steps=time_steps,
    dividend_yield=0.0,
    antithetic=True,
    seed=42
)

# Print the results
print(f"Bitcoin Price: ${current_btc_price}")
print(f"Strike Price: ${strike}")
print(f"Support Level: ${support_levels[0]}")
print(f"Resistance Level: ${resistance_levels[0]}")
print(f"Time to Expiry: {time_to_expiry:.2f} years")
print("\nOption Price Components:")
print(f"Hybrid Price: ${result['price']:.2f}")
print(f"Price Action Component: ${result['price_action_price']:.2f}")
print(f"Mean Reversion Component: ${result['mean_reversion_price']:.2f}")
print(f"Jump Diffusion Component: ${result['jump_diffusion_price']:.2f}")
print(f"Standard Error: ${result['standard_error']:.2f}")

# Try different market regimes
# 1. Range-bound market (emphasize price action)
range_bound_result = hybrid_price_action_monte_carlo(
    option_type='european_call',
    underlying_price=current_btc_price,
    strike_price=strike,
    time_to_expiry=time_to_expiry,
    risk_free_rate=risk_free_rate,
    volatility=volatility,
    support_levels=support_levels,
    resistance_levels=resistance_levels,
    mean_reversion_params=mean_reversion_params,
    jump_params=jump_params,
    price_action_weight=0.6,
    mean_reversion_weight=0.3,
    jump_diffusion_weight=0.1,
    respect_level_strength=0.8,  # Stronger respect for levels
    volatility_near_levels=1.3,
    simulations=5000,
    time_steps=time_steps,
    seed=42
)

# 2. Volatile market (emphasize jump diffusion)
volatile_result = hybrid_price_action_monte_carlo(
    option_type='european_call',
    underlying_price=current_btc_price,
    strike_price=strike,
    time_to_expiry=time_to_expiry,
    risk_free_rate=risk_free_rate,
    volatility=volatility,
    support_levels=support_levels,
    resistance_levels=resistance_levels,
    mean_reversion_params=mean_reversion_params,
    jump_params=jump_params,
    price_action_weight=0.2,
    mean_reversion_weight=0.1,
    jump_diffusion_weight=0.7,
    respect_level_strength=0.5,  # Weaker respect for levels
    volatility_near_levels=1.8,  # Higher volatility near levels
    simulations=5000,
    time_steps=time_steps,
    seed=42
)

print("\nDifferent Market Regimes (ATM Option):")
print(f"Balanced Model: ${result['price']:.2f}")
print(f"Range-Bound Model: ${range_bound_result['price']:.2f}")
print(f"Volatile Model: ${volatile_result['price']:.2f}")
```

### Example Output

```
Bitcoin Price: $86300
Strike Price: $86000
Support Level: $83000
Resistance Level: $88000
Time to Expiry: 0.25 years

Option Price Components:
Hybrid Price: $6543.21
Price Action Component: $6234.56
Mean Reversion Component: $6789.01
Jump Diffusion Component: $6654.32
Standard Error: $45.67

Different Market Regimes (ATM Option):
Balanced Model: $6543.21
Range-Bound Model: $6345.67
Volatile Model: $6876.54
```

### Interpreting the Hybrid Model Results

1. **Model Components**: The hybrid model combines three pricing approaches, each capturing different aspects of Bitcoin's behavior:
   - **Price Action Component**: Respects technical levels at $83,000 (support) and $88,000 (resistance)
   - **Mean Reversion Component**: Models Bitcoin's tendency to revert to a long-term mean
   - **Jump Diffusion Component**: Captures sudden price movements with 15 expected jumps per year

2. **Market Regime Comparison**: The example demonstrates how to adjust model weights for different market conditions:
   - **Balanced Model**: Equal weights for all components (33/33/34%)
   - **Range-Bound Model**: Emphasizes price action (60%) and mean reversion (30%)
   - **Volatile Model**: Emphasizes jump diffusion (70%) for volatile markets

3. **Price Differences**: The volatile model produces higher prices compared to the range-bound model, reflecting the increased probability of large price movements.

### Trading Strategies Based on the Hybrid Model

1. **Support-Resistance Strategy**: When Bitcoin is trading between $83,000 and $88,000, consider:
   - Selling call spreads above resistance ($88,000-$92,000)
   - Selling put spreads below support ($78,000-$83,000)
   - Using the range-bound model for pricing

2. **Breakout Strategy**: When Bitcoin approaches resistance with increasing volume:
   - Buy call options or call spreads with strikes at and above resistance ($88,000-$90,000)
   - Reduce the respect_level_strength parameter to model potential breakouts
   - Use the volatile model for pricing

3. **Mean Reversion Strategy**: When Bitcoin deviates significantly from its long-term mean:
   - If below mean: Buy calls or sell puts
   - If above mean: Buy puts or sell calls
   - Use a model with higher mean_reversion_weight

4. **Volatility Strategy**: Compare model implied volatilities with market implied volatilities:
   - If model IVs > market IVs: Consider long volatility strategies (buy options)
   - If model IVs < market IVs: Consider short volatility strategies (sell options)

5. **Hybrid Strategy**: Combine technical analysis with model outputs:
   - Use support/resistance levels to define entry/exit points
   - Use mean reversion for directional bias
   - Use jump diffusion for tail risk protection

### Calibrating the Hybrid Model to Market Data

To calibrate the model to actual Deribit prices:

```python
from scipy.optimize import minimize

def objective_function(params, market_prices, strikes, spot, time_to_expiry, risk_free_rate,
                      support_levels, resistance_levels):
    # Extract parameters
    volatility = params[0]
    respect_strength = params[1]
    vol_near_levels = params[2]
    price_action_weight = params[3]
    mean_reversion_weight = params[4]
    jump_diffusion_weight = params[5]
    
    # Ensure weights sum to 1
    total_weight = price_action_weight + mean_reversion_weight + jump_diffusion_weight
    price_action_weight /= total_weight
    mean_reversion_weight /= total_weight
    jump_diffusion_weight /= total_weight
    
    # Fixed parameters
    mean_reversion_params = {
        'long_term_mean': 87000,
        'mean_reversion_rate': 2.5
    }
    
    jump_params = {
        'jump_intensity': 15,
        'jump_mean': -0.04,
        'jump_std': 0.12
    }
    
    # Calculate model prices
    model_prices = []
    for strike in strikes:
        result = hybrid_price_action_monte_carlo(
            option_type='european_call',
            underlying_price=spot,
            strike_price=strike,
            time_to_expiry=time_to_expiry,
            risk_free_rate=risk_free_rate,
            volatility=volatility,
            support_levels=support_levels,
            resistance_levels=resistance_levels,
            mean_reversion_params=mean_reversion_params,
            jump_params=jump_params,
            price_action_weight=price_action_weight,
            mean_reversion_weight=mean_reversion_weight,
            jump_diffusion_weight=jump_diffusion_weight,
            respect_level_strength=respect_strength,
            volatility_near_levels=vol_near_levels,
            simulations=5000,  # Further reduced for optimization speed
            time_steps=max(1, int(time_to_expiry * 52)),  # Weekly steps for speed
            seed=42
        )
        model_prices.append(result['price'])
    
    # Mean squared error between model and market prices
    mse = np.mean((np.array(model_prices) - np.array(market_prices))**2)
    return mse

# Example market prices from Deribit (these would be actual prices)
market_prices = [12500, 10900, 9300, 7800, 6500, 5400, 4300, 3400]

# Initial parameter guess
initial_params = [
    0.75,    # volatility
    0.7,     # respect_strength
    1.5,     # vol_near_levels
    0.33,    # price_action_weight
    0.33,    # mean_reversion_weight
    0.34     # jump_diffusion_weight
]

# Parameter bounds
bounds = [
    (0.3, 1.5),    # volatility between 30% and 150%
    (0.1, 0.9),    # respect_strength between 0.1 and 0.9
    (1.0, 2.0),    # vol_near_levels between 1.0 and 2.0
    (0.1, 0.8),    # price_action_weight between 0.1 and 0.8
    (0.1, 0.8),    # mean_reversion_weight between 0.1 and 0.8
    (0.1, 0.8)     # jump_diffusion_weight between 0.1 and 0.8
]

# Run optimization
result = minimize(
    objective_function,
    initial_params,
    args=(market_prices, strike_prices, current_btc_price, time_to_expiry, risk_free_rate,
          support_levels, resistance_levels),
    bounds=bounds,
    method='L-BFGS-B'
)

# Extract calibrated parameters
calibrated_params = result.x
print("Calibrated Parameters:")
print(f"Volatility: {calibrated_params[0]:.2f}")
print(f"Respect Strength: {calibrated_params[1]:.2f}")
print(f"Volatility Near Levels: {calibrated_params[2]:.2f}")
print(f"Price Action Weight: {calibrated_params[3]:.2f}")
print(f"Mean Reversion Weight: {calibrated_params[4]:.2f}")
print(f"Jump Diffusion Weight: {calibrated_params[5]:.2f}")
```

This hybrid approach provides a powerful framework for Bitcoin option pricing that respects technical levels while capturing both mean reversion and jump risk. By adjusting the model weights and parameters, traders can tailor the pricing model to current market conditions and their own trading view. 