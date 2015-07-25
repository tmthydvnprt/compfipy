# compfipy todo / roadmap

## Basic Code Modules

### Calculator
- [x] General finance calculations
- [ ]

### Models
- [x] create stochastic models
- [ ]

### Asset
- [x] OCLHV time series data storage
- [x] OCLHV time series calculations
- [x] OCLHV time series technical equations
- [x] Performance measures
- [x] Risk measures
- [x] Market Comparisons
- [x] Summary Asset
- [ ]

### Portfolio
- [ ] Aggregate specific Assets in Table
- [ ] Define 'holding' of each Asset
- [ ] Enter and Exit holding for one Asset
- [ ] Rebalance whole Portfolio
- [ ] Return weights
- [ ] Total Unrealized Performance measures (based on specific time)
- [ ] Total Unrealized Risk measures (based on specific time)
- [ ] Total Unrealized Market Comparisons (based on specific time)
- [ ] Total Realized Performance measures (based on holdings)
- [ ] Total Realized Risk measures (based on holdings)
- [ ] Total Realized Market Comparisons (based on holdings)
- [ ] Summarize Portfolio
- [ ]

### Market
- [ ] Aggregate of all Assets
- [ ] Aggregate by sector?
- [ ] Aggregate by cap?
- [ ]

### Events
- [ ] Define event as some parameter (price, SAM, MACD, etc.) passing a threshold
- [ ] Search an Asset for the event
- [ ] Return booleans of when the event happens
- [ ]

### Strategy
- [ ] Defines how a Events will affect a Portfolio
- [ ] Define Enter and Exit Positions based on one or more events
- [ ] Update Holdings of Portfolio based on the Strategy
- [ ]

## Overall Function and Flow
- [ ] create trend trading algorithm
- [ ] test trend algos on stochastic models, just for process completeness, not correctness at the moment

## Parameter Optimization
- [ ] get asset history
- [ ] design genetic algorithm to solve for stochastic model parameters to model asset history
- [ ] use multi-temporal windows of asset history to design parameters
- [ ] see how multi-temporal parameters change
- [ ] find general best fit parameters for stochastic models
- [ ] use genetic algorithm to train trending algos parameters, using some weighted combination of profit, volatility, risk, etc. as fitness function
- [ ] once trained on blind stochastic model run on historical asset data
- [ ] do this once for a simple case
- [ ] repeat for maximizing return
- [ ] repeat for minimizing risk
- [ ] define a way to combine the two
- [ ]
