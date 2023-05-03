"""
This is the config file for the PRIME simulation on the ABIDES platform
N.B. Technical agents arrive at 10 times the rate of fundamental agents.
     This is to allow them to keep track of historic data more easily, without
     relying on a centralised price history.
"""

from Kernel import Kernel
from agent.ExchangeAgent import ExchangeAgent
from agent.InitialOrderAgent import InitialOrderAgent
from agent.ZI_market_order import ZI_market_order_agent
from agent.ZI_limit_order import ZI_limit_order_agent
from agent.ZIP import ZeroIntelligencePlus
from agent.MomentumAgent import MomentumAgent
from agent.MeanReversionAgent import MeanReversionAgent
from agent.market_makers.SpreadBasedMarketMakerAgent import SpreadBasedMarketMakerAgent

from util.order import LimitOrder
from util.oracle.ExternalFileOracle import ExternalFileOracle
from util import util

import numpy as np
import pandas as pd
import sys
import datetime as dt
from dateutil.parser import parse


# Some config files require additional command line parameters to easily
# control agent or simulation hyperparameters during coarse parallelization.
import argparse

parser = argparse.ArgumentParser(description='Detailed options for baseline config.')
parser.add_argument('-b', '--book_freq', 
                    default=None,
                    help='Frequency at which to archive order book for visualization'
                    )
parser.add_argument('-c', '--config', 
                    required=True,
                    help='Name of config file to execute'
                    )
parser.add_argument('-l', '--log_dir', 
                    default=None,
                    help='Log directory name (default: unix timestamp at program start)'
                    )
parser.add_argument('-n', '--obs_noise', 
                    type=float, 
                    default=1000000,
                    help='Observation noise variance for zero intelligence agents (sigma^2_n)'
                    )
parser.add_argument('-o', '--log_orders', 
                    action='store_true',
                    help='Log every order-related action by every agent.'
                    )
parser.add_argument('-s', '--seed', 
                    type=int, 
                    default=None,
                    help='numpy.random.seed() for simulation'
                    )
parser.add_argument('-v', '--verbose', 
                    action='store_true',
                    help='Maximum verbosity!'
                    )
parser.add_argument('--config_help', 
                    action='store_true',
                    help='Print argument options for this config file'
                    )
parser.add_argument('-sd', '--start_date',
                    required=True,
                    type=parse,
                    help='historical date being simulated in format YYYYMMDD.'
                    )
parser.add_argument('-ed', '--end_date',
                    required=True,
                    type=parse,
                    help='historical date being simulated in format YYYYMMDD.'
                    )
parser.add_argument('-st', '--start_time',
                    default='09:30:00',
                    type=parse,
                    help='Starting time of simulation.'
                    )
parser.add_argument('-et', '--end_time',
                    default='16:00:00',
                    type=parse,
                    help='Ending time of simulation.'
                    )                 
parser.add_argument('-zi_l', '--ZI_limit_order_agent', 
                    type=int, 
                    default=500,
                    help='number of limit order zero intelligence agents to add to the simulation'
                    )
parser.add_argument('-zi_m', '--ZI_market_order_agent', 
                    type=int, 
                    default=500,
                    help='number of market order zero intelligence agents to add to the simulation'
                    )
parser.add_argument('-zip', '--zero_intelligence_plus', 
                    type=int, 
                    default=0,
                    help='number of zero intelligence plus agents to add to the simulation'
                    )
parser.add_argument('-mmt', '--momentum', 
                    type=int, 
                    default=0,
                    help='number of momentum agents to add to the simulation'
                    )
parser.add_argument('-mr', '--mean_reversion', 
                    type=int, 
                    default=0,
                    help='number of mean reversion agents to add to the simulation'
                    )
parser.add_argument('-mm', '--market_maker', 
                    type=int, 
                    default=0,
                    help='number of market maker agents to add to the simulation'
                    )
args, remaining_args = parser.parse_known_args()

if args.config_help:
  parser.print_help()
  sys.exit()

# Historical date to simulate.  Required even if not relevant.
start_date = pd.to_datetime(args.start_date)
end_date = pd.to_datetime(args.end_date)

# Requested log directory.
log_dir = args.log_dir

# Requested order book snapshot archive frequency.
book_freq = args.book_freq
# 20220302 Chris Cho - input 'None' from the bash script is taken in as a string
# therefore this adjustment is necessary to actually remove book frequency
if book_freq == 'None':
  book_freq = None

# Observation noise variance for zero intelligence agents.
# This is a property of the agents, not the stock.
# Later, it could be a matrix across both.
sigma_n = args.obs_noise

# Random seed specification on the command line.  Default: None (by clock).
# If none, we select one via a specific random method and pass it to seed()
# so we can record it for future use.  (You cannot reasonably obtain the
# automatically generated seed when seed() is called without a parameter.)

# Note that this seed is used to (1) make any random decisions within this
# config file itself and (2) to generate random number seeds for the
# (separate) Random objects given to each agent.  This ensure that when
# the agent population is appended, prior agents will continue to behave
# in the same manner save for influences by the new agents.  (i.e. all prior
# agents still have their own separate PRNG sequence, and it is the same as
# before)

seed = args.seed
if not seed: seed = int(pd.Timestamp.now().timestamp() * 1000000) % (2**31 - 1)
np.random.seed(seed)

# Config parameter that causes util.util.print to suppress most output.
# Also suppresses formatting of limit orders (which is time consuming).
util.silent_mode = not args.verbose
LimitOrder.silent_mode = not args.verbose

# Config parameter that causes every order-related action to be logged by
# every agent.  Activate only when really needed as there is a significant
# time penalty to all that object serialization!
log_orders = None#args.log_orders#


print ("Silent mode: {}".format(util.silent_mode))
print ("Logging orders: {}".format(log_orders))
print ("Book freq: {}".format(book_freq))
print ("Book freq type:", type(book_freq))
print ("ZeroIntelligenceAgent noise: {:0.4f}".format(sigma_n))
print ("Configuration seed: {}\n".format(seed))



# Since the simulator often pulls historical data, we use a real-world
# nanosecond timestamp (pandas.Timestamp) for our discrete time "steps",
# which are considered to be nanoseconds.  For other (or abstract) time
# units, one can either configure the Timestamp interval, or simply
# interpret the nanoseconds as something else.

# What is the earliest available time for an agent to act during the
# simulation?
midnight = start_date
kernelStartTime = midnight

# When should the Kernel shut down?  (This should be after market close.)
# Here we go for 5 PM the same day.
kernelStopTime = end_date + pd.to_timedelta('24:00:00')

# This will configure the kernel with a default computation delay
# (time penalty) for each agent's wakeup and recvMsg.  An agent
# can change this at any time for itself.  (nanoseconds)
defaultComputationDelay = 0        # one second


# IMPORTANT NOTE CONCERNING AGENT IDS: the id passed to each agent must:
#    1. be unique
#    2. equal its index in the agents list
# This is to avoid having to call an extra getAgentListIndexByID()
# in the kernel every single time an agent must be referenced.


# This is a list of symbols the exchange should trade.  It can handle any number.
# It keeps a separate order book for each symbol.  The example data includes
# only JPM.  This config uses generated data, so the symbol doesn't really matter.

# megashock_lambda_a is used to select spacing for megashocks (using an exponential
# distribution equivalent to a centralized Poisson process).  Megashock mean
# and variance control the size (bimodal distribution) of the individual shocks.

# Note: sigma_s is no longer used by the agents or the fundamental (for sparse discrete simulation).


symbols = { 'BTC' : { 'r_bar' : 1e5, 
                      'kappa' : 1.67e-12, 
                      'agent_kappa' : 1.67e-15, 
                      'sigma_s' : 0, 
                      'fund_vol' : 1e-4, 
                      'megashock_lambda_a' : 2.77778e-13, 
                      'megashock_mean' : 1e3, 
                      'megashock_var' : 5e4, 
                      'random_state' : np.random.RandomState(seed=np.random.randint(low=0,high=2**31 - 2, dtype='uint64')) } }



### Configure the Kernel.
kernel = Kernel("Base Kernel", random_state = np.random.RandomState(seed=np.random.randint(low=0,high=2**31 - 1, dtype='uint64')))



### Configure the agents.  When conducting "agent of change" experiments, the
### new agents should be added at the END only.
agent_count = 0
agents = []
agent_types = []


### Configure an exchange agent.

mkt_open = start_date + pd.to_timedelta(args.start_time.strftime('%H:%M:%S'))
mkt_close = end_date + pd.to_timedelta(args.end_time.strftime('%H:%M:%S'))


# Configure an appropriate oracle for all traded stocks.
# All agents requiring the same type of Oracle will use the same oracle instance.
oracle = ExternalFileOracle(symbols)


# Create the exchange.
num_exchanges = 1
agents.extend([ ExchangeAgent(id=j,
                              name="Exchange Agent {}".format(j),
                              type="ExchangeAgent", 
                              mkt_open=mkt_open, 
                              mkt_close=mkt_close, 
                              symbols=[s for s in symbols],
                              book_freq=book_freq,
                              wide_book=False,
                              pipeline_delay=0,
                              computation_delay = 0,
                              stream_history = 10,
                              log_orders=True,#log_orders, 
                              random_state = np.random.RandomState(seed=np.random.randint(low=0,high=2**31 - 1, dtype='uint64')))
                for j in range(agent_count, agent_count + num_exchanges) ])
agent_types.extend(["ExchangeAgent" for j in range(num_exchanges)])
agent_count += num_exchanges


# Create the initial order agent.
num_initial_order = 1
agents.extend([ InitialOrderAgent(id=j,
                                  name="Initial Order Agent {}".format(j),
                                  type="InitialOrderAgent", 
                                  mkt_open_time=mkt_open, 
                                  mkt_close_time=mkt_close, 
                                  symbol='BTC',
                                  log_orders=True,#log_orders, 
                                  random_state = np.random.RandomState(seed=np.random.randint(low=0,high=2**31 - 1, dtype='uint64')))
                for j in range(agent_count, agent_count + num_initial_order) ])
agent_types.extend(["InitialOrderAgent" for j in range(num_initial_order)])
agent_count += num_initial_order


### Configure some zero intelligence agents.

# Cash in this simulator is always in CENTS.
starting_cash = 10000000

# Here are the zero intelligence agents.
symbol = 'BTC'
s = symbols[symbol]

# Tuples are: (# agents, R_min, R_max, eta).


# ZI agents placing limit orders
zi_l = [ (args.ZI_limit_order_agent, 0, 0, 1) ] 

# ZI strategy split.  Note that agent arrival rates are quite small, because our minimum
# time step is a nanosecond, and we want the agents to arrive more on the order of
# minutes.
for i,x in enumerate(zi_l):
  strat_name = "Type {} [{} <= R <= {}, eta={}]".format(i+1, x[1], x[2], x[3])
  agents.extend([ ZI_limit_order_agent(id = j, 
                                        name="Limit ZI Agent {} {}".format(j, strat_name), 
                                        type="ZI_limit_order_agent {}".format(strat_name), 
                                        mkt_open_time=mkt_open, 
                                        mkt_close_time=mkt_close, 
                                        symbol=symbol,
                                        starting_cash=starting_cash, 
                                        sigma_n=sigma_n, #observational noise variance
                                        r_bar=s['r_bar'], #true mean fundamental value
                                        kappa=s['agent_kappa'], #mean reversion parameter
                                        sigma_s=s['fund_vol'], #shock variance
                                        sigma_pv=10, #private value variance
                                        q_max=10, #max unit holding
                                        R_min=x[1], #minimum surplus
                                        R_max=x[2], #maximum surplus
                                        eta=x[3], #strategic threshold
                                        lambda_a=1/6e9,#10, #average arrival rate set to once every 60 sec
                                        random_state = np.random.RandomState(seed=np.random.randint(low=0,high=2**32, dtype='uint64')),
                                        log_orders=log_orders
                                        ) for j in range(agent_count,agent_count+x[0])])
  agent_types.extend([ "ZI_limit_order_agent {}".format(strat_name) for j in range(x[0])])
  agent_count += x[0]


# ZI agents placing market orders
zi_m = [ (args.ZI_market_order_agent, 0, 0, 1) ]

# ZI strategy split.  Note that agent arrival rates are quite small, because our minimum
# time step is a nanosecond, and we want the agents to arrive more on the order of
# minutes.
for i,x in enumerate(zi_m):
  strat_name = "Type {} [{} <= R <= {}, eta={}]".format(i+1, x[1], x[2], x[3])
  agents.extend([ ZI_market_order_agent(id = j, 
                                        name="market ZI Agent {} {}".format(j, strat_name), 
                                        type="ZI_market_order_agent {}".format(strat_name), 
                                        mkt_open_time=mkt_open, 
                                        mkt_close_time=mkt_close, 
                                        symbol=symbol,
                                        starting_cash=starting_cash, 
                                        sigma_n=sigma_n, #observational noise variance
                                        r_bar=s['r_bar'], #true mean fundamental value
                                        kappa=s['agent_kappa'], #mean reversion parameter
                                        sigma_s=s['fund_vol'], #shock variance
                                        sigma_pv=10, #private value variance
                                        q_max=10, #max unit holding
                                        R_min=x[1], #minimum surplus
                                        R_max=x[2], #maximum surplus
                                        eta=x[3], #strategic threshold
                                        lambda_a=1/6e9,#10, #average arrival rate set to once every 60 sec
                                        random_state = np.random.RandomState(seed=np.random.randint(low=0,high=2**32, dtype='uint64')),
                                        log_orders=log_orders
                                        ) for j in range(agent_count,agent_count+x[0])])
  agent_types.extend([ "ZI_market_order_agent {}".format(strat_name) for j in range(x[0])])
  agent_count += x[0]


# 100 ZIP agents
zi_plus = [ (args.zero_intelligence_plus, 0, 0, 1) ]

# ZI strategy split.  Note that agent arrival rates are quite small, because our minimum
# time step is a nanosecond, and we want the agents to arrive more on the order of
# minutes.
for i,x in enumerate(zi_plus):
  strat_name = "Type {} [{} <= R <= {}, eta={}]".format(i+1, x[1], x[2], x[3])

  agents.extend([ ZeroIntelligencePlus(id=j, 
                                       name="ZIP Agent {} {}".format(j, strat_name), 
                                       type="ZeroIntelligencePlus {}".format(strat_name),
                                       mkt_open_time=mkt_open, 
                                       mkt_close_time=mkt_close, 
                                       symbol=symbol,
                                       starting_cash=starting_cash,
                                       sigma_n=sigma_n,
                                       q_max=10,
                                       R_min=x[1],
                                       R_max=x[2],
                                       eta=x[3],
                                       lambda_a=1/6e10, #average arrival rate set to once every 60 sec
                                       random_state = np.random.RandomState(seed=np.random.randint(low=0,high=2**31 - 1, dtype='uint64')), 
                                       log_orders=log_orders) 
  for j in range(agent_count,agent_count+x[0]) ])

  agent_types.extend([ "ZeroIntelligencePlus {}".format(strat_name) for j in range(x[0]) ])
  agent_count += x[0]


# Momentum 20 agents
mmt = [ (args.momentum, 50, 100)]


for i,x in enumerate(mmt):
  strat_name = "Type {} [minimum order size = {}, minimum order size = {}]".format(i+1, x[1], x[2])

  agents.extend([ MomentumAgent(id=j, 
                                name="MomentumAgent {} {}".format(j, strat_name),
                                type="MomentumAgent", 
                                symbol=symbol, 
                                starting_cash=starting_cash,
                                min_size=x[1], 
                                max_size=x[2],
                                lambda_a=1/6e8, #average arrival rate set to once every 60 sec
                                log_orders=log_orders,
                                random_state=np.random.RandomState(seed=np.random.randint(low=0,high=2**31 - 1, dtype='uint64')), 
                                short_duration=5,
                                long_duration=10,
                                margin=0) 
  for j in range(agent_count,agent_count+x[0]) ])

  agent_types.extend([ "MomentumAgent {}".format(strat_name) for j in range(x[0]) ])
  agent_count += x[0]


# Mean Reversion 20 agents
mr = [ (args.mean_reversion, 50, 100)]


for i,x in enumerate(mr):
  strat_name = "Type {} [minimum order size = {}, minimum order size = {}]".format(i+1, x[1], x[2])

  agents.extend([ MeanReversionAgent(id=j, 
                                     name="MeanReversionAgent {} {}".format(j, strat_name), 
                                     type="MeanReversionAgent",
                                     symbol=symbol, 
                                     starting_cash=starting_cash,
                                     min_size=x[1], 
                                     max_size=x[2],
                                     lambda_a=1/6e8, #average arrival rate set to once every 60 sec
                                     log_orders=log_orders,
                                     random_state=np.random.RandomState(seed=np.random.randint(low=0,high=2**31 - 1, dtype='uint64')), 
                                     short_duration=5,
                                     long_duration=10,
                                     margin=0)
  for j in range(agent_count,agent_count+x[0]) ])

  agent_types.extend([ "MeanReversionAgent {}".format(strat_name) for j in range(x[0]) ])
  agent_count += x[0]


# Market maker 1 agents
mm = [(args.market_maker, 100, 10)]

for i,x in enumerate(mm):
  strat_name = "Type {} [order size = {}, window size = {}]".format(i+1, x[1], x[2])

  agents.extend([ SpreadBasedMarketMakerAgent(id=j, 
                                              name="SpreadBasedMarketMakerAgent {} {}".format(j, strat_name), 
                                              type="SpreadBasedMarketMakerAgent", 
                                              symbol=symbol,
                                              starting_cash=starting_cash,
                                              order_size=x[1],
                                              window_size=x[2],
                                              num_ticks=20,
                                              lambda_a=1/6e10, #average arrival rate set to once every 6 sec
                                              subscribe=False,
                                              subscribe_freq=10e9,
                                              subscribe_num_levels=1,
                                              log_orders=log_orders,
                                              random_state = np.random.RandomState(seed=np.random.randint(low=0,high=2**31 - 1, dtype='uint64'))) 
  for j in range(agent_count,agent_count+x[0]) ])

  agent_types.extend([ "SpreadBasedMarketMakerAgent {}".format(strat_name) for j in range(x[0]) ])
  agent_count += x[0]


### Configure a simple message latency matrix for the agents.  Each entry is the minimum
### nanosecond delay on communication [from][to] agent ID.

# Square numpy array with dimensions equal to total agent count.  Most agents are handled
# at init, drawn from a uniform distribution from:
# Times Square (3.9 miles from NYSE, approx. 21 microseconds at the speed of light) to:
# Pike Place Starbucks in Seattle, WA (2402 miles, approx. 13 ms at the speed of light).
# Other agents can be explicitly set afterward (and the mirror half of the matrix is also).

# This configures all agents to a starting latency as described above.
#latency = np.random.uniform(low = 21000, high = 13000000, size=(len(agent_types),len(agent_types)))
latency = np.zeros((len(agent_types),len(agent_types)))

"""
# Overriding the latency for certain agent pairs happens below, as does forcing mirroring
# of the matrix to be symmetric.
for i, t1 in zip(range(latency.shape[0]), agent_types):
  for j, t2 in zip(range(latency.shape[1]), agent_types):
    # Three cases for symmetric array.  Set latency when j > i, copy it when i > j, same agent when i == j.
    if j > i:
      # Presently, strategy agents shouldn't be talking to each other, so we set them to extremely high latency.
      if (t1 == "ZeroIntelligenceAgent" and t2 == "ZeroIntelligenceAgent"):
        latency[i,j] = 1000000000 * 60 * 60 * 24    # Twenty-four hours.
    elif i > j:
      # This "bottom" half of the matrix simply mirrors the top.
      latency[i,j] = latency[j,i]
    else:
      # This is the same agent.  How long does it take to reach localhost?  In our data center, it actually
      # takes about 20 microseconds.
      latency[i,j] = 20000
"""

# Configure a simple latency noise model for the agents.
# Index is ns extra delay, value is probability of this delay being applied.
noise = [ 0.25, 0.25, 0.20, 0.15, 0.10, 0.05 ]



# Start the kernel running.
kernel.runner(agents = agents, startTime = kernelStartTime,
              stopTime = kernelStopTime, agentLatency = latency,
              latencyNoise = noise,
              defaultComputationDelay = defaultComputationDelay,
              oracle = oracle, log_dir = log_dir)

