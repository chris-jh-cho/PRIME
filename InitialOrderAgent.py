from agent.TradingAgent import TradingAgent
from util.util import log_print

from math import sqrt
import numpy as np
import pandas as pd


class InitialOrderAgent(TradingAgent):

    def __init__(self, id, name, type, mkt_open_time, mkt_close_time, symbol='IBM', 
                starting_cash=100000, sigma_n=50, r_bar=100000, kappa=0.05,
                sigma_s=100000, q_max=10000, sigma_pv=5000000, R_min=0, 
                R_max=0, eta=1.0, lambda_a=0.005, log_orders=False, 
                random_state=None):

        # Base class init.
        super().__init__(id, name, type, starting_cash=starting_cash, log_orders=log_orders, random_state=random_state)

        # Store important parameters particular to the ZI agent.
        self.symbol = symbol  # symbol to trade
        self.sigma_n = sigma_n  # observation noise variance
        self.r_bar = r_bar  # true mean fundamental value
        self.kappa = kappa  # mean reversion parameter
        self.sigma_s = sigma_s  # shock variance
        self.q_max = q_max  # max unit holdings
        self.sigma_pv = sigma_pv  # private value variance
        self.R_min = R_min  # min requested surplus
        self.R_max = R_max  # max requested surplus
        self.eta = eta  # strategic threshold
        self.lambda_a = lambda_a  # mean arrival rate of ZI agents
        self.mkt_open_time = mkt_open_time
        self.mkt_close_time = mkt_close_time
        self.order_size = 1
        self.counter = 0

        # track whether the agent should be filling the LOB or cancelling orders
        self.fill = True
        self.cancel = False
        self.first_run = True
        self.second_run = True

        # The agent uses this to track whether it has begun its strategy or is still
        # handling pre-market tasks.
        self.trading = False

        # The agent begins in its "complete" state, not waiting for
        # any special event or condition.
        self.state = 'AWAITING_WAKEUP'

        # The agent maintains two priors: r_t and sigma_t (value and error estimates).
        self.r_t = r_bar
        self.sigma_t = 0

        # The agent must track its previous wake time, so it knows how many time
        # units have passed.
        self.prev_wake_time = None

        # The agent has a private value for each incremental unit. (note that the size is 3 times the 
        # size to account for very large orders being generated by the power law distribution. This
        # is inconsequential for the purpose of the market, but may give weirg numbers when calculating
        # final valuation)
        self.theta = [int(x) for x in sorted(
            np.round(self.random_state.normal(loc=0, scale=sqrt(sigma_pv), size=(q_max * 3))).tolist(),
            reverse=True)]

    def kernelStarting(self, startTime):
        # self.kernel is set in Agent.kernelInitializing()
        # self.exchangeID is set in TradingAgent.kernelStarting()

        super().kernelStarting(startTime)

        self.oracle = self.kernel.oracle

    def kernelStopping(self):
        # Always call parent method to be safe.
        super().kernelStopping()

        # Print end of day valuation.
        H = int(round(self.getHoldings(self.symbol), -2) / 100)
        # May request real fundamental value from oracle as part of final cleanup/stats.
        if self.symbol != 'ETF':
            rT = self.oracle.observePrice(self.symbol, self.currentTime, sigma_n=0, random_state=self.random_state)
        else:
            portfolio_rT, rT = self.oracle.observePortfolioPrice(self.symbol, self.portfolio, self.currentTime,
                                                                 sigma_n=0,
                                                                 random_state=self.random_state)


    def wakeup(self, currentTime):
        # Parent class handles discovery of exchange times and market_open wakeup call.
        super().wakeup(currentTime)

        self.state = 'INACTIVE'

        if not self.mkt_open or not self.mkt_close:
            # TradingAgent handles discovery of exchange times.
            return
        else:
            if not self.trading:
                self.trading = True

                # Time to start trading!
                log_print("{} is ready to start trading now.", self.name)
                print(self.name, "is ready to start trading at time", currentTime)

        # Steady state wakeup behavior starts here.

        # If we've been told the market has closed for the day, we will only request
        # final price information, then stop.
        if self.mkt_closed and (self.symbol in self.daily_close_price):
            # Market is closed and we already got the daily close price.
            return

        # Schedule a wakeup for the next time this agent should arrive at the market
        # (following the conclusion of its current activity cycle).
        # We do this early in case some of our expected message responses don't arrive.

        # Agents should arrive according to a Poisson process.  This is equivalent to
        # each agent independently sampling its next arrival time from an exponential
        # distribution in alternate Beta formation with Beta = 1 / lambda, where lambda
        # is the mean arrival rate of the Poisson process.

        # wake up once after 5 minutes
        wake_time = self.getWakeFrequency()
        self.setWakeup(currentTime + wake_time)

        # If the market has closed and we haven't obtained the daily close price yet,
        # do that before we cease activity for the day.  Don't do any other behavior
        # after market close.
        if self.mkt_closed and (not self.symbol in self.daily_close_price):
            self.getCurrentSpread(self.symbol)
            self.state = 'AWAITING_SPREAD'
            return

        if self.cancel==True and self.fill==True:
            self.state = 'INACTIVE'
            return 


        if self.cancel == True:
            self.cancelOrders()
            self.state = 'AWAITING_WAKEUP'
            print("Orders are now cancelled at time", currentTime)
            return

        # The ZI agent doesn't try to maintain a zero position, so there is no need to exit positions
        # as some "active trading" agents might.  It might exit a position based on its order logic,
        # but this will be as a natural consequence of its beliefs.

        # In order to use the "strategic threshold" parameter (eta), the ZI agent needs the current
        # spread (inside bid/ask quote).  It would not otherwise need any trade/quote information.

        # If the calling agent is a subclass, don't initiate the strategy section of wakeup(), as it
        # may want to do something different.

        if self.fill == True:

            self.cancel = True

            if type(self) == InitialOrderAgent:
                self.getCurrentSpread(self.symbol)
                self.state = 'AWAITING_SPREAD'
            else:
                self.state = 'ACTIVE'

    def updateEstimates(self):
        # Called by a background agent that wishes to obtain a new fundamental observation,
        # update its internal estimation parameters, and compute a new total valuation for the
        # action it is considering.

        # The agent obtains a new noisy observation of the current fundamental value
        # and uses this to update its internal estimates in a Bayesian manner.
        obs_t = np.round(self.oracle.observeDiscretePrice(self.symbol, self.currentTime, width=0,
                                                          random_state=self.random_state))

        return obs_t

    def placeOrder(self):
        # Called when it is time for the agent to determine a limit price and place an order.
        # updateEstimates() returns the agent's current total valuation for the share it
        # is considering to trade and whether it will buy or sell that share.
        p = self.updateEstimates()

        # Place the order.
        for i in range(1, 100):
            self.order_size = np.min([i + 1, 5])
            self.placeLimitOrder(self.symbol, self.order_size, True, int(round(p) - i))
            self.placeLimitOrder(self.symbol, self.order_size, False, int(round(p) + i))


    def receiveMessage(self, currentTime, msg):
        # Parent class schedules market open wakeup call once market open/close times are known.
        super().receiveMessage(currentTime, msg)

        # We have been awakened by something other than our scheduled wakeup.
        # If our internal state indicates we were waiting for a particular event,
        # check if we can transition to a new state.

        if self.state == 'AWAITING_SPREAD':
            # We were waiting to receive the current spread/book.  Since we don't currently
            # track timestamps on retained information, we rely on actually seeing a
            # QUERY_SPREAD response message.

            if msg.body['msg'] == 'QUERY_SPREAD':
                # This is what we were waiting for.

                # But if the market is now closed, don't advance to placing orders.
                if self.mkt_closed: return

                # We now have the information needed to place a limit order with the eta
                # strategic threshold parameter.
                self.placeOrder()
                self.state = 'AWAITING_WAKEUP'

    # Cancel all open orders.
    def cancelOrders(self):

        self.cancelOrder()

        return True

    # wakeup once after 5 minutes
    def getWakeFrequency(self):
        if self.first_run == True:
            self.first_run = False
            return pd.Timedelta('{}ns'.format(int(10)))

        elif self.second_run == True:
            self.second_run = False
            return pd.Timedelta('{}ns'.format(int(300000000000)))

        else:
            return pd.Timedelta('{}ns'.format(int(30000000000000)))