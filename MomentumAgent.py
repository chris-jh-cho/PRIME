from agent.TradingAgent import TradingAgent
import pandas as pd
import numpy as np


class MomentumAgent(TradingAgent):

    """
    Simple Trading Agent that compares the "n" past mid-price observations with
    the "m" past observations and places a buy limit order if the
    "n" mid-price average >= "m" mid-price average plus margin, or a sell
    limit order if the "n" mid-price average <= "m" mid-price average minus
    margin
    """

    def __init__(self, id, name, type, symbol='IBM', starting_cash=100000,
                 min_size=50, max_size=100, lambda_a=0.05,
                 log_orders=False, random_state=None, short_duration=20,
                 long_duration=40, margin=0):

        super().__init__(id, name, type, starting_cash=starting_cash,
                         log_orders=log_orders, random_state=random_state)

        # received information
        self.symbol = symbol
        self.min_size = min_size  # Minimum order size
        self.max_size = max_size  # Maximum order size
        self.short_duration = short_duration
        self.long_duration = long_duration
        self.margin = margin
        self.lambda_a = lambda_a
        self.log_orders = log_orders

        # initialise setup
        self.order_size = self.random_state.randint(self.min_size, self.max_size)
        self.mid_list = []
        self.ma_short = None
        self.ma_long = None
        self.state = "AWAITING_WAKEUP"
        self.bidvol = 0
        self.askvol = 0

    def kernelStarting(self, startTime):
        super().kernelStarting(startTime)

    def kernelStopping(self):
        # Always call parent method to be safe.
        super().kernelStopping()

    def wakeup(self, currentTime):

        """ Agent wakeup is determined by self.wake_up_freq """

        can_trade = super().wakeup(currentTime)

        if not can_trade:
            return

        self.getCurrentSpread(self.symbol)
        self.state = 'AWAITING_SPREAD'

    def receiveMessage(self, currentTime, msg):

        """
        Mean reversion agent actions are determined after obtaining the best
        bid and ask in the LOB
        """

        super().receiveMessage(currentTime, msg)
        if (self.state == 'AWAITING_SPREAD' and
                msg.body['msg'] == 'QUERY_SPREAD'):

            # query bid/ask price
            bid, bidvolume, ask, askvolume = self.getKnownBidAsk(self.symbol)
            self.bidvol = bidvolume
            self.askvol = askvolume

            if bid and ask:
                
                mid = (bid + ask) / 2
                self.mid_list.append(mid)

                # determine mid-price

                if len(self.mid_list) > self.long_duration:

                    self.mid_list.pop(0)


                    # Determine Moving Average "n" after n datapoints
                    self.ma_short = MomentumAgent.ma(self.mid_list, n=self.short_duration)[-1].round(0)
                    self.ma_long = MomentumAgent.ma(self.mid_list, n=self.long_duration)[-1].round(0)

                    # Only start comparing once both MAs become available
                    if self.ma_short and self.ma_long:
                        
                        # 20210513 Chris Cho: Query new order size
                        buyorder = np.round(self.askvol**0.35)
                        sellorder = np.round(self.bidvol**0.35)
                        # 20200928 Chris Cho: Added the margin function
                        if (self.ma_short < self.ma_long - self.margin):

                            self.placeMarketOrder(self.symbol, quantity=sellorder,
                                                is_buy_order=False)


                        elif (self.ma_short > self.ma_long + self.margin):

                            self.placeMarketOrder(self.symbol, quantity=buyorder,
                                                is_buy_order=True)
                        

            # set wakeup time
            self.setWakeup(currentTime + self.getWakeFrequency())
            self.state = 'AWAITING_WAKEUP'

    def getWakeFrequency(self):

        delta_time = self.random_state.exponential(scale=1.0 / self.lambda_a)
        return pd.Timedelta('{}ns'.format(int(round(delta_time))))

    # 20201026 Chris Cho: function to query order size
    def getOrderSize(self):

        # round up the order size to prevent orders of size 0
        order_size = np.ceil(70/np.random.power(3.5))

        # select random number
        i = self.random_state.rand()

        # with a chance, submit order as it is
        if i < 0.8:
            self.order_size = order_size

        # otherwise, round to nearest 10 orders
        else:

            # quick hack to prevent orders rounding to 0
            if order_size < 5:
                order_size += 5

            # round to nearest 10
            self.order_size = np.round(order_size, -1)

        return None

    @staticmethod
    def ma(a, n=20):
        ret = np.cumsum(a, dtype=float)
        ret[n:] = ret[n:] - ret[:-n]
        return ret[n - 1:] / n
