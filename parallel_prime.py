"""
Parallel wrapper to run the PRIME model using the ABIDES platform
Note that the technical agents arrive at 10 times the rate of fundamental
agents, hence 1 technical agent = 10 fundamental agents in effect
"""

import argparse
import os
from multiprocessing import Pool
from numpy.core.numeric import normalize_axis_tuple
import psutil
import datetime as dt
import numpy as np
import pandas as pd
from dateutil.parser import parse
import pyDOE


def run_in_parallel(num_simulations, num_parallel, config, log_dir, verbose,
                    log_orders, book_freq, start_date, end_date, start_time,
                    end_time, noise):

    global_seeds = np.random.randint(0, 2 ** 31, num_simulations)
    print(f'Global Seeds: {global_seeds}')
    

    # initialise processes
    processes = []

    # divide the simulated timeframe into equal duration, with an output array
    # of dates and time
    start   = pd.to_datetime(str(start_date) + ' ' + str(start_time))
    end     = pd.to_datetime(str(end_date) + ' ' + str(end_time))
    dur     = (end - start)/num_simulations
    
    # assign number of atents according to predetermined configeration
    zi_l_count      = 1000
    zi_m_count      = 30
    zip_count       = 0
    mmt_count       = 1
    mr_count        = 1
    mm_count        = 0

    # iterate over number of simulations
    for i in range(num_simulations):

        # set seed according to global seed
        seed = global_seeds[i]

        # calculate start and end datetime for each simulation
        sim_start   = start + dur*i
        sim_end     = start + dur*(i + 1)

        # convert datetime into the ABIDES string format (YYYYMMDD & hh:mm:ss)
        # N.B. seconds are always rounded down
        start_date  = str(sim_start.date()).replace('-', '')
        start_time  = str(sim_start.time())[:8]
        end_date    = str(sim_end.date()).replace('-', '')
        end_time    = str(sim_end.time())[:8]

        # print input and output to track the times covered by each simulation
        print(f"\nSimulation {i} of {num_simulations - 1} \n\
        - - - - - \n\
        Raw Pandas datetime \n\
         Start datetime: {sim_start} \n\
         End time: {sim_end} \n\
        - - - - - \n\
        Cleaned datetime for ABIDES \n\
         Start date: {start_date} \n\
         Start time: {start_time} \n\
         End date: {end_date} \n\
         End time: {end_time}")

        # append the run script in bash for each simulation
        processes.append(f'python -u abides.py -c {config} \
        -l {log_dir}_sim_{i} -o {log_orders} \
        {"-v" if verbose else ""} -s {seed} -b {book_freq} \
        -sd {start_date} -ed {end_date} -st {start_time} \
        -et {end_time} -zi_l {zi_l_count} -zi_m {zi_m_count} \
        -zip {zip_count} -mmt {mmt_count} -mr {mr_count} \
        -mm {mm_count} -n {noise}')

    print(processes)  
    pool = Pool(processes=num_parallel)
    pool.map(run_process, processes)


def run_process(process):
    os.system(process)


if __name__ == "__main__":
    sim_start_time = dt.datetime.now()

    parser = argparse.ArgumentParser(description='Main config to run multiple ABIDES simulations in parallel')
    parser.add_argument('-b', '--book_freq', 
                        default=None,
                        help='Frequency at which to archive order book for visualization'
                        )
    parser.add_argument('-ns', '--num_simulations', 
                        type=int,
                        default=1,
                        help='Total number of simulations to run')
    parser.add_argument('-np', '--num_parallel', 
                        type=int,
                        default=None,
                        help='Number of simulations to run in parallel')
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
                        help='historical date being simulated in format YYYYMMDD.'
                        )
    parser.add_argument('-ed', '--end_date',
                        required=True,
                        help='historical date being simulated in format YYYYMMDD.'
                        )
    parser.add_argument('-st', '--start_time',
                        default='09:30:00',
                        help='Starting time of simulation.'
                        )
    parser.add_argument('-et', '--end_time',
                        default='16:00:00',
                        help='Ending time of simulation.'
                        )                 
    parser.add_argument('-zi_l', '--zero_intelligence_limit_order', 
                        type=int, 
                        default=600,
                        help='number of zero intelligence agents to add to the simulation'
                        )
    parser.add_argument('-zi_m', '--zero_intelligence_market_order', 
                        type=int, 
                        default=200,
                        help='number of zero intelligence agents to add to the simulation'
                        )
    parser.add_argument('-zip', '--zero_intelligence_plus', 
                        type=int, 
                        default=50,
                        help='number of zero intelligence plus agents to add to the simulation'
                        )
    parser.add_argument('-mmt', '--momentum', 
                        type=int, 
                        default=24,
                        help='number of momentum agents to add to the simulation'
                        )
    parser.add_argument('-mr', '--mean_reversion', 
                        type=int, 
                        default=25,
                        help='number of mean reversion agents to add to the simulation'
                        )
    parser.add_argument('-mm', '--market_maker', 
                        type=int, 
                        default=1,
                        help='number of market maker agents to add to the simulation'
                        )

    args, remaining_args = parser.parse_known_args()
    
    seed            = args.seed
    num_simulations = args.num_simulations
    num_parallel    = args.num_parallel if args.num_parallel else psutil.cpu_count() # count of the CPUs on the machine
    config          = args.config
    log_dir         = args.log_dir
    verbose         = args.verbose
    book_freq       = args.book_freq
    start_date      = args.start_date
    end_date        = args.end_date
    start_time      = args.start_time
    end_time        = args.end_time
    noise           = args.obs_noise
    log_orders      = args.log_orders


    print(f'Total number of simulation: {num_simulations}')
    print(f'Number of simulations to run in parallel: {num_parallel}')
    print(f'Configuration: {config}')

    np.random.seed(seed)

    run_in_parallel(num_simulations = num_simulations,
                    num_parallel    = num_parallel,
                    config          = config,
                    log_dir         = log_dir,
                    verbose         = verbose,
                    book_freq       = book_freq,
                    start_date      = start_date,
                    end_date        = end_date,
                    start_time      = start_time,
                    end_time        = end_time,
                    noise           = noise,
                    log_orders      = log_orders)

    sim_end_time = dt.datetime.now()
    print(f'Total time taken to run in parallel: {sim_end_time - sim_start_time}')
