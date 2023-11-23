############################
# Binpack_Transaction_Manager
#
# Provides a way to create a list of transactions using a portfolio management scheme.
# Binpack refers to the optimization algorithm used to determine the set of buys and sells.
############################

############################
# Import
############################
import package_utils as vu
import pandas as pd
import math


############################
# Global Variables
############################
EXCESS_IDENTIFIER = "EXCESS_"


############################
# Global Functions
############################
#-------------------------------------------------------------------
def package_get_optimized_transaction_list(package_db, strategy, target_port_size, market_frame, fee_manager,
                                         tax_manager=None, use_tax_uncertainty=False, curr_holdings_frame=None,
                                         verbose=False, max_percent_capital_in_sector=100,
                                         excess_threshold=None, excess_size_penalty=0, uncertainty_factor=0,
                                         min_tx_magnitude_frac=0.01, protector_keep_cash=False,
                                         protector_etf="SHY-US", protector_engaged=False,
                                         protector_invested_frac=(1.0/3.0), min_cash_to_keep=0,
                                         correlation_threshold=1.0, correlation_dataframe=None,
                                         target_date=None):

    # Given a set of holdings, a group of stocks from the market, and a strategy, figures out the best
    # list of transactions to maximize the projected forward return
    # Returns a tuple of (the transaction_list, the target_port_frame, and final cash value).

    # ==================================================================================================
    # PARAMETERS - A list of the parameters that the binpacker takes in and what they all mean
    # ==================================================================================================
    #
    # "PACKAGE_DB": An object that is used to retrieve data from the database.
    #
    # "STRATEGY": A string that tells the binpacker which strategy to run (currently, the binpacker
    # only supports the advanced strategy).
    #
    # "TARGET_PORT_SIZE": An integer that is the target number of stocks that the portfolio should
    # have in it - this target is not enforced so the final portfolio can have any number of stocks
    # in it. When protector is used, this number is divided by 3 - only one-third of the
    # capital is invested and the rest is in cash.
    #
    # "MARKET_FRAME": A Pandas dataframe that contains data on all stocks that passed the stock
    # screen. This data includes the stock's sector ID, score (represents how good the stock is to
    # buy - calculated differently for each different strategy), market cap, and its sector rank
    # (its rank in the sector when sorted by score).
    #
    # "FEE_MANAGER": A class tht calculates fees for transactions.
    #
    # "CURR_HOLDINGS_FRAME": A Pandas dataframe that contains information on the current holdings
    # of the portfolio. This dataframe contains an entry for each stock that is currently being
    # held as well as one entry for cash (ticker is "USDCAD=X"). This dataframe will have all
    # dollar values in US currency.
    #
    # "VERBOSE": A boolean that determines whether or not the binpacker prints statements
    # describing its decisions. These print statements have information that is useful when
    # debugging.
    #
    # "MAX_PERCENT_CAPITAL_IN_SECTOR": A float that limits the maximum amount of capital that can be put
    # into any one sector. This limit is never violated by the binpacker. If the current portfolio
    # has too much capital in one sector, the binpacker will sell stocks from that sector until
    # the capital in the sector is below the limit.
    #
    # "EXCESS_THRESHOLD": A float that is used to calculate the threshold after which a stock is
    # deemed an 'excess' holding - this threshold equals: excess threshold * ideal target per stock.
    # If this threshold is very low, the portfolio will always be rebalanced and a position in a
    # stock will never grow too large. If this value is set higher, the binpacker will "ride its
    # winners" (not immediately rebalance a stock if it has done very well).
    #
    # "EXCESS_SIZE_PENALTY": A float used to reduce the expected ROI for any stock that is deemed
    # an "excess" holding. This is done so that extremely large positions are sold which leads to
    # diversification and a reduction of risk.
    #
    # "UNCERTAINTY_FACTOR": A float that represents the inherent uncertainty present when making a
    # transaction. If one stock has an expected ROI that is only slightly higher than the expected
    # ROI of a stock you currently hold, it may not be worth it to exchange the two. The binpacker
    # will reduce the expected ROI of all buy transactions by the uncertainty factor to prevent the
    # binpacker from making multiple transaction for only a very slight gain. The uncertainty factor
    # is generally between 0 and 1, but it can theoretically be any value. If it is set to negative
    # infinity, all buy and sell transactions will be fulfilled. If on the other hand it is set to
    # positive infinity, no transactions except the forced ones will be fulfilled.
    #
    # "MIN_TX_MAGNITUDE_FRAC": A float used to calculate min_tx_magnitude which equal this value
    # times capital. The min_tx_magnitude limits the minimum size of a buy or sell transaction and
    # also limits the minimum size of any holding after all transactions have been made. This means
    # that the min_tx_magnitude scales with capital. This minimum is set for two reasons: 1) To
    # minimize trading commissions paid and 2) To prevent the binpacker from leaving extremely small
    # positions of various stocks (over time, the portfolio could accumulate many such small positions
    # if this restriction isn't added).
    #
    # "PROTECTOR_KEEP_CASH": A boolean value that is used to determine whether or not the
    # protector "vessel" should be the protector_etf or cash. This introduces some complexity which
    # is covered in a comment inside the liquidate_protector_etf function.
    #
    # "PROTECTOR_ETF": A string that is the ticker of the ETF to buy when protector is used.
    # In this case, two-thirds of all capital is used to buy this ETF. When protector_keep_cash is true,
    # the protector_etf is still used internally in the binpacker, but it is all converted into cash at
    # the very end (in the liquidate_protector_etf function).
    #
    # "PROTECTOR_ENGAGED": A boolean that determines whether or not protector should be used.
    # This value is true iff the current SPY price is below its 200 day moving average. This
    # calculation is done in the portfolio_simulator.
    #
    # "MIN_CASH_TO_KEEP": A float that forces the binpacker to end up with a certain amount of cash
    # after all transactions are complete. This restriction is used for modelling taxes. Because
    # taxes must be paid immediately, the binpacker needs to retain a certain amount of cash to be
    # able to pay these taxes.
    #
    # "CORRELATION_THRESHOLD": A float that is used to determine whether or not a stock is similar
    # to another stock. The binpacker picks a number of stocks equal to the target port size which
    # becomes the target stock list. If the past 200 day price correlation between a stock in this target
    # list and a stock you already hold is realtively high (higher than the correlation threshold),
    # then which of the two you own doesn't really matter. In this case, the stock in the target
    # list is switched out for the one you already own (to limit uneccesary transactions). For
    # example, if Coke is in the target list but you already own Pepsi, then Pepsi would be put into
    # the target list instead (Coke and Pepsi's prices move very closely, so which one you own
    # doesn't matter too much).
    #
    # "CORRELATION_DATAFRAME": A Pandas dataframe that has random correlation values between held
    # stocks and desired stocks. This parameter is only used by the random tester - for the regular
    # flow, the correlation dataframe is created within this function.
    #
    # "TARGET_DATE": An integer that has the date at which to run the binpacker in YYYYMMDD format.
    #
    # ==================================================================================================

    # ==================================================================================================
    # LEGEND - Terms used throughout the code
    # ==================================================================================================
    #
    # "TARGET/IDEAL PORTFOLIO": the portfolio as defined by the strategy while not limiting the number
    # of transactions in any way (blindly following the strategy without trying to reduce fees paid).
    # Note that normally when the code talks about a target_frame it is talking about the market_frame
    # but with the target investment value set (usually to 0, and to an actual value for stocks we want
    # to hold in the ideal portfolio).
    #
    # "TARGET TRANSACTIONS": the set of transactions required to go from the current portfolio to the
    # target portfolio.
    #
    # "FINAL TRANSACTIONS": the set of transactions that the optimization algorithm has deemed to be
    # optimal and will lead to the highest monthly ROI, a relatively high margin of safety, and the
    # least amount of fees paid.
    #
    # ==================================================================================================

    # ==================================================================================================
    # High level flow of this module
    # ==================================================================================================
    # STEP 1 - Bookkeeping - figure out the total capital, cash, and other limits
    #          such as minimum transaction size and limit of capital in any one sector.
    # STEP 2 - Target Setter - set target (desired) holdings by considering how much the desired transactions
    #          will lose to trading friction.
    # STEP 3 - Binpack Optimizer - optimize holdings based on ROI estimates to come up with a list of transactions.
    #     3a - Out of Market Seller - sell anything that is not in our universe anymore.
    #     3b - Pre-Opt Small Seller - inside the binpack optimizer, it will sell any small positions to
    #          provide an opportunity to put those funds into something useful. For example, this helps avoid
    #          big losers from hanging around as small stub positions.
    #     3c - Protector and Tax Seller - raises cash to put into a safe cash-like ETF (or cash itself). Also
    #          makes sure enough cash is kept for any tax payments demanded by the caller.
    #     3d - Optimizer - this is the actual binpack optimizer.
    # STEP 4 - Post-Opt Small Seller - sell any holding that is smaller than the minimum transaction size.
    # STEP 5 - Optionally Cash Protector - if yields are low, we convert the protector ETF into cash
    #          because it's not worth paying trading costs for a no-yield ETF.
    # STEP 6 - Verification - checks after everything is done.


    #######################################
    # STEP 1 - Bookkeeping
    #######################################
    if protector_engaged and protector_invested_frac==1.0:
        protector_engaged = False
        print ("\n\n--> WARNING: Disabling protectors because the invested_frac is 1.0 anyway")

    assert excess_threshold > 1.0  # 1.0 is the initial investment target, so excess must be higher than that
    local_market_frame = market_frame.copy()  # We modify the market frame so make a copy to be safe

    # Calculate capital in US dollars
    capital = curr_holdings_frame['Investment Total USD'].sum()
    min_tx_magnitude = capital * min_tx_magnitude_frac

    # Rename index column in market frame to be ticker instead of security id
    curr_indices = local_market_frame.index
    for index in curr_indices:
        local_market_frame = local_market_frame.rename(index={index: vu.get_ticker_for_security_id(package_db, index)})

    if verbose:
        num_curr_investments = curr_holdings_frame[curr_holdings_frame.index.astype(str).str[-3:-2] == "-"].shape[0]
        print("Portfolio has %d advanced stocks" % num_curr_investments)

    # Controls how much money can be put into any one sector (used to force some level of diversification)
    max_capital_per_sector = capital*max_percent_capital_in_sector/100.0

    # Make in_market_holdings_frame, which is a copy of the local_market_frame, but with an extra column with
    # our current holdings. This copy also excludes anything not in the market, which is good because
    # we special-case forced selling of things not in the market_frame later.
    in_market_holdings_frame = local_market_frame.copy()
    in_market_holdings_frame['Investment'] = curr_holdings_frame['Investment Total USD']
    in_market_holdings_frame = in_market_holdings_frame.fillna(0)

    # If we are using protectors, decrease the number of stocks desired by two-thirds to reduce risk
    post_protector_stocks = round(target_port_size * (protector_invested_frac)) if protector_engaged else target_port_size

    cash = curr_holdings_frame.at['USDCAD=X', 'Investment Total USD'] + \
        (curr_holdings_frame.at['1', 'Investment Total USD'] if '1' in curr_holdings_frame.index.tolist() else 0)

    if verbose:
        print("Found cash of %.2f" % cash)

    #######################################
    # STEP 2 - Target Setter
    #######################################
    # This function call generates a "target portfolio" which is the ideal portfolio
    # if you made unlimited transactions. It includes all market stocks, but has 0 target for
    # all but the ones we want to hold.
    target_port_frame, target_investment_per_stock, protector_investment = gen_ideal_targets(
        package_db, cash, curr_holdings_frame,
        strategy, local_market_frame, capital, target_port_size, max_capital_per_sector, fee_manager,
        in_market_holdings_frame, protector_engaged, protector_etf, post_protector_stocks,
        min_cash_to_keep, correlation_threshold, correlation_dataframe, verbose)

    if min_tx_magnitude > ((capital/target_port_size) / 4.0):
        # This is not an ERROR because the testing code uses higher min_tx_magnitudes to make sure it
        # works when it is high. The warning should stick out in the output to catch experiments
        # where larger portfolios are being used without updating the min_tx_magnitude.
        print ("\n\n--> WARNING: To be safe, you " \
             "should have a min_tx_magnitude high enough to trade for your portfolio size - " \
             "check your numbers and adjust this assert if needed (port size %d, min_tx %g)" % (target_port_size, min_tx_magnitude))
    # min_tx_magnitude has to be at least 100 because really low values may cause an issue in
    # multiple parts of the code. For example, if a sell transaction of magnitude $5 goes
    # through, then it will cause you to lose money even though you expected it to generate cash.
    assert min_tx_magnitude >= 100, "\n\n--> ERROR: The binpack transaction manager doesn't " \
                                    "yet support a min_tx_magnitude lower than 100\n\n"

    #######################################
    # STEP 3 - Binpack Optimizer
    #######################################
    # Now that we have a target to shoot for, call the optimizer.
    final_txs, cash_kept = calculate_optimized_txs(package_db, local_market_frame, capital, cash,
            curr_holdings_frame, in_market_holdings_frame, target_port_frame, excess_threshold, excess_size_penalty,
            min_tx_magnitude, uncertainty_factor, max_capital_per_sector, target_investment_per_stock, fee_manager,
            tax_manager, use_tax_uncertainty, protector_engaged, protector_investment, protector_etf,
            min_cash_to_keep, target_date, verbose)

    #######################################
    # STEP 4 - Post-Opt Small Seller
    #######################################
    # After the optimized transactions list is created, ensure that the min_tx_magnitude is not being violated.
    # The min_tx_magnitude not only limits the size of transactions, but also limits the minimum size of a
    # position after all transaction have been made (for example, if min_tx_magnitude equals 50000, then there
    # cannot be a position with a size less than 50000). If a position that violates this limit exists, the
    # following function will immediately sell it.
    final_txs, cash_kept = resolve_post_optimization_small_position_forced_sells(
        min_tx_magnitude, protector_etf, final_txs, fee_manager, cash_kept, verbose)

    #######################################
    # STEP 5 - Optionally Cash Protector
    #######################################
    # The following block sells the protector_etf if cash is the chosen
    # protector vessel - more detail in the liquidate_protector_etf function
    if protector_keep_cash and protector_engaged:
        final_txs, cash_kept = liquidate_protector_etf(final_txs, cash_kept, protector_etf,
                                                       fee_manager)

    #######################################
    # STEP 6 - Verification
    #######################################
    # This function checks that after all transactions are resolved, there is no position with a
    # size smaller than the min_tx_magnitude. See comment above for more details.
    check_position_sizes(final_txs, min_tx_magnitude)
    # This function checks that after all transactions are resolved, the max_capital_per_sector has not been violated
    check_sector_limits(package_db, final_txs, max_capital_per_sector, min_tx_magnitude)
    # The following function checks to make sure that all non-forced
    # transactions have a magnitude larger than the min_tx_magnitude
    check_tx_sizes(final_txs, min_tx_magnitude)

    assert cash_kept >= min_cash_to_keep, "\n\n--> ERROR: The amount of cash being kept ($%d) is less than " \
                                          "the minimum amount of cash that has to be kept ($%d)"

    if verbose:
        # Make sure that the whole frame is printed out and no part of it is cut off
        print("\nFINAL RETURNED TX DATAFRAME:")
        vu.print_entire_array_or_frame(final_txs)
        print ("(Also dumped to CSV file)")
        final_txs.to_csv("binpack_debug_dump_final_returned_frame.csv")

    return final_txs, target_port_frame, cash_kept
#-------------------------------------------------------------------


############################
# Local Functions
############################
#-------------------------------------------------------------------
def check_position_sizes(final_txs, min_tx_magnitude):
    # This function ensures that all positions have a size greater than or equal to min_tx_magnitude
    # after transactions are resolved

    for ticker in final_txs.index.tolist():
        if final_txs.at[ticker, 'Sell_Amt'] > 0:
            final_holdings_value = final_txs.at[ticker, 'Curr_Holdings'] - final_txs.at[ticker, 'Sell_Amt']
        else:
            final_holdings_value = final_txs.at[ticker, 'Hold_Amt'] + final_txs.at[ticker, 'Buy_Amt'] - \
                                   final_txs.at[ticker, 'Tx_Fees_Paid']

        # Allow for some rounding error for this assertion
        assert final_holdings_value >= -1, "\n\n--> ERROR: The final holdings in %s of -$%d is negative" \
                                          % (ticker, abs(final_holdings_value))
        assert not 0 < round(final_holdings_value, 2) < min_tx_magnitude, \
            "\n\n--> ERROR: The final holdings in %s of $%d is less than the minimum of $%d\n\n" \
            % (ticker, final_holdings_value, min_tx_magnitude)
#-------------------------------------------------------------------
def check_sector_limits(package_db, final_txs, max_capital_per_sector, min_tx_magnitude):
    # This function ensures that the total amount of capital invested in a given sector is
    # less than max_capital_per_sector

    sector_list = package_db.get_list_for_query("SELECT Sector_Id FROM Sectors")
    # Create a dictionary that tracks how much capital has been invested into each sector
    capital_per_sector_dict = {i[0]: 0 for i in sector_list}

    for ticker, row in final_txs.iterrows():
        sec_id = vu.get_security_id_for_ticker(package_db, ticker)
        sector_id = vu.get_sector_id_for_sec_id(package_db, sec_id, error_on_fail=False)

        # If a sector ID is not found for a stock, then just skip it.
        # This may happen for things like the protector_etf.
        if sector_id is None:
            continue
        else:
            if row['Buy_Amt'] > 0:
                capital_per_sector_dict[sector_id] += row['Hold_Amt'] + row['Buy_Amt'] - row['Tx_Fees_Paid']
            else:
                capital_per_sector_dict[sector_id] += row['Hold_Amt']

            # Sometimes, the total capital invested in a sector violates the limit because certain sell transactions
            # are not resolved because they are too small (size of transaction is less than min_tx_magnitude). In
            # this case, min_tx_magnitude takes priority, so those transactions are left unresolved. Because of this,
            # this function needs to check if the sector limit was violated after assuming that those small sell
            # transactions had gone through.
            if (0 < (row['Curr_Holdings'] - row['Target_Holdings']) < min_tx_magnitude) and row['Sell_Amt'] == 0:
                capital_per_sector_dict[sector_id] -= row['Curr_Holdings'] - row['Target_Holdings']

    for sector_id, total_capital in capital_per_sector_dict.items():
        # Less than one allows for some margin to avoid round-off errors
        assert total_capital - max_capital_per_sector < 1, \
            "\n\n--> ERROR: The total capital invested in sector %d ($%d) is greater than the " \
            "maximum allowed ($%d)\n\n" % (sector_id, total_capital, max_capital_per_sector)
#-------------------------------------------------------------------
def check_tx_sizes(final_txs, min_tx_magnitude):
    # The following function ensures that all the transactions that
    # were decided on are greater than the min_tx_magnitude

    for ticker in final_txs.index:
        if not final_txs.at[ticker, 'Forced_Tx?']:
            assert round(final_txs.at[ticker, 'Buy_Amt'], 2) >= round(min_tx_magnitude, 2) or \
                   round(final_txs.at[ticker, 'Buy_Amt'], 2) == 0, \
                "\n\n--> ERROR: The buy transaction for %s has a magnitude of $%d, which is lower than " \
                "the min_tx_magnitude of $%d" % (ticker, final_txs.at[ticker, 'Buy_Amt'], min_tx_magnitude)

            assert round(final_txs.at[ticker, 'Sell_Amt'], 2) >= round(min_tx_magnitude, 2) or \
                   round(final_txs.at[ticker, 'Sell_Amt'], 2) == 0, \
                "\n\n--> ERROR: The sell transaction for %s has a magnitude of $%d, which is lower than " \
                "the min_tx_magnitude of $%d" % (ticker, final_txs.at[ticker, 'Sell_Amt'], min_tx_magnitude)
#-------------------------------------------------------------------
def liquidate_protector_etf(final_txs, cash_kept, protector_etf, fee_manager):
    # This function is only called if protectors is engaged and the user wants to invest in cash when
    # using protectors. If this is the case, the rest of the binpacker assumes that the vessel for 
    # protector is still the protector_etf. When this function is called, all holdings in the protector_etf
    # are sold for cash. Throughout the binpacker, the program thinks that it needs to pay fees to buy the
    # protector_etf when it really doesn't need to (because selling automatically results in cash - there is
    # no need to "buy" cash). This throws off the binpacker's calculations by small amounts so it isn't an
    # ideal solution but it's reasonable because the errors just result in small amounts of extra cash.

    if final_txs.at[protector_etf, 'Curr_Holdings'] > 0:
        # If you have some holdings in the protectors ETF, sell everything you have for cash
        final_txs.at[protector_etf, 'Target_Holdings'] = 0

        if final_txs.at[protector_etf, 'Buy_Amt'] > 0:
            cash_kept += final_txs.at[protector_etf, 'Buy_Amt']
            final_txs.at[protector_etf, 'Buy_Amt'] = 0

        # The following step is necessary because the portfolio simulator asserts that the remaining
        # cash is equal to what was expected (cash_kept represents what was expected)
        if final_txs.at[protector_etf, 'Sell_Amt'] > 0:
            # If part of the position in the protector_etf has already been sold, the trading_commission
            # doesn't need to be paid for any subsequent sell transactions (can treat all sell transactions
            # on one security as one large transaction)
            cash_kept += fee_manager.get_cash_amt_for_sell_size(final_txs.at[protector_etf, 'Hold_Amt'],
                                                                protector_etf) + fee_manager.trading_commission
        else:
            # If there were no sell transactions for the protector_etf, then
            # this is the first one and the trading commission must be paid
            cash_kept += fee_manager.get_cash_amt_for_sell_size(final_txs.at[protector_etf, 'Hold_Amt'], protector_etf)

        final_txs.at[protector_etf, 'Sell_Amt'] += final_txs.at[protector_etf, 'Hold_Amt']
        final_txs.at[protector_etf, 'Hold_Amt'] = 0
    else:
        # If you don't have any holdings in the protectors ETF, delete all
        # transactions for the protectors ETF
        # The following two steps are necessary because the portfolio simulator asserts that the remaining
        # cash is equal to what was expected (cash_kept represents what was expected)
        cash_kept += final_txs.at[protector_etf, 'Buy_Amt']
        final_txs = final_txs.drop(protector_etf)

    return final_txs, cash_kept
#-------------------------------------------------------------------
def calculate_optimized_txs (package_db,
                             market_frame, capital, cash, curr_holdings_frame, in_market_holdings_frame, target_port_frame,
                             excess_threshold, excess_size_penalty, min_tx_magnitude, uncertainty_factor,
                             max_capital_per_sector, target_investment_per_stock, fee_manager, tax_manager,
                             use_tax_uncertainty, protector_engaged, protector_investment, protector_etf, min_cash_to_keep,
                             target_date, verbose):
    # Ignoring package_db, this function takes several batches of parameters as input (organized as one batch per line) -
    # the first batch reflects the holdings and targeted holdings in various formats. The second batch
    # has parameters that are used to control the optimization process. The third batch has properties of
    # interest. The fourth batch controls "protectors" and taxes.

    assert protector_engaged or protector_investment == 0, protector_investment

    # Step 1
    #
    # Get a dataframe with a working copy of the possible transactions we can do (including all
    # in-market holdings (which could be sold if we wanted to) and all target buys (which could be bought
    # if we wanted to)).
    # We are calling this the currently_available_tx_df because this working copy will be edited by the various
    # optimizer phases below.

    currently_available_tx_df = get_discretionary_tx_df(package_db,
        in_market_holdings_frame, curr_holdings_frame, target_port_frame, cash, fee_manager, tax_manager,
        use_tax_uncertainty, excess_threshold,
        excess_size_penalty, uncertainty_factor, target_investment_per_stock, min_tx_magnitude,
        protector_engaged, protector_etf, target_date)

    # Step 2
    #
    # Now resolve any sales we are forced to make. There are three types of forced sales. The first
    # happens when a holding is not in the market frame. The second happens when a position is smaller
    # than the minimum transaction size.
    # The third happens when protectors is engaged and forces some of the portfolio to be
    # liquidated into a safer holding, or when impending tax payments require liquidating some holdings
    # to cash.
    #
    # This step uses the currently_available_tx_df because of protectors
    # (which can sell things that are otherwise discretionary).

    final_tx_columns = ["CurrHeld", "Target", "HoldAmt", "BuyAmt", "SellAmt", "HoldTarget",
                        "BuyTarget", "SellTarget", "SectorId", "Forced Tx?", "Reason"]
    # This function call initializes final_tx_dataframe so that each ticker has a row
    # The values in each row will later be updated in both steps 2 and 3
    final_tx_dataframe = initialize_final_tx_dataframe(currently_available_tx_df, final_tx_columns)

    # Pass final_tx_dataframe as the "final_tx_dataframe" that the function uses to build up the list.
    cash, final_tx_dataframe = resolve_out_of_market_forced_sells(
        curr_holdings_frame, cash, market_frame, fee_manager, final_tx_columns,
        protector_engaged, protector_etf, capital,
        final_tx_dataframe, verbose)
    cash, final_tx_dataframe, currently_available_tx_df = resolve_pre_optimization_small_position_forced_sells(
        cash, currently_available_tx_df, min_tx_magnitude, protector_etf, fee_manager, final_tx_dataframe, verbose)
    cash, final_tx_dataframe, currently_available_tx_df = resolve_protectorrails_and_taxes(
        curr_holdings_frame, cash, market_frame, fee_manager, final_tx_columns,
        protector_engaged, protector_investment, protector_etf, capital,
        currently_available_tx_df, final_tx_dataframe, min_cash_to_keep, verbose)
    currently_available_tx_df.at["Cash", 'CurrHeld'] = cash

    # Step 3
    #
    # Use an optimization step to finalize the list of transactions.

    final_txs, cash_kept = optimize_and_finalize_txs(
        package_db, target_date, final_tx_dataframe, cash, currently_available_tx_df, max_capital_per_sector,
        fee_manager, capital, min_tx_magnitude, final_tx_columns,
        target_investment_per_stock, excess_threshold, protector_etf, min_cash_to_keep, market_frame, verbose)

    # Return the final transaction list, and the final cash value
    return final_txs, cash_kept
#-------------------------------------------------------------------
def gen_ideal_targets(package_db, cash, curr_holdings_frame, strategy, market_frame, capital, target_port_size,
                      max_capital_per_sector, fee_manager, in_market_holdings_frame,
                      protector_engaged, protector_etf, post_protector_stocks, min_cash_to_keep,
                      correlation_threshold, correlation_dataframe, verbose):
    # Generates the target portfolio (the ideal portfolio). Returns (i) a dataframe of the targets,
    # (ii) the target value per stock used when making the dataframe, and (iii) the target value in
    # a cash-like ETF for protectors (if applicable, otherwise 0).

    if strategy == "advanced":
        assert market_frame.shape[0] >= target_port_size \
            , "\n\n--> ERROR: The market frame only has %d rows, less than the %d required\n\n" \
            % (market_frame.shape[0], target_port_size)

        # Total capital after all fees have been deducted is not possible to calculate because it
        # depends on the target transactions that need to be made which depends on the amount of
        # capital after fees have been deducted (creates a circular dependency). The following loop
        # tries different values of final capital (capital after fees) and checks whether or not
        # they would be valid. Validity is based on whether or not that much capital is left after
        # fees have been deducted.

        # max_trade_bid_ask_spread is the worst possible bid-ask spread. Given this worst case
        # assumption, it is possible to figure out the worst case scenario in terms of how much capital
        # is left after paying fees. This assumption must be made because the real bid-ask spread may
        # be a different value for different stocks.
        max_trade_bid_ask_spread = fee_manager.get_spread(max_mode=True)
        # min_target starts at a value equal to capital minus the fees required to sell all holdings
        # then buy stock with all the cash, all divided by 18 (assume whole portfolio is allocated
        # incorrectly and the number of transactions necessary to correct this is 1 per stock in the
        # current and target portfolio)
        capital_after_sells = capital * (1 - max_trade_bid_ask_spread) - curr_holdings_frame.shape[0] * fee_manager.trading_commission
        # min_cash_to_keep will automatically use up capital as if it is a stock that needs to be bought
        min_target = ((capital_after_sells - min_cash_to_keep - (target_port_size * fee_manager.trading_commission)) *
                      (1 - max_trade_bid_ask_spread)) / target_port_size

        # max_target is equal to capital divided by 18 (assume whole portfolio is already allocated
        # correctly so no transactions are necessary and therefore no money will be lost in fees)
        # min_cash_to_keep will automatically use up capital as if it is a stock that needs to be bought
        max_target = capital / target_port_size

        max_stocks_per_sector = round(max_capital_per_sector / max_target)
        market_frame['ValidSectorLimit'] = 0
        market_frame.loc[market_frame['SectorAdvancedScore'] <= max_stocks_per_sector, 'ValidSectorLimit'] = 1
        # Make sure there are enough stocks that can be part of the target portfolio
        assert market_frame['ValidSectorLimit'].sum() >= post_protector_stocks, \
            (market_frame, market_frame['ValidSectorLimit'].sum(), post_protector_stocks, max_stocks_per_sector)
        market_frame = market_frame.sort_values(by=['ValidSectorLimit', 'Advanced_Score'], ascending=False)

        target_tickers_list = market_frame.iloc[0:post_protector_stocks, :].index.tolist()
        # Flip target_tickers_list so it is sorted in ascending order of advanced score
        target_tickers_list.reverse()

        # Only create a correlation_dataframe if it isn't already passed in by the portfolio simulator
        # This feature is used by the transaction manager tester which creates its own randomized correlation_dataframe
        if correlation_threshold < 1.0 and correlation_dataframe is None:
            held_tickers_list = in_market_holdings_frame[in_market_holdings_frame['Investment'] > 0].index.tolist()

            # Only keep secs that are in target_tickers_list but aren't in held_tickers_list
            unique_target_secs_list = list(set(target_tickers_list).difference(held_tickers_list))
            # Convert all tickers to security_ids
            unique_target_secs_list = [vu.get_security_id_for_ticker(package_db, i) for i in unique_target_secs_list]

            # Only keep secs that are in held_tickers_list but aren't in target_tickers_list
            unique_held_secs_list = list(set(held_tickers_list).difference(target_tickers_list))
            # Convert all tickers to security_ids
            unique_held_secs_list = [vu.get_security_id_for_ticker(package_db, i) for i in unique_held_secs_list]

            # Each rows of the correlation_frame corresponds to a stock in unique_target_secs_list
            # and each column corresponds to a stock unique_held_secs_list
            correlation_frame = vu.get_correlation_matrix_frame(
                package_db, unique_target_secs_list, unique_held_secs_list, only_correlate_same_sector=True)

            # For each stock in unique_target_secs_list, try to find a replacement from unique_held_secs_list. This is
            # done by creating a correlation frame that tracks how closely any two stocks prices are in the last 200 days.
            # Any stock in unique_target_secs_list can be swapped out with a stock in unique_held_secs_list that has a
            # high enough correlation with it.
            for index in correlation_frame.index.tolist():
                if correlation_frame.shape[1] == 0:
                    break
                row = correlation_frame.loc[index, :]

                max_correlation = max(row.tolist())
                if max_correlation > correlation_threshold:
                    max_correlation_sec = row[row == max_correlation].index[0]
                    old_ticker = vu.get_ticker_for_security_id(package_db, index)
                    new_ticker = vu.get_ticker_for_security_id(package_db, max_correlation_sec)

                    target_tickers_list.remove(old_ticker)
                    target_tickers_list.append(new_ticker)

                    if verbose:
                        print("Switched %s in for %s in the list of %s target stocks (have a correlation of %.2f)"
                              % (new_ticker, old_ticker, post_protector_stocks, row[max_correlation_sec]))

                    # Once a stock from unique_held_secs_list is used to replace a
                    # stock from unique_target_secs_list, it cannot be used again
                    correlation_frame = correlation_frame.drop(columns=max_correlation_sec)

        # Use the forced sell code to get a value of cash that the forced sales will provide
        forced_sell_cash_supply, ignore_this_value = resolve_out_of_market_forced_sells(
            curr_holdings_frame, 0, market_frame, fee_manager,
            None, protector_engaged, protector_etf, capital, None, verbose)

        # Error threshold is the maximum amount of capital held in cash if all the transactions are resolved
        # (some cash will remain in excess and won't be used up to fulfill the transactions)
        error_threshold = 100
        # Excess cash tracks how much cash is unused given certain targets
        excess_cash = error_threshold + 1  # Default value that allows loop to run
        test_target = None
        protector_etf_target = 0
        attempts = 0

        test_target_port_frame = None  # Default value, will later be a dataframe
        # Loop runs until the excess_cash is not negative (debt) and is sufficiently low
        while round(excess_cash, 2) > error_threshold or round(excess_cash, 2) < 0:
            # Every iteration of the loop, pick a test_target value in between the min and max
            # target values and check what the real final_capital would have been. Then, update
            # the min or max target values accordingly. test_target represents the value put into
            # each stock in the ideal portfolio.
            attempts += 1
            if attempts == 100:
                assert False, "\n\n--> ERROR: Failed to find a target after 100 attempts\n\n"

            test_target = (min_target + max_target)/2

            if verbose:
                print("\nTrying target of $%.0f:" % test_target)

            # test_target_port_frame is a dataframe with the ideal portfolio using the
            # target value that is currently being tested
            # Note that test_target_port_frame will not include any stocks that are not in market_frame
            # (eg. acquisitions, bankruptcies, etc) - those are forced sales that need special case
            # treatment after target_txs is populated.
            test_target_port_frame = market_frame.copy(deep=False)
            test_target_port_frame['Investment'] = 0
            test_target_port_frame.loc[target_tickers_list, 'Investment'] = test_target

            protector_etf_holdings = curr_holdings_frame.at[protector_etf, 'Investment Total USD'] \
                if protector_etf in curr_holdings_frame.index.tolist() else 0
            # Put the money that isn't going into stocks into the protector_etf
            protector_etf_target = test_target * (target_port_size - post_protector_stocks)

            # Run the following code if you want capital in the protector_etf or
            # if you already have capital in the protector_etf
            if protector_etf_holdings != 0 or protector_etf_target != 0:
                # Remove any row with the protector_etf that was previously concatenated to in_market_holdings_frame
                if protector_etf in in_market_holdings_frame.index.tolist():
                    in_market_holdings_frame = in_market_holdings_frame.drop(index=protector_etf)
                # Add an empty row into in_market_holdings_frame and then change the 'Investment' value
                in_market_holdings_frame = pd.concat([in_market_holdings_frame, pd.DataFrame(
                    [[None] * in_market_holdings_frame.shape[1]], columns=in_market_holdings_frame.columns,
                    index=[protector_etf])])
                in_market_holdings_frame.at[protector_etf, 'Investment'] = protector_etf_holdings

                # Remove any row with the protector_etf that was previously concatenated to test_target_port_frame
                if protector_etf in test_target_port_frame.index.tolist():
                    test_target_port_frame = test_target_port_frame.drop(index=protector_etf)
                # Add an empty row into test_target_port_frame and then change the 'Investment' value
                test_target_port_frame = pd.concat([test_target_port_frame, pd.DataFrame(
                    [[None] * test_target_port_frame.shape[1]], columns=test_target_port_frame.columns,
                    index=[protector_etf])])
                test_target_port_frame.at[protector_etf, 'Investment'] = protector_etf_target

            # target_txs is the list of transactions necessary to go from the current portfolio to the ideal portfolio
            # Again, see "Note" about forced sales before this binary-search loop.
            target_txs = get_target_txs(in_market_holdings_frame, test_target_port_frame)

            supply_value = cash + forced_sell_cash_supply
            for ticker in target_txs[target_txs < 0].index:
                supply_value += fee_manager.get_cash_amt_for_sell_size(abs(target_txs[ticker]), ticker)
            # min_cash_to_keep tracks the minimum amount of cash that needs to be kept at the end of the optimization
            # process making it a factor that uses up some cash that cannot be spent to buy stocks.
            demand_value = min_cash_to_keep
            for ticker in target_txs[target_txs > 0].index:
                demand_value += fee_manager.get_buy_size_for_required_stock_amt(target_txs[ticker], ticker)

            excess_cash = supply_value - demand_value
            if excess_cash < 0:
                max_target = test_target
                if verbose:
                    print("Target of $%d is too high and is impossible to reach ($%d short of cash to "
                          "reach ideal portfolio)" % (test_target, abs(excess_cash)))
            elif excess_cash > error_threshold:
                min_target = test_target
                if verbose:
                    print("Target of $%d is too low and leads to too much excess cash ($%d in excess cash)"
                          % (test_target, excess_cash))
            elif verbose:
                print("Target of $%d has been accepted because it minimizes excess cash ($%d in excess cash)"
                      % (test_target, excess_cash))

        # When the loop exits, test_target_port_frame is now the accepted target_port_frame
        # and test_target is now the accepted target
        ideal_target = test_target
        final_target_port_frame = test_target_port_frame.copy(deep=False)
        return final_target_port_frame, ideal_target, protector_etf_target
    else:
        assert False, "\n\n--> ERROR: %s strategy is not supported yet\n\n" % strategy
#-------------------------------------------------------------------
def initialize_final_tx_dataframe(currently_available_tx_df, final_tx_columns):
    # This function initializes the final_tx_dataframe by adding to it all the transactions
    # All hold and sell transactions are considered to be resolved, whereas buy transactions aren't resolved
    # In step 3 of the optimization process, some of these resolved sells will be
    # undone and some unresolved buys will be resolved

    final_tx_dataframe = pd.DataFrame(columns=final_tx_columns)
    final_tx_dataframe.index.name = "Ticker"

    currently_available_tx_df = currently_available_tx_df.sort_values(by='MonthlyROI', ascending=False)
    # Add a row to final_tx_dataframe for every stock in currently_available_tx_df
    for index, row in currently_available_tx_df.iterrows():
        if index in final_tx_dataframe.index.tolist():
            pass
        elif str(index)[0:len(EXCESS_IDENTIFIER)] == EXCESS_IDENTIFIER:
            ticker = index[len(EXCESS_IDENTIFIER):]
            final_tx_dataframe.loc[ticker, 'CurrHeld'] += row['CurrHeld']
            final_tx_dataframe.loc[ticker, 'SellAmt'] += row['SellTarget']
            final_tx_dataframe.loc[ticker, 'SellTarget'] += row['SellTarget']
        else:
            final_tx_dataframe = pd.concat([final_tx_dataframe,
                pd.DataFrame([[row['CurrHeld'], row['Target'], row['HoldTarget'], 0.0, row['SellTarget'],
                               row['HoldTarget'], row['BuyTarget'], row['SellTarget'], row['SectorId'], False, '']],
                             columns=final_tx_columns, index=[index])])

    # Cash is a byproduct of other transactions so it doesn't have to have its own transaction
    final_tx_dataframe = final_tx_dataframe.drop(index="Cash")
    return final_tx_dataframe
#-------------------------------------------------------------------
def get_target_txs(in_market_holdings_frame, target_port_frame):
    # Make a series (starting with a dataframe) that shows the gap between desired and held values (in US dollars)
    target_tx_series = target_port_frame[['Investment', 'Advanced_Score']] \
                       - in_market_holdings_frame[['Investment', 'Advanced_Score']]
    target_tx_series['Advanced_Score'] = target_port_frame['Advanced_Score']
    # Filter the series to only hold things that need transactions
    target_tx_series = target_tx_series[target_tx_series['Investment'] != 0].sort_values(
        by=['Investment', 'Advanced_Score'], ascending=False)
    # This is where it actually becomes a series from a dataframe
    target_tx_series = target_tx_series['Investment']
    return target_tx_series
#-------------------------------------------------------------------
def get_discretionary_tx_df (package_db, in_market_holdings_frame, curr_holdings_frame, target_port_frame,
                             cash, fee_manager, tax_manager, use_tax_uncertainty,
                             excess_threshold, excess_size_penalty, uncertainty_factor, target_investment_per_stock,
                             min_tx_magnitude, protector_engaged, protector_etf, target_date):
    # This function returns a dataframe which shows the stocks that can be discretionarily traded.
    # Because it is only discretionary transactions, this dataframe doesn't include securities
    # that aren't in the market frame (those will be force sold). This function also has a unique behaviour with
    # regards to excess holdings (stocks in which the position is excessively large). For such positions, this
    # function splits that position into two pieces: the excess holding and the regular holding. For example, if
    # MSFT-US had a holding size of $3000 and the excess threshold was $2000, this holding would get split into
    # two pieces: an excess holding of size $1000 ($3000 - $2000) and a regular holding of size $2000 (equal to
    # the excess threshold of $2000).
    in_market_port_securities = in_market_holdings_frame[in_market_holdings_frame['Investment'] > 0].index.tolist()
    target_port_securities = target_port_frame[target_port_frame['Investment'] > 0].index.tolist()

    # If the protector_etf is in the chosen securities list, remove it
    if protector_etf in target_port_securities:
        target_port_securities.remove(protector_etf)

    # tx_dataframe tracks the following 8 metrics
    tx_df_columns = ['CurrHeld', 'Target', 'TxType', 'BuyTarget', 'SellTarget', 'HoldTarget', 'MonthlyROI',
                     'SectorId']
    tx_dataframe = pd.DataFrame([[cash, None, None, None, None, None, 0, None]], columns=tx_df_columns, index=['Cash'])

    # in_market_and_target_tickers includes all unique tickers in in_market_port_securities and target_port_securities
    in_market_and_target_tickers = list(set(in_market_port_securities + target_port_securities))
    for ticker in in_market_and_target_tickers:
        in_market_held = in_market_holdings_frame.loc[ticker, 'Investment']
        excess_capital_in_stock = max(0, in_market_held - (target_investment_per_stock * excess_threshold))
        if excess_capital_in_stock > min_tx_magnitude:
            # If the holdings in the stock exceed a certain amount (target_investment*excess_threshold),
            # then two transaction blocks are created: one for all the holdings within the threshold,
            # and another for all the holdings exceeding the threshold
            # The optimization process relies on the excess portion of a stock to have lower
            # monthly ROI than the regular portion of the stock
            assert excess_size_penalty > 0
            excess_tx_summary_frame = gen_tx_summary_for_stock(package_db, curr_holdings_frame, target_port_frame,
                                                               fee_manager, tax_manager, use_tax_uncertainty,
                                                               excess_capital_in_stock, 0, tx_df_columns, ticker,
                                                               in_market_port_securities, uncertainty_factor, target_date,
                                                               excess=True, excess_size_penalty=excess_size_penalty)
            tx_dataframe = pd.concat([tx_dataframe, excess_tx_summary_frame])
            in_market_held -= excess_capital_in_stock

        tx_summary_frame = gen_tx_summary_for_stock(package_db, curr_holdings_frame, target_port_frame,
                                                    fee_manager, tax_manager, use_tax_uncertainty,
                                                    in_market_held, target_investment_per_stock, tx_df_columns, ticker,
                                                    in_market_port_securities, uncertainty_factor, target_date)
        tx_dataframe = pd.concat([tx_dataframe, tx_summary_frame])
    return tx_dataframe
#-------------------------------------------------------------------
def gen_tx_summary_for_stock(package_db, curr_holdings_frame, target_port_frame, fee_manager, tax_manager,
                             use_tax_uncertainty, in_market_held, target_investment, column_names, ticker,
                             in_market_port_securities, uncertainty_factor,
                             target_date, excess=False, excess_size_penalty=0):
    # This function is called by get_discretionary_tx_df and it creates a row of data for a given stock.
    # This data will be useful when the optimizer is deciding whether or not to transact on a stock.

    tx_summary = pd.DataFrame([[0.0, 0.0, "", 0.0, 0.0, 0.0, 0.0, 0]], columns=column_names, index=[ticker])
    # CurrHeld tracks how much of the stock is currently held (but only applies for in-market holdings)
    tx_summary.at[ticker, 'CurrHeld'] = in_market_held
    # Target tracks how much of the stock would be held in the ideal portfolio
    # Target is 0 unless the ticker is in the target_port_frame ("Excess" transactions will also have a target of 0)
    tx_summary.at[ticker, 'Target'] = target_investment \
        if ticker in target_port_frame[target_port_frame['Investment'] > 0].index.tolist() and not excess else 0
    # BuyTarget tracks how much of the stock to buy to reach the ideal portfolio
    # BuyTarget should represent the amount of money it would take to buy enough stock to reach
    # the target (so it should include fees that need to be paid to resolve the transaction)
    tx_summary.at[ticker, 'BuyTarget'] = max(fee_manager.get_buy_size_for_required_stock_amt(
        tx_summary.at[ticker, 'Target'] - tx_summary.at[ticker, 'CurrHeld'], ticker), 0)

    if ticker in in_market_port_securities:
        # SellTarget tracks how much of the stock to sell to reach the ideal portfolio
        tx_summary.at[ticker, 'SellTarget'] = max(tx_summary.at[ticker, 'CurrHeld'] -
                                                  tx_summary.at[ticker, 'Target'], 0)
        # HoldTarget tracks how much of the stock to hold to reach the ideal portfolio
        tx_summary.at[ticker, 'HoldTarget'] = tx_summary.at[ticker, 'CurrHeld'] - \
            tx_summary.at[ticker, 'SellTarget']

    if tx_summary.at[ticker, 'BuyTarget'] > 0:
        # TxType tracks the type of transaction; chosen from: "Buy", "Sell", "Hold", and "Excess Sell"
        tx_summary.at[ticker, 'TxType'] = "Buy"
        # If buying the stock, MonthlyROI tracks how much return the stock would provide
        # if it is bought (bid ask spread included)
        advanced_rank = target_port_frame.at[ticker, 'Advanced_Rank']
        # The formula for return on investment is derived from past performance data (for more details,
        # check Package\Advanced\score_to_return_prediction\AdvancedRk_vs_Premium_Correlation).
        # For buy transactions, subtract the bid ask spread because it reduces returns.
        # Also apply uncertainty_factor to avoid buying without significant gain.
        return_on_investment = vu.get_expected_monthly_return(
            advanced_rank, -(fee_manager.get_spread(ticker) + uncertainty_factor), target_date)
        tx_summary.at[ticker, 'MonthlyROI'] = return_on_investment
    elif tx_summary.at[ticker, 'SellTarget'] > 0:
        # TxType tracks the type of transaction; chosen from: "Buy", "Sell", "Hold", and "Excess Sell"
        if excess:
            tx_summary.at[ticker, 'TxType'] = "Excess Sell"
            used_excess_penalty = excess_size_penalty
        else:
            tx_summary.at[ticker, 'TxType'] = "Sell"
            used_excess_penalty = 0
        # If selling the stock, MonthlyROI tracks how much return the stock would provide
        # if it is held and the sell transaction is ignored (includes a size penalty -
        # don't want to hold too much of one stock)
        advanced_rank = target_port_frame.at[ticker, 'Advanced_Rank']

        # If applicable, use tax information to adjust the expected return based on gain/loss status and tax rates.
        if use_tax_uncertainty:
            assert tax_manager is not None
            this_inv_value = float(curr_holdings_frame.at[ticker, 'Investment Total USD'])
            assert this_inv_value > 0, (ticker, this_inv_value, curr_holdings_frame, target_port_frame)
            sec_id = vu.get_security_id_for_ticker (package_db, ticker)
            holding_period_so_far = tax_manager.holding_period_dict[sec_id]
            # A negative tax adder means it will increase the expected return, making it less likely to sell.
            # The tax adder calculation below will get a positive adder if there are unrealized losses,
            # which makes the expected return lower, and therefore more likely to sell. The 1/12 is to make
            # it a monthly return value.
            if holding_period_so_far < 12:
                tax_adder = -(1.0/12.0)*((tax_manager.unrealized_gains_dict[sec_id]*tax_manager.short_term_tax_rate)/this_inv_value)
            else:
                tax_adder = -(1.0/12.0)*((tax_manager.unrealized_gains_dict[sec_id]*tax_manager.long_term_tax_rate)/this_inv_value)
        else:
            tax_adder = 0

        # The formula for return on investment is derived from past performance data (for more details,
        # check Package\Advanced\AdvancedRk_vs_Premium_Correlation)
        # For excess sell transactions, subtract the excess_size_penalty to factor in the risk of not diversifying
        return_on_investment = vu.get_expected_monthly_return(advanced_rank, -(used_excess_penalty+tax_adder), target_date)
        tx_summary.at[ticker, 'MonthlyROI'] = return_on_investment
    else:
        tx_summary.at[ticker, 'TxType'] = "Hold"

    tx_summary.at[ticker, 'SectorId'] = target_port_frame.at[ticker, 'Sector_Id']

    if excess:
        tx_summary.index = [EXCESS_IDENTIFIER + ticker]
    return tx_summary
#-------------------------------------------------------------------
def resolve_out_of_market_forced_sells(curr_holdings_frame, cash, market_frame, fee_manager,
        final_tx_columns, protector_engaged, protector_etf, capital,
        final_tx_dataframe, verbose):
    # This function sells all securities that aren't in the market frame to minimize risk.
    # If final_tx_columns is None, then only cash will be returned, and final_tx_dataframe is returned as None.
    # This allows this function to be called by the early step that binary searches for a target per stock.
    assert (final_tx_columns is None and final_tx_dataframe is None) or \
           (final_tx_columns is not None and final_tx_dataframe is not None)
    for ticker in curr_holdings_frame.index.tolist():
        # If the ticker is the protector_etf, its forced transaction will be handled in the
        # "resolve_protectorrails_and_taxes" function so there is no need to worry about it here
        if ticker == "USDCAD=X" or ticker == "1" or ticker == protector_etf:
            pass
        # If the ticker isn't in the market frame then force sell it
        elif ticker not in market_frame.index.tolist():
            # If the holding is not cash and is not the protector_etf and isn't in the market frame, then sell it
            value = curr_holdings_frame.at[ticker, 'Investment Total USD']
            cash_earned = fee_manager.get_cash_amt_for_sell_size(value, ticker)
            cash += cash_earned
            # Don't print the verbose statement if this function is being called from the target
            # finder function (if it is, then final_tx_columns will be None so it won't print)
            if verbose and final_tx_columns is not None:
                print("\nFound ticker %s that isn't in the market frame and sold it for USD $%d"
                      % (ticker, cash_earned))
            if final_tx_columns is not None:
                final_tx_dataframe = pd.concat([final_tx_dataframe, pd.DataFrame(
                    [[value, 0, 0, 0, value, 0, 0, value, None, True, "Forced Sell - Done to minimize losses because "
                                                                      "this stock is not in the market frame"]],
                    columns=final_tx_columns, index=[ticker])])
    if final_tx_columns is None:
        return cash, None
    else:
        return cash, final_tx_dataframe
#-------------------------------------------------------------------
def resolve_protectorrails_and_taxes(curr_holdings_frame, cash, market_frame, fee_manager,
                                   final_tx_columns, protector_engaged, protector_investment, protector_etf, capital,
                                   currently_available_tx_df, final_tx_dataframe, min_cash_to_keep, verbose):
    # This function sells all securities that need to be liquidated for protectors.
    # It also sells some securities to generate cash that will be kept to pay taxes.

    curr_protector_investment = curr_holdings_frame.at[protector_etf, 'Investment Total USD'] \
        if protector_etf in curr_holdings_frame.index.tolist() else 0
    target_protector_tx_size = protector_investment - curr_protector_investment
    # total_cash_required tracks how much cash is necessary to buy up to the target of the protector_etf
    # and have cash equal to or greater than min_cash_to_keep at the end.
    total_cash_required = min_cash_to_keep - cash

    if target_protector_tx_size > 0:
        # In this case, you need to buy more of the protector_etf which will use up a certain amount of cash
        # To reach the target dollars worth of buys of the protector_etf, you need to use up extra capital
        # which will go towards paying the fees
        target_protector_tx_size_with_fees = fee_manager.get_buy_size_for_required_stock_amt(
            target_protector_tx_size, protector_etf)
        total_cash_required += target_protector_tx_size_with_fees
    elif target_protector_tx_size < 0:
        # In this case you have too much of the protector_etf and force selling it will generate some more cash
        target_protector_tx_size_with_fees = fee_manager.get_cash_amt_for_sell_size(
            abs(target_protector_tx_size), protector_etf)
        total_cash_required -= target_protector_tx_size_with_fees
    else:
        target_protector_tx_size_with_fees = 0

    # cash_still_required starts off equal to total_cash_required and it will be the value that
    # is updated when cash is generated in the following loop
    cash_still_required = total_cash_required
    # If a sell transaction is smaller than "trading_commission / (1 - trade_bid_ask_spread),"
    # then there is no point in resolving that sell since it will result in a loss of capital
    # because of the fees that need to be paid
    possible_forced_sells = currently_available_tx_df.copy(deep=True)
    for index in currently_available_tx_df.index:
        ticker, is_excess = get_ticker_from_index(index)
        # The SellTarget will be None when looking at cash, so remove rows that have a SellTarget of None
        if possible_forced_sells.at[index, 'SellTarget'] is None or possible_forced_sells.at[index, 'SellTarget'] <= \
                fee_manager.trading_commission / (1 - fee_manager.get_spread(ticker)):
            possible_forced_sells = possible_forced_sells.drop(index=index)
    possible_forced_sells = possible_forced_sells.sort_values(by='MonthlyROI', ascending=True)

    max_cash_generatable = 0
    for index in possible_forced_sells.index:
        ticker, is_excess = get_ticker_from_index(index)
        max_cash_generatable += fee_manager.get_cash_amt_for_sell_size(
            possible_forced_sells.at[index, 'SellTarget'], ticker)

    # Trading commissions were paid for every sell transaction in possible_forced_sells when coming up
    # with max_cash_generateable. But, some transactions were excess transactions, so they were the
    # second sell transaction for a given ticker. In these cases, the trading commission was already
    # paid for any transaction with that ticker, so it doesn't need to be paid again - add back in the
    # trading commissions that were for excess transactions
    max_cash_generatable += possible_forced_sells[possible_forced_sells['TxType'] ==
                                                  "Excess Sell"].count()['TxType'] * fee_manager.trading_commission
    assert max_cash_generatable > cash_still_required, (max_cash_generatable, cash_still_required)

    excess_txs_completed = []
    for index in possible_forced_sells.index.tolist():
        if cash_still_required <= 0:
            break

        ticker, excess = get_ticker_from_index(index)
        if excess:
            excess_txs_completed.append(ticker)

        # Update the cash_still_require value based on how much stock you sold to provide capital
        if not excess and ticker in excess_txs_completed:
            # If the excess portion of this stock was already sold, then there is no need to pay the
            # trading commission again for the non-excess portion of the same stock as they can be
            # grouped into the same transaction (this is why trading_commission is added in at the end)
            cash_still_required -= fee_manager.get_cash_amt_for_sell_size(
                currently_available_tx_df.at[index, 'SellTarget'], ticker) + fee_manager.trading_commission
        else:
            cash_still_required -= fee_manager.get_cash_amt_for_sell_size(
                currently_available_tx_df.at[index, 'SellTarget'], ticker)

        # The sell transaction for all stocks has already been resolved, and the ony way it can be undone
        # is if it gets chosen as part of the bin-packing algorithm in step 3. By removing it from
        # currently_available_tx_df (what the bin-packing algorithm is choosing from), you are preventing
        # the sell transaction from being undone, thereby making it a forced sell.
        currently_available_tx_df = currently_available_tx_df.drop(index=index)
        # Set the transaction to a forced transaction
        final_tx_dataframe.at[ticker, 'Forced Tx?'] = True
        # Change the reason for the transaction
        final_tx_dataframe.at[ticker, 'Reason'] = "Forced Sell - Sold to provide cash to buy the protector " \
                                                  "ETF (%s)" % protector_etf

    # The target cash required should be reached at this point and you shouldn't need to buy any more
    assert cash_still_required <= 0, (cash_still_required, possible_forced_sells)

    # The extra cash generated doesn't have to be used anywhere so it can be held
    cash += abs(cash_still_required)

    final_tx_dataframe = pd.concat([final_tx_dataframe,
                     pd.DataFrame([[curr_protector_investment, protector_investment, 0, 0, 0, 0, 0, 0, None, True,
                       "Protector - Bought a protector ETF to hedge against a large market crash"]],
                       columns=final_tx_columns, index=[protector_etf])])

    # When resolving the transaction for the protector_etf, force sell it as much as
    # needed to get to the target if more of the protector_etf is being held than
    # necessary, otherwise buy more until you hit the target
    if curr_protector_investment > protector_investment:
        final_tx_dataframe.at[protector_etf, 'SellAmt'] = abs(target_protector_tx_size)
        final_tx_dataframe.at[protector_etf, 'SellTarget'] = abs(target_protector_tx_size)

        final_tx_dataframe.at[protector_etf, 'HoldAmt'] = protector_investment
        final_tx_dataframe.at[protector_etf, 'HoldTarget'] = protector_investment
    else:
        # When buying, use up some extra cash to end up buying a total
        # amount of stock worth target_protector_tx_size
        final_tx_dataframe.at[protector_etf, 'BuyAmt'] = target_protector_tx_size_with_fees \
            if target_protector_tx_size > 0 else 0
        final_tx_dataframe.at[protector_etf, 'BuyTarget'] = target_protector_tx_size_with_fees \
            if target_protector_tx_size > 0 else 0

        final_tx_dataframe.at[protector_etf, 'HoldAmt'] = curr_protector_investment
        final_tx_dataframe.at[protector_etf, 'HoldTarget'] = curr_protector_investment

    return cash, final_tx_dataframe, currently_available_tx_df
#-------------------------------------------------------------------
def resolve_pre_optimization_small_position_forced_sells(cash, currently_available_tx_df, min_tx_magnitude,
                                        protector_etf, fee_manager,
                                        final_tx_dataframe, verbose):
    # This function forced sells all securities that have a position size less than the min_tx_magnitude
    for ticker in currently_available_tx_df.index.tolist():
        # Excess holdings should never be force-sold in this function because they only represent a portion of the
        # total position in the stock. The real position in the stock will be much larger because it involves the
        # holdings in the stock that are not in the excess bucket.
        if ticker[0:len(EXCESS_IDENTIFIER)] == EXCESS_IDENTIFIER:
            continue
        else:
            value = currently_available_tx_df.at[ticker, 'CurrHeld']
            # If the ticker is the protector_etf, its forced transaction will be handled in the
            # "resolve_protectorrails_and_taxes" function so there is no need to worry about it here
            if ticker == "Cash" or ticker == protector_etf:
                continue
            # If the holdings for the stock are below the min_tx_magnitude
            # and it is not in the target list, then forced sell it
            elif currently_available_tx_df.at[ticker, 'CurrHeld'] < min_tx_magnitude and \
                    currently_available_tx_df.at[ticker, 'Target'] == 0:
                cash_earned = fee_manager.get_cash_amt_for_sell_size(value, ticker)
                cash += cash_earned

                if verbose:
                    print("\nFound ticker %s that only had a position of $%d (less than the min_tx_magnitude "
                          "of $%d) and sold it for USD $%d" % (ticker, value, min_tx_magnitude, cash_earned))

                # The sell transaction for all stocks has already been resolved, and the ony way it can be undone
                # is if it gets chosen as part of the bin-packing algorithm in step 3. By removing it from
                # currently_available_tx_df (what the bin-packing algorithm is choosing from), you are preventing
                # the sell transaction from being undone, thereby making it a forced sell.
                currently_available_tx_df = currently_available_tx_df.drop(index=ticker)
                # Set the transaction to a forced transaction
                final_tx_dataframe.at[ticker, 'Forced Tx?'] = True
                # Change the reason for the transaction
                final_tx_dataframe.at[ticker, 'Reason'] = \
                    "Forced Sell - Sold because the position in this stock ($%d) was smaller than the " \
                    "min_tx_magnitude of $%d" % (value, min_tx_magnitude)

    return cash, final_tx_dataframe, currently_available_tx_df
#-------------------------------------------------------------------
def resolve_post_optimization_small_position_forced_sells(min_tx_magnitude, protector_etf, final_txs,
                                                          fee_manager, cash_kept, verbose):
    # This function sells any position that has a size lower than the min_tx_magnitude after all transactions
    # have been resolved. The final holding size of the stock is determined by taking the amount you are going
    # to hold plus the amount you will buy minus the fees you need to pay to complete that buy transaction
    for ticker in final_txs.index.tolist():
        stock_buy_amt = fee_manager.get_stock_amt_for_buy_size(final_txs.at[ticker, 'Buy_Amt'], ticker) \
            if final_txs.at[ticker, 'Buy_Amt'] > 0 else 0

        if 0 < round(final_txs.at[ticker, 'Hold_Amt'] + stock_buy_amt, 2) < min_tx_magnitude and ticker != protector_etf:
            # A buy transaction should never be made if, after resolving this buy transaction, the position in a
            # stock will still be too small and have to be force-sold anyway.
            assert final_txs.at[ticker, 'Buy_Amt'] == 0, (final_txs.loc[ticker, :], min_tx_magnitude)

            # The following variable checks if there is already a sell transaction for this stock
            sell_tx_made = True if final_txs.at[ticker, 'Sell_Amt'] > 0 else False
            holdings_after_txs = final_txs.at[ticker, 'Hold_Amt']
            final_txs.at[ticker, 'Sell_Amt'] += final_txs.at[ticker, 'Hold_Amt']

            if sell_tx_made:
                # If you have already made a sell transaction for a stock, there is no need to pay the
                # trading commission again for any more sell transactions you make for that stock
                fees_on_forced_sell = final_txs.at[ticker, 'Hold_Amt'] * fee_manager.get_spread(ticker)
                final_txs.at[ticker, 'Tx_Fees_Paid'] += fees_on_forced_sell
                cash_earned = final_txs.at[ticker, 'Hold_Amt'] - fees_on_forced_sell
            else:
                final_txs.at[ticker, 'Tx_Fees_Paid'] = final_txs.at[ticker, 'Hold_Amt'] - fee_manager.\
                    get_cash_amt_for_sell_size(final_txs.at[ticker, 'Hold_Amt'], ticker)
                cash_earned = final_txs.at[ticker, 'Hold_Amt'] - final_txs.at[ticker, 'Tx_Fees_Paid']

            cash_kept += cash_earned
            final_txs.at[ticker, 'Hold_Amt'] = 0
            final_txs.at[ticker, 'Forced_Tx?'] = True
            final_txs.at[ticker, 'Tx_Notes'] = "Forced Sell - the post-optimization position size " \
                                               "for this stock was too small so it was force-sold"

            if verbose:
                print("\nFound ticker %s that will only have a position of $%d after transactions are "
                      "resolved (less than the min_tx_magnitude of $%d) and sold it for %d USD"
                      % (ticker, holdings_after_txs, min_tx_magnitude, cash_earned))

    return final_txs, cash_kept
#-------------------------------------------------------------------
def optimize_and_finalize_txs (package_db, target_date, final_tx_dataframe, cash, currently_available_tx_df, max_capital_per_sector,
                  fee_manager, capital, min_tx_magnitude, final_tx_columns,
                  target_investment_per_stock, excess_threshold_multiplier, protector_etf, min_cash_to_keep,
                  market_frame, verbose):
    # This function returns a list of optimized transactions using a bin-packing algorithm.
    # This list includes previously determined transactions (eg. from forced sales).
    #
    # The bin-packing works by assuming all sell transactions will happen and then fills up remaining
    # capital with buy and sell blocks (sell blocks mean that the stock won't, after all, be sold,
    # and buy blocks mean that the buy transaction will be done). This assumption helps us cleanly track changes
    # to capital from the buy and sell decisions.

    ###################################
    # STEP 1 - Undo sells that are too small
    ###################################
    # Ignore all sell transactions that are too small in size and assume you are holding instead of selling
    # Buy transactions will get filtered in the main loop, so there is no reason to do it here
    #
    # NOTE: This loop will sometimes cause the binpacker to violate the sector limits. This is because the sector
    # limits rely on all sell transactions to go through if they are to be followed. For example, if you are holding
    # $50,000 of GOOG-US and $50,000 of FB-US then you are holding $100,000 in sector 1 even if the limit is $80,000.
    # If the targets for both GOOG-US and FB-US is $40,000, it is possible to sell $10,000 of each and fulfill the
    # sector limit. But this may be restricted because a transaction of size $10,000 may be below the min_tx_magnitude.
    # In this case, the sector limit will be violated in the interests of not violating the min_tx_magnitude.

    for ticker, row in final_tx_dataframe.query("SellAmt < %s & SellAmt > 0" % min_tx_magnitude).iterrows():
        # Never undo protector_etf transactions or forced sell transactions regardless of their transaction size
        if row['Forced Tx?']:
            continue

        # At this point in the optimization process, 'SellTarget' should always equal 'SellAmt'
        assert row['SellTarget'] == row['SellAmt'], (row['SellTarget'], row['SellAmt'])

        # If a sell transaction is smaller than the min_tx_magnitude (this loop only looks for sell transactions that
        # are small, so that requirement is already met), then it shouldn't be resolved. In this case, it should be
        # removed from currently_available_tx_df so it doesn't get chosen in the optimization loop (we are already
        # undoing the transaction, so if it is chosen, there is no transaction left to undo for this stock).
        if ticker in currently_available_tx_df.index.tolist():
            currently_available_tx_df = currently_available_tx_df.drop(index=ticker)
        if EXCESS_IDENTIFIER + ticker in currently_available_tx_df.index.tolist():
            currently_available_tx_df = currently_available_tx_df.drop(index=EXCESS_IDENTIFIER + ticker)

        final_tx_dataframe.at[ticker, 'HoldAmt'] += row['SellAmt']
        final_tx_dataframe.at[ticker, 'SellAmt'] = 0.0
        final_tx_dataframe.at[ticker, 'Reason'] = "Hold - Sell transaction is too small and " \
                                                  "therefore not worth resolving"

    ###################################
    # STEP 2 - Figure out remaining_capital and sector limit dictionary
    ###################################
    sector_list = package_db.get_list_for_query("SELECT Sector_Id FROM Sectors")
    # remaining_sector_capital_dict is a dictionary that keeps track of how
    # much more capital can be invested in a given sector
    remaining_sector_capital_dict = {i[0]: max_capital_per_sector - final_tx_dataframe[
        final_tx_dataframe['SectorId'] == i[0]]['HoldAmt'].sum() for i in sector_list}

    # remaining_capital tracks how much capital still remains to be invested
    remaining_capital = capital

    # Update remaining_capital accordingly For each transaction that has already been resolved
    for ticker, row in final_tx_dataframe.iterrows():
        # remaining_capital starts off as the total value of all your holdings, so it assumes you
        # would have received the full (no fee) amount of your sell transactions. Any existing sell transactions
        # therefore don't reflect the fees paid, so those fees must be subtracted here. The fees are equal to
        # the amount sold minus the cash received for the sale.
        remaining_capital -= (row['SellAmt'] - fee_manager.get_cash_amt_for_sell_size(
                                                     row['SellAmt'], ticker) if row['SellAmt'] > 0 else 0)
        # Subtract the magnitude of the buy transaction (the fees paid to resolve it are included in this value)
        # because that is a capital allocation decision that reduces remaining_capital.
        remaining_capital -= row['BuyAmt']
        # Subtract the amount you are holding because that is a capital allocation
        # decision that reduces remaining_capital.
        remaining_capital -= row['HoldAmt']

    cash_kept = min_cash_to_keep
    # min_cash_to_keep is the amount of cash that needs to be kept no matter what. This cash was
    # generated in the "resolve_protectorrails_and_taxes" function and it cannot be reallocated
    # so it is effectively using up some amount of capital.
    remaining_capital -= min_cash_to_keep

    ###################################
    # STEP 3 - Optimization Loop
    ###################################
    currently_available_tx_df = currently_available_tx_df.sort_values(by='MonthlyROI', ascending=False)
    # The following loop runs until all of your capital has been properly allocated. For each run of the
    # loop, the optimizer picks one transaction to resolve (in the case of buys) or one transaction to undo
    # (in the case of sells). The optimizer always makes transactions in order of descending ROI (high ROI
    # txs first, then low ROI ones).
    while remaining_capital > 0:
        tx_found = False  # Tracks whether a valid transaction was found in the following loop
        # Loop through every transaction in descending ROI, and make transactions
        # in order of descending quality.
        for index, data in currently_available_tx_df.iterrows():
            if verbose:
                print("Remaining Capital: %.0f" % remaining_capital)
            # "is_excess" is not currently used by the optimizer. But, because the concept of "excess"
            # holdings is an integral part of the optimizer, this variable may be necessary in the future.
            ticker, is_excess = get_ticker_from_index(index)

            # If the optimizer has determined that the next best holding is cash, put all the remaining capital
            # in cash. The main while loop will then exit as there will now be no more remaining capital.
            if ticker == "Cash":
                cash_kept += remaining_capital
                if verbose:
                    print("The remaining $%.0f will be put into cash" % cash_kept)
                # Put remaining capital into cash if it is the best choice
                remaining_capital = 0
                tx_found = True
            else:
                # target_tx_magnitude: The size of the transaction that is being considered without considering
                #       any constraints (remaining capital and the total capital invested in a given sector)
                # final_tx_magnitude: The maximum size of the transaction after all constraints have been
                #       considered; changes throughout the loop as new constraints are found; for buys,
                #       it is the total amount of capital being used to buy the stock and for sells it is
                #       the value of the stock being sold (before computing fees)
                # block_size: The number that has to be subtracted from the remaining capital; for sells,
                #       the fees you have to pay are subtracted from this value because you are undoing a
                #       sell transaction (thereby getting back the money used up to pay fees back when this sell
                #       transaction was automatically resolved)

                # ==================================================================================================
                # STEP A: Determine the maximum size of the transaction - this is limited by the following factors:
                # the target size of the transaction, the amount of capital allowed in the stock's sector, and the
                # amount of remaining capital. The size of the transaction also can't cause the transaction to
                # leave a small portion of a stock in holdings.
                # ==================================================================================================

                # final_tx_magnitude starts as the target magnitude of the transaction but then takes into
                # consideration other limitations. final_tx_magnitude doesn't include fess paid because it isn't
                # necessary when checking if the transaction is large enough (size of transaction is determined by
                # the amount of capital being used up).
                final_tx_magnitude = max(currently_available_tx_df.at[index, "BuyTarget"],
                                         currently_available_tx_df.at[index, "SellTarget"])
                limiting_factor = "target size of the transaction"

                # The magnitude of the final transaction is limited by 3 factors: the magnitude of
                # the transaction size, remaining capital, and remaining capital in the stock's sector
                # The sector_dict only tracks how much capital is in each sector, and this
                # doesn't include the fees that are paid to get to that value
                tx_magnitude_after_fees = fee_manager.get_stock_amt_for_buy_size(final_tx_magnitude, ticker) \
                      if data['TxType'] == "Buy" else final_tx_magnitude
                if data['SectorId'] is not None and remaining_sector_capital_dict[data['SectorId']] < tx_magnitude_after_fees:
                    # For buys, the size of the transaction needs to include the fees required to
                    # reach a certain sized position in a stock
                    if data['TxType'] == "Buy":
                        tx_magnitude_after_fees = remaining_sector_capital_dict[data['SectorId']]
                        final_tx_magnitude = fee_manager.get_buy_size_for_required_stock_amt(
                            remaining_sector_capital_dict[data['SectorId']], ticker)
                    # For sells, you are undoing a transaction so fees don't need to be added into the transaction size
                    else:
                        tx_magnitude_after_fees = remaining_sector_capital_dict[data['SectorId']]
                        final_tx_magnitude = remaining_sector_capital_dict[data['SectorId']]
                    limiting_factor = "amount of capital allowed in sector %s" % data['SectorId']

                # When looking at this code remember that "sell" transactions are completely earmarked
                # earlier (so that in this loop, the binpacker only undoes sells where the sell doesn't make sense).
                # This means that if a sell transaction has the highest ROI, the remaining_capital will be used
                # to undo the sale that was earmarked before this loop. This represents the case where a
                # stock was supposed to be sold, but no other buys could be done (often because they were too
                # small to transact on individually), which then means undoing some of the planned sale
                # is the best way to use up the capital (versus the alternative of keeping it in cash).
                if remaining_capital < final_tx_magnitude:
                    if data['TxType'] == "Sell" or data['TxType'] == "Excess Sell":
                        # real_remaining_capital is calculated using the sum of an infinite geometric series. You
                        # first undo remaining_capital worth of sells, but this frees up more capital which used
                        # to be going into fees (with a value of remaining_capital * trade_bid_ask_spread). Once
                        # you sell a bit more, you free up more capital and so on, so forth. So the infinite series
                        # takes the form (rc) + (rc * tbas) + (rc * tbas^2) + ... + (rc * tbas^infinity) where rc
                        # is remaining_capital and tbas is trade_bid_ask_spread. The sum of this infinite series
                        # is equal to the remaining_capital/(1 - trade_bid_ask_spread).
                        real_remaining_capital = remaining_capital/(1 - fee_manager.get_spread(ticker))
                        # If it is possible to complete the whole sell transaction for the stock, this would
                        # free up some more capital in the form of "unpaying" the trading_commission
                        if (final_tx_dataframe.at[ticker, 'SellAmt'] == final_tx_magnitude and
                                real_remaining_capital + fee_manager.trading_commission >= final_tx_magnitude):
                            # Capital is no longer the limiting factor
                            pass
                        # Otherwise, remaining capital might become the main limiting factor
                        elif real_remaining_capital < final_tx_magnitude:
                            tx_magnitude_after_fees = real_remaining_capital
                            final_tx_magnitude = real_remaining_capital
                            limiting_factor = "amount of remaining capital"
                    # For buys, remaining capital is already an accurate value, so it
                    # doesn't need to be adjusted as with sells
                    else:
                        tx_magnitude_after_fees = fee_manager.get_stock_amt_for_buy_size(remaining_capital, ticker)
                        final_tx_magnitude = remaining_capital
                        limiting_factor = "amount of remaining capital"

                # Always ensure that the size of the sell transaction that isn't being undone is larger than
                # the min_tx_magnitude - this ensures that the min_tx_magnitude limit is not violated
                if (data['TxType'] == "Sell" or data['TxType'] == "Excess Sell") and \
                        (0 < data['SellTarget'] - final_tx_magnitude <= min_tx_magnitude):
                    tx_magnitude_after_fees = data['SellTarget'] - min_tx_magnitude
                    final_tx_magnitude = data['SellTarget'] - min_tx_magnitude
                    limiting_factor = "size of the sell transaction that isn't being undone " \
                                      "(must be greater than or equal to %.0f)" % min_tx_magnitude

                # A stock can only be transacted on if the transaction is large enough. A transaction is considered
                # large enough if the value of the stock bought is larger than the min_tx_magnitude (generally about
                # 1% of total capital). This limitation ensures that the $10 trading commission is no longer very
                # relevant when optimizing the transactions. This limitation is unnecessary when undoing sell
                # transactions.
                if tx_magnitude_after_fees > min_tx_magnitude or \
                        (tx_magnitude_after_fees > 0 and (data['TxType'] == "Sell" or data['TxType'] == "Excess Sell")):
                    if verbose:
                        if tx_magnitude_after_fees > min_tx_magnitude:
                            print("%s transaction for %s is large enough to be transacted on - "
                                  "tx size of %.0f (limited by the %s) is larger than the minimum of %.0f"
                                  % (data['TxType'], ticker, tx_magnitude_after_fees, limiting_factor, min_tx_magnitude))
                        else:
                            assert (tx_magnitude_after_fees > 0 and (
                                    data['TxType'] == "Sell" or data['TxType'] == "Excess Sell"))
                            print("%s transaction for %s is a sell-undo and so can be transacted on - "
                                  "tx size of %.0f (limited by the %s) is larger than the minimum of 0"
                                  % (data['TxType'], ticker, tx_magnitude_after_fees, limiting_factor))

                    # ==============================================================================================
                    # STEP B: If the transaction is large enough to be transacted on, apply all the effects of that
                    # transaction. This includes paying fees, updating remaining capital, adding the transaction to
                    # the final transactions list, and updating the available transactions dataframe.
                    # ==============================================================================================

                    block_size = final_tx_magnitude  # Tracks block size including fees
                    if data['TxType'] == "Sell" or data['TxType'] == "Excess Sell":
                        # If it is a sell transaction, then decrease the size of the block by the trade bid
                        # ask spread that need to be paid. This needs to be done because the size of the
                        # capital bucket was previously reduced by these fees. Because putting in a sell
                        # block means the sell transaction has been ignored, it is necessary to compensate
                        # for the reduction in capital.
                        block_size *= (1 - fee_manager.get_spread(ticker))

                        final_tx_dataframe.at[ticker, 'SellAmt'] -= final_tx_magnitude
                        final_tx_dataframe.at[ticker, 'HoldAmt'] += final_tx_magnitude
                        # Undoing a sell transaction doesn't require paying fees
                        real_fees_paid = 0

                        # SellAmt should technically never go below zero, but because of rounding error and sometimes
                        # a very slight error in computing fees, SellAmt will sometimes be below zero by less than
                        # 1. If it is slightly negative, it will be considered equal to 0.
                        if -1 <= final_tx_dataframe.at[ticker, 'SellAmt'] <= 0:
                            final_tx_dataframe.at[ticker, 'SellAmt'] = 0
                            # Similar to reducing block size by the trade bid ask spread. Can only undo
                            # the payment of the trading commission if the whole transaction is complete.
                            block_size -= fee_manager.trading_commission

                    else:
                        # block_size is not adjusted as for sells because buys are actually being
                        # executed, not retroactively undone as was the case for sells
                        final_tx_dataframe.at[ticker, 'BuyAmt'] += final_tx_magnitude
                        real_fees_paid = final_tx_magnitude - fee_manager.get_stock_amt_for_buy_size(
                            final_tx_magnitude, ticker)

                    # Use adjusted block size when reducing remaining capital to "undo" the
                    # payment of fees that no longer need to be paid
                    remaining_capital -= block_size
                    if data['SectorId'] is not None:
                        # Use real transaction magnitude value when reducing the amount of capital remaining in each
                        # sector because this value is unrelated to the adjusted capital value (based of real capital)
                        # Don't include fees when subtracting from remaining_sector_capital_dict because the its purpose
                        # is to track how much capital is invested in a given sector (fees aren't invested into a sector)
                        remaining_sector_capital_dict[data['SectorId']] -= final_tx_magnitude - real_fees_paid

                    # Remove the row with this stock because its transaction has already been resolved
                    currently_available_tx_df = currently_available_tx_df.drop(index=index)
                    tx_found = True
                    break
                else:
                    if verbose:
                        # This block can only be reached if the transaction is too small and
                        # it's not a positive sell-undo transaction - either way, it is too small.
                        min_value = 0 if (data['TxType'] == "Sell" or data['TxType'] == "Excess Sell") else min_tx_magnitude
                        print("%s transaction for %s is too small to be transacted on - "
                              "tx size of %.0f (limited by the %s) is smaller than the minimum of %.0f"
                              % (data['TxType'], ticker, tx_magnitude_after_fees, limiting_factor, min_value))

                    # Remove the row with this stock because its transaction size is too small
                    currently_available_tx_df = currently_available_tx_df.drop(index=index)

        if not tx_found:
            if verbose:
                print("No transactions that are larger than the minimum transaction "
                      "size of %.0f so exiting the optimization loop" % min_tx_magnitude)
            break

    assert -1 <= remaining_capital <= 0, remaining_capital

    if verbose:
        # Make sure that the whole frame is printed out and no part of it is cut off
        print("\nFINAL TX DATAFRAME:")
        vu.print_entire_array_or_frame(final_tx_dataframe)

    final_txs = get_final_tx_df(final_tx_dataframe, target_investment_per_stock * excess_threshold_multiplier,
                                market_frame, fee_manager, cash_kept)

    return final_txs, cash_kept
#-------------------------------------------------------------------
def get_final_tx_df(final_tx_dataframe, excess_threshold, market_frame, fee_manager, cash_kept):
    # This function converts the tx_dataframe into a list that is readable by the portfolio_simulator
    final_txs = pd.DataFrame(
        columns=['Ticker', 'Sector_Id', 'Curr_Holdings', 'Target_Holdings', 'Projected_1m_ROI', 'Hold_Amt',
                 'Buy_Amt', 'Sell_Amt', 'Tx_Fees_Paid', 'Forced_Tx?', 'Tx_Notes'])
    final_txs = final_txs.set_index('Ticker')

    for ticker, row in final_tx_dataframe.iterrows():
        if row['CurrHeld'] > 0 or row['Target'] > 0:
            final_txs = pd.concat([final_txs, pd.DataFrame([[None] * final_txs.shape[1]],
                                                          columns=final_txs.columns, index=[ticker])])
        else:
            # For stocks in which you had no investment and still have no investment, don't list a transaction
            continue

        final_txs.at[ticker, 'Sector_Id'] = row['SectorId']
        final_txs.at[ticker, 'Curr_Holdings'] = row['CurrHeld']
        final_txs.at[ticker, 'Target_Holdings'] = row['Target']
        final_txs.at[ticker, 'Projected_1m_ROI'] = vu.get_expected_monthly_return(
            market_frame.at[ticker, 'Advanced_Rank'], 0) if ticker in market_frame.index.tolist() else None
        final_txs.at[ticker, 'Hold_Amt'] = row['HoldAmt']
        final_txs.at[ticker, 'Buy_Amt'] = row['BuyAmt']
        final_txs.at[ticker, 'Sell_Amt'] = row['SellAmt']
        final_txs.at[ticker, 'Forced_Tx?'] = row['Forced Tx?']

        if final_txs.at[ticker, 'Buy_Amt'] > 0:
            final_txs.at[ticker, 'Tx_Fees_Paid'] = final_txs.at[ticker, 'Buy_Amt'] - fee_manager.\
                get_stock_amt_for_buy_size(final_txs.at[ticker, 'Buy_Amt'], ticker)
        elif final_txs.at[ticker, 'Sell_Amt'] > 0:
            final_txs.at[ticker, 'Tx_Fees_Paid'] = final_txs.at[ticker, 'Sell_Amt'] - fee_manager.\
                get_cash_amt_for_sell_size(final_txs.at[ticker, 'Sell_Amt'], ticker)
        else:
            final_txs.at[ticker, 'Tx_Fees_Paid'] = 0

        # TODO - update the 'Tx_Notes' column to include more info in a more organized way
        # Assign a reason that explains why the transaction is useful and explains the
        # reason for the gap between the real transaction and the target transaction
        if row['Reason'] != "":
            final_txs.at[ticker, 'Tx_Notes'] = row['Reason']
        elif row['HoldAmt'] == 0 and 0 < row['BuyAmt'] < row['BuyTarget']:
            final_txs.at[ticker, 'Tx_Notes'] = "Partial Initiate"
        elif row['HoldAmt'] > 0 and 0 < row['BuyAmt'] < row['BuyTarget']:
            final_txs.at[ticker, 'Tx_Notes'] = "Partial Increase"
        elif row['HoldAmt'] > 0 and 0 < row['SellAmt'] < row['SellTarget']:
            final_txs.at[ticker, 'Tx_Notes'] = "Partial Decrease"
        elif row['HoldAmt'] == 0 and 0 < row['SellAmt'] < row['SellTarget']:
            final_txs.at[ticker, 'Tx_Notes'] = "Partial Liquidate"
        elif excess_threshold < row['HoldAmt'] < row['CurrHeld']:
            final_txs.at[ticker, 'Tx_Notes'] = "Partial Rebalance - Position in stock wss too large"
        elif row['HoldAmt'] == 0 and row['BuyAmt'] > 0 and row['BuyAmt'] == row['BuyTarget']:
            final_txs.at[ticker, 'Tx_Notes'] = "Full Initiate"
        elif row['HoldAmt'] > 0 and row['BuyAmt'] > 0 and row['BuyAmt'] == row['BuyTarget']:
            final_txs.at[ticker, 'Tx_Notes'] = "Full Increase"
        elif row['HoldAmt'] > 0 and row['SellAmt'] > 0 and row['SellAmt'] == row['SellTarget']:
            final_txs.at[ticker, 'Tx_Notes'] = "Full Decrease"
        elif row['HoldAmt'] == 0 and row['SellAmt'] > 0 and row['SellAmt'] == row['SellTarget']:
            final_txs.at[ticker, 'Tx_Notes'] = "Full Liquidate"
        elif row['HoldAmt'] == excess_threshold and row['CurrHeld'] > excess_threshold:
            final_txs.at[ticker, 'Tx_Notes'] = "Full Rebalance - Position in stock was too large"
        elif row['HoldAmt'] > 0 and row['BuyAmt'] == 0 and row['SellAmt'] == 0:
            final_txs.at[ticker, 'Tx_Notes'] = "Hold"
        elif row['HoldAmt'] == 0 and row['BuyAmt'] == 0 and row['SellAmt'] == 0:
            final_txs.at[ticker, 'Tx_Notes'] = "Hold - Stock is in the target list but " \
                                               "do not make any transactions on it"
        else:
            print (row)
            assert False, "Should never happen!"

    return final_txs
#-------------------------------------------------------------------
def get_ticker_from_index(index):
    # This function returns a ticker given an index as well as whether or not a holding is excess
    if index[0:len(EXCESS_IDENTIFIER)] == EXCESS_IDENTIFIER:
        # THe index includes "EXCESS_" at the beginning of the ticker if it is an
        # excess holding, so remove this extra bit to get the actual ticker
        ticker = index[len(EXCESS_IDENTIFIER):]
        return ticker, True
    else:
        ticker = index
        return ticker, False
