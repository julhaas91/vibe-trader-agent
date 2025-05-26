import time
import os
import datetime
import numpy as np
import pandas as pd
import yfinance as yf
import logging
import matplotlib.pyplot as plt
from scipy.optimize import differential_evolution
from pypfopt import expected_returns, risk_models, black_litterman
import json

class PortfolioOptimizer:
    """
    A class for optimizing portfolio allocation using Black-Litterman model and Monte Carlo simulation.
    """
    
    def __init__(self, 
                 tickers=["SPY", "QQQ", "TLT", "GLD", "BIL"],
                 horizon_years=10,
                 start_portfolio=100_000,
                 target_portfolio=200_000,
                 max_drawdown=0.25,
                 worst_day_limit=0.10,
                 sigma_max=0.20,
                 cash_min=0.05,
                 upper_bounds=0.5,
                 cash_ticker="BIL",
                 scenarios=10_000,
                 mc_freq=12,
                 lambda_sigma=100,
                 lambda_drawdown=500,
                 lambda_worst_day=300,
                 lambda_cash=1000,
                 output_dir=None,
                 log_prefix="optimization",
                 max_iterations=100,
                 # Black-Litterman view parameters
                 bl_view_matrix_P=None,
                 bl_view_vector_Q=None,
                 bl_view_uncertainty_omega=None):
        """
        Initialize the portfolio optimizer with configuration parameters.
        
        Args:
            tickers (list): List of ticker symbols
            horizon_years (int): Investment horizon in years
            start_portfolio (float): Initial portfolio value
            target_portfolio (float): Target portfolio value
            max_drawdown (float): Maximum allowed drawdown
            worst_day_limit (float): Maximum allowed single-day drop
            sigma_max (float): Maximum allowed annual volatility
            cash_min (float): Minimum cash allocation
            upper_bounds (float): Maximum weight per asset
            cash_ticker (str): Ticker symbol for cash equivalent
            scenarios (int): Number of Monte Carlo scenarios
            mc_freq (int): Rebalancing frequency per year
            lambda_sigma (float): Penalty strength for volatility constraint
            lambda_drawdown (float): Penalty strength for drawdown constraint
            lambda_worst_day (float): Penalty strength for worst-day constraint
            lambda_cash (float): Penalty strength for cash constraint
            output_dir (str): Directory to save results (if None, will be created)
            log_prefix (str): Prefix for log files and output directory
            max_iterations (int): Maximum number of iterations for optimization
            bl_view_matrix_P (np.ndarray): View matrix P in Black-Litterman model
            bl_view_vector_Q (np.ndarray): View vector Q in Black-Litterman model
            bl_view_uncertainty_omega (np.ndarray): View uncertainty matrix Omega
        """
        # Portfolio configuration
        self.TICKERS = tickers
        self.HORIZON_YEARS = horizon_years
        self.START_PORTFOLIO = start_portfolio
        self.TARGET_PORTFOLIO = target_portfolio
        self.CASH_TICKER = cash_ticker
        
        # Risk constraints
        self.MAX_DRAWDOWN = max_drawdown
        self.WORST_DAY_LIMIT = worst_day_limit
        self.SIGMA_MAX = sigma_max
        self.CASH_MIN = cash_min
        self.UPPER_BOUNDS = upper_bounds
        
        # Simulation parameters
        self.SCENARIOS = scenarios
        self.MC_FREQ = mc_freq
        
        # Penalty weights
        self.LAMBDA_SIGMA = lambda_sigma
        self.LAMBDA_DRAWDOWN = lambda_drawdown
        self.LAMBDA_WORST_DAY = lambda_worst_day
        self.LAMBDA_CASH = lambda_cash
        
        # Optimization tracking
        self.history = {'generation': [], 'objective': []}
        self.iteration = 0
        
        # Setup output directory and logging
        self.output_dir = output_dir
        self.log_prefix = log_prefix
        self._setup_logging()
        
        # Black-Litterman view parameters
        self.bl_view_matrix_P = bl_view_matrix_P
        self.bl_view_vector_Q = bl_view_vector_Q
        self.bl_view_uncertainty_omega = bl_view_uncertainty_omega
        
        # Will be set during optimization
        self.posterior_mu = None
        self.posterior_sigma = None
        
        # Add this to the initialization
        self.MAX_ITERATIONS = max_iterations

    def _setup_logging(self):
        """Setup logging configuration."""
        if self.output_dir is None:
            timestamp = time.strftime('%Y%m%d_%H%M%S')
            self.output_dir = os.path.join(
                os.path.dirname(__file__),
                f"{self.log_prefix}_{timestamp}_T{int(self.TARGET_PORTFOLIO)}_DD{int(self.MAX_DRAWDOWN*100)}"
                f"_WD{int(self.WORST_DAY_LIMIT*100)}_VOL{int(self.SIGMA_MAX*100)}"
                f"_CASH{int(self.CASH_MIN*100)}"
            )
        os.makedirs(self.output_dir, exist_ok=True)

        # Always use log_prefix for the log file name
        log_file = os.path.join(self.output_dir, f"{self.log_prefix}.log")

        # Configure root logger
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s %(levelname)s: %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )

        # Create file handler
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(logging.Formatter('%(asctime)s %(levelname)s: %(message)s'))

        # Create console handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(logging.Formatter('%(asctime)s %(levelname)s: %(message)s'))

        # Setup logger
        self.logger = logging.getLogger(f"{self.log_prefix}_{id(self)}")
        self.logger.setLevel(logging.INFO)
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)

        self.logger.info("=== Starting new optimization run ===")
        self.logger.info(f"Output directory: {self.output_dir}")
        self.logger.info(f"Configuration: T={self.TARGET_PORTFOLIO}, DD={self.MAX_DRAWDOWN}, "
                        f"WD={self.WORST_DAY_LIMIT}, VOL={self.SIGMA_MAX}, CASH={self.CASH_MIN}")

    def _simulate_portfolio_paths(self, w, mu, Sigma, V0, T_years, freq, n_sims):
        """Simulate portfolio paths using Monte Carlo method."""
        dt = 1 / freq
        n_steps = int(T_years * freq)
        L = np.linalg.cholesky(Sigma * dt)
        rnd = np.random.randn(n_sims, n_steps, len(w))
        shocks = rnd @ L.T
        ret = mu.values * dt + shocks
        V = np.full((n_sims, n_steps + 1), V0)
        
        for t in range(n_steps):
            V[:, t+1] = V[:, t] * (1 + (ret[:, t] @ w))
            
        return V

    def _calculate_metrics(self, V):
        """Calculate portfolio metrics from simulated paths."""
        prob = np.mean(V[:, -1] >= self.TARGET_PORTFOLIO)
        peaks = np.maximum.accumulate(V, axis=1)
        drawdowns = (peaks - V) / peaks
        avg_dd = np.mean(np.max(drawdowns, axis=1))
        daily_ret = V[:, 1:] / V[:, :-1] - 1
        worst = -np.min(daily_ret, axis=1)
        avg_wd = np.mean(worst)
        avg_final = np.mean(V[:, -1])
        return prob, avg_dd, avg_wd, avg_final, V[:, -1]

    def simulate_metrics(self, w, mu, Sigma, V0, T_years, freq, n_sims):
        """Simulate portfolio and calculate metrics."""
        V = self._simulate_portfolio_paths(w, mu, Sigma, V0, T_years, freq, n_sims)
        return self._calculate_metrics(V)

    def _calculate_penalties(self, w, Sigma, dd, wd):
        """Calculate all penalty terms."""
        vol = np.sqrt(w.T @ Sigma.values @ w)
        sigma_penalty = self.LAMBDA_SIGMA * max(0, vol - self.SIGMA_MAX)**2
        drawdown_penalty = self.LAMBDA_DRAWDOWN * max(0, dd - self.MAX_DRAWDOWN)**2
        worst_day_penalty = self.LAMBDA_WORST_DAY * max(0, wd - self.WORST_DAY_LIMIT)**2
        cash_penalty = self.LAMBDA_CASH * max(0, self.CASH_MIN - w[self.TICKERS.index(self.CASH_TICKER)])**2
        return sigma_penalty, drawdown_penalty, worst_day_penalty, cash_penalty

    def objective(self, x, posterior_mu, posterior_sigma):
        """Objective function for optimization."""
        x = np.clip(x, 0, 1)
        w = x / x.sum()
        
        # Calculate metrics
        p, dd, wd, avg_final, _ = self.simulate_metrics(
            w, posterior_mu, posterior_sigma,
            self.START_PORTFOLIO, self.HORIZON_YEARS, self.MC_FREQ, self.SCENARIOS
        )
        
        # Calculate penalties
        pen_s, pen_d, pen_w, pen_c = self._calculate_penalties(w, posterior_sigma, dd, wd)
        
        # Calculate objective
        obj = -p + pen_s + pen_d + pen_w + pen_c
        
        # Log results
        self.logger.info(
            f"[Obj] w={w}, P={p:.4f}, DD={dd:.4f}, WD={wd:.4f}, Final={avg_final:.2f}, "
            f"pen_s={pen_s:.2f}, pen_d={pen_d:.2f}, pen_w={pen_w:.2f}, pen_c={pen_c:.2f}, obj={obj:.4f}"
        )
        return obj

    def de_callback(self, xk, convergence):
        """Callback function for differential evolution."""
        self.iteration += 1
        w = np.clip(xk, 0, 1)
        w /= w.sum()
        obj = self.objective(xk, self.posterior_mu, self.posterior_sigma)
        self.history['generation'].append(self.iteration)
        self.history['objective'].append(obj)
        self.logger.info(f"[Callback] Gen={self.iteration}, obj={obj:.4f}, w={w}")
        return False

    def _setup_black_litterman(self, prices):
        """Setup Black-Litterman model parameters."""
        # Calculate historical parameters
        mu_hist = expected_returns.mean_historical_return(prices)
        sigma_hist = risk_models.sample_cov(prices)
        caps = pd.Series({t: yf.Ticker(t).info.get("marketCap", np.nan) for t in self.TICKERS})
        cap_wts = caps.fillna(caps.median()) / caps.sum()
        delta = black_litterman.market_implied_risk_aversion(prices)
        prior_mu = black_litterman.market_implied_prior_returns(cap_wts, delta, sigma_hist)
        
        # Setup views (empty if not provided)
        if self.bl_view_matrix_P is None or self.bl_view_vector_Q is None:
            self.bl_view_matrix_P = np.zeros((0, len(self.TICKERS)))
            self.bl_view_vector_Q = np.array([])
        
        # Setup view uncertainty (empty if not provided)
        if self.bl_view_uncertainty_omega is None:
            self.bl_view_uncertainty_omega = np.diag(np.array([]))
        elif len(self.bl_view_uncertainty_omega.shape) == 1:
            # Convert to diagonal matrix if provided as vector
            self.bl_view_uncertainty_omega = np.diag(self.bl_view_uncertainty_omega)
        
        # Validate dimensions
        n_assets = len(self.TICKERS)
        n_views = self.bl_view_matrix_P.shape[0]
        
        if self.bl_view_matrix_P.shape[1] != n_assets:
            raise ValueError(f"View matrix P must have {n_assets} columns")
        if self.bl_view_vector_Q.shape[0] != n_views:
            raise ValueError(f"View vector Q must have length {n_views}")
        if self.bl_view_uncertainty_omega.shape != (n_views, n_views):
            raise ValueError(f"View uncertainty matrix Omega must be {n_views}x{n_views}")
        
        # Create BL model
        bl = black_litterman.BlackLittermanModel(
            cov_matrix=sigma_hist,
            pi=prior_mu,
            P=self.bl_view_matrix_P,
            Q=self.bl_view_vector_Q,
            omega=self.bl_view_uncertainty_omega
        )
        
        return bl.bl_returns(), bl.bl_cov()

    def optimize(self, run_id=None, save_outputs=True):
        """
        Run the portfolio optimization process.
        
        Args:
            run_id (str): Unique identifier for this run (if None, will be generated)
            save_outputs (bool): Whether to save outputs to files (default: True)
            
        Returns:
            dict: Optimization results including weights and metrics
        """
        if run_id is None:
            run_id = time.strftime('%Y%m%d_%H%M%S')
            
        # 1) Data & BL Prior
        prices = yf.download(self.TICKERS, start="2010-01-01", auto_adjust=True)["Close"].dropna()
        self.posterior_mu, self.posterior_sigma = self._setup_black_litterman(prices)

        # 2) Optimization
        bounds = [(0, self.UPPER_BOUNDS) for _ in self.TICKERS]
        t0 = time.perf_counter()
        result = differential_evolution(
            self.objective, bounds,
            args=(self.posterior_mu, self.posterior_sigma),
            popsize=15, maxiter=self.MAX_ITERATIONS,
            workers=-1, polish=True,
            disp=False, callback=self.de_callback
        )
        elapsed = time.perf_counter() - t0
        self.logger.info(f"Optimization completed in {elapsed:.2f}s")

        # 3) Final Metrics
        w_opt = np.clip(result.x, 0, 1)
        w_opt /= w_opt.sum()
        p_opt, dd_opt, wd_opt, avg_final, finals = self.simulate_metrics(
            w_opt, self.posterior_mu, self.posterior_sigma,
            self.START_PORTFOLIO, self.HORIZON_YEARS, self.MC_FREQ, self.SCENARIOS
        )
        vol_opt = np.sqrt(w_opt.T @ self.posterior_sigma.values @ w_opt)
        cash_alloc = w_opt[self.TICKERS.index(self.CASH_TICKER)]

        # 4) Log & Print Summary
        self._log_results(run_id, elapsed, p_opt, avg_final, dd_opt, wd_opt, vol_opt, cash_alloc)

        # 5) Create visualizations if save_outputs is True
        if save_outputs:
            self._create_visualization(run_id, w_opt, finals, prices)

        # 6) Prepare results dictionary
        results_dict = {
            'inputs': {
                'tickers': self.TICKERS,
                'horizon_years': self.HORIZON_YEARS,
                'start_portfolio': self.START_PORTFOLIO,
                'target_portfolio': self.TARGET_PORTFOLIO,
                'max_drawdown': self.MAX_DRAWDOWN,
                'worst_day_limit': self.WORST_DAY_LIMIT,
                'sigma_max': self.SIGMA_MAX,
                'cash_min': self.CASH_MIN,
                'upper_bounds': self.UPPER_BOUNDS,
                'cash_ticker': self.CASH_TICKER,
                'scenarios': self.SCENARIOS,
                'mc_freq': self.MC_FREQ,
                'max_iterations': self.MAX_ITERATIONS,
                'log_prefix': self.log_prefix,
                'output_dir': self.output_dir,
                'bl_views': {
                    'P': self.bl_view_matrix_P.tolist() if self.bl_view_matrix_P is not None else None,
                    'Q': self.bl_view_vector_Q.tolist() if self.bl_view_vector_Q is not None else None,
                    'omega': self.bl_view_uncertainty_omega.tolist() if self.bl_view_uncertainty_omega is not None else None
                }
            },
            'results': {
                'weights': w_opt.tolist(),
                'success_prob': float(p_opt),
                'avg_final': float(avg_final),
                'avg_drawdown': float(dd_opt),
                'avg_worst_day': float(wd_opt),
                'volatility': float(vol_opt),
                'cash_allocation': float(cash_alloc),
                'elapsed_time': float(elapsed),
                'iterations': self.iteration
            }
        }

        # Save results as JSON if save_outputs is True
        if save_outputs:
            json_path = os.path.join(self.output_dir, f"{self.log_prefix}_results.json")
            with open(json_path, 'w') as f:
                json.dump(results_dict, f, indent=2)

        return results_dict

    def _create_visualization(self, run_id, w_opt, finals, prices):
        """Create all visualization plots."""
        # Objective plot
        plt.figure()
        plt.plot(self.history['generation'], self.history['objective'], marker='o')
        plt.title('Objective over Generations')
        plt.xlabel('Gen')
        plt.ylabel('Obj')
        plt.grid(True)
        plt.savefig(os.path.join(self.output_dir, f"{self.log_prefix}_objective_{run_id}.png"))
        plt.close()

        # Weights plot
        plt.figure()
        plt.bar(self.TICKERS, w_opt)
        plt.title('Optimal Weights')
        plt.ylabel('Weight')
        plt.savefig(os.path.join(self.output_dir, f"{self.log_prefix}_weights_{run_id}.png"))
        plt.close()

        # Distribution plot
        plt.figure()
        plt.hist(finals, bins=50)
        plt.axvline(self.TARGET_PORTFOLIO, color='red', linestyle='--')
        plt.title('Final Portfolio Distribution')
        plt.xlabel('Value')
        plt.ylabel('Freq')
        plt.savefig(os.path.join(self.output_dir, f"{self.log_prefix}_dist_{run_id}.png"))
        plt.close()

        # Historical evolution
        hist_returns = prices.pct_change().dropna()
        port_returns = hist_returns.dot(w_opt)
        V_hist = self.START_PORTFOLIO * (1 + port_returns).cumprod()
        plt.figure()
        plt.plot(V_hist.index, V_hist.values)
        plt.title('Historical Portfolio Value')
        plt.xlabel('Date')
        plt.ylabel('Portfolio Value')
        plt.savefig(os.path.join(self.output_dir, f"{self.log_prefix}_hist_evolution_{run_id}.png"))
        plt.close()

        # Monte Carlo paths
        n_paths = 50
        V_paths = self._simulate_portfolio_paths(
            w_opt, self.posterior_mu, self.posterior_sigma,
            self.START_PORTFOLIO, self.HORIZON_YEARS, self.MC_FREQ, n_paths
        )
        plt.figure()
        for i in range(n_paths):
            plt.plot(np.linspace(0, self.HORIZON_YEARS, V_paths.shape[1]), V_paths[i], alpha=0.5)
        plt.title('Monte Carlo Future Paths')
        plt.xlabel('Years')
        plt.ylabel('Portfolio Value')
        plt.savefig(os.path.join(self.output_dir, f"{self.log_prefix}_mc_paths_{run_id}.png"))
        plt.close()

    def _log_results(self, run_id, elapsed, p_opt, avg_final, dd_opt, wd_opt, vol_opt, cash_alloc):
        """Log and print optimization results."""
        summary = (
            f"Run {run_id} Summary:\n"
            f" Time: {elapsed:.2f}s\n"
            f" SuccessProb={p_opt:.4f}\n"
            f" AvgFinal={avg_final:.2f}\n"
            f" AvgDrawdown={dd_opt:.4f}\n"
            f" AvgWorstDay={wd_opt:.4f}\n"
            f" Volatility={vol_opt:.4f}\n"
            f" CashAlloc={cash_alloc:.4f}\n"
            f" Constraints: sigma≤{self.SIGMA_MAX} {'OK' if vol_opt<=self.SIGMA_MAX else 'VIOLATION'}, "
            f"drawdown≤{self.MAX_DRAWDOWN} {'OK' if dd_opt<=self.MAX_DRAWDOWN else 'VIOLATION'}, "
            f"worstDay≤{self.WORST_DAY_LIMIT} {'OK' if wd_opt<=self.WORST_DAY_LIMIT else 'VIOLATION'}, "
            f"cash≥{self.CASH_MIN} {'OK' if cash_alloc>=self.CASH_MIN else 'VIOLATION'}"
        )
        self.logger.info(summary)
        
        print("=== Final Results ===")
        print(f"Success probability:       {p_opt:.4f}")
        print(f"Average final value:       {avg_final:.2f}")
        print(f"Average drawdown:          {dd_opt:.4f}")
        print(f"Average worst-day drop:    {wd_opt:.4f}")
        print(f"Ex-ante volatility:        {vol_opt:.4f}")
        print(f"Cash allocation:           {cash_alloc:.4f}")
        print(f"Optimization time (s):     {elapsed:.2f}")
        print(f"Results directory:         {self.output_dir}\n") 
