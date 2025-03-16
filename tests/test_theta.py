import numpy as np
import matplotlib.pyplot as plt

def target_log_density(theta, ua=10):
    """
    Computes the log density of the Exponential(ua) distribution.
    
    The exponential PDF is:
        f(theta) = ua * exp(-ua * theta)    for theta > 0
    Thus, log f(theta) = log(ua) - ua * theta.
    
    If theta is not positive, return -infinity.
    """
    if theta <= 0:
        return -np.inf
    return np.log(ua) - ua * theta

def propose_theta(theta, sigma=0.2):
    """
    Proposes a new theta using a log-random-walk:
    
      log(theta') = log(theta) + sigma * epsilon, where epsilon ~ N(0,1)
      => theta' = theta * exp(sigma * epsilon)
    
    Also computes the log ratio of the proposal densities (Jacobian correction),
    which for this transformation is: -epsilon * sigma - 0.5 * sigma**2.
    """
    epsilon = np.random.normal(0, 1)
    theta_prime = theta * np.exp(sigma * epsilon)
    log_q_ratio = np.log(1/theta)
    return theta_prime, log_q_ratio

def mcmc_sampling(n_iter=100000, theta_init=0.1, sigma=5, ua=10):
    """
    Runs a simple Metropolis-Hastings sampler for theta.
    
    Parameters:
      n_iter     : Number of MCMC iterations.
      theta_init : Initial value of theta.
      sigma      : Proposal scale for the log random-walk.
      ua         : Rate parameter for the target Exponential distribution.
      
    Returns:
      samples : A NumPy array of sampled theta values.
    """
    samples = np.zeros(n_iter)
    theta_current = theta_init
    current_log_target = target_log_density(theta_current, ua)
    accept_count = 0
    
    for i in range(n_iter):
        # Propose a new theta and get the proposal correction term
        theta_prime, log_q_ratio = propose_theta(theta_current, sigma)
        log_target_prime = target_log_density(theta_prime, ua)
        
        # Calculate the log acceptance ratio:
        # log_alpha = [log target(theta') - log target(theta)] + log(q(theta|theta')/q(theta'|theta))
        log_alpha = (log_target_prime - current_log_target) + log_q_ratio
        
        # Accept or reject the new proposal
        if np.log(np.random.rand()) < log_alpha:
            theta_current = theta_prime
            current_log_target = log_target_prime
            accept_count += 1
        
        samples[i] = theta_current
        
    print("Acceptance rate:", accept_count / n_iter)
    return samples

if __name__ == "__main__":
    # Run the MCMC sampler with a larger sigma
    samples = mcmc_sampling(n_iter=100000, theta_init=0.1, sigma=5, ua=10)
    
    # Optionally, plot a histogram of the samples.
    x = np.linspace(0, 0.5, 1000)
    # Theoretical density for Exp(ua): f(x) = ua * exp(-ua*x)
    pdf = 10 * np.exp(-10 * x)
    
    plt.figure(figsize=(8,6))
    plt.hist(samples, bins=50, density=True, alpha=0.5, label="MCMC samples")
    plt.plot(x, pdf, "r-", lw=2, label="Exponential(ua=10) PDF")
    plt.xlabel("theta")
    plt.ylabel("Density")
    plt.title("MCMC Samples for theta with Exponential(ua=10) Target")
    plt.legend()
    plt.show()