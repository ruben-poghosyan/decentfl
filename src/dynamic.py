# Simulations for Ranking Mechanism
import numpy as np
from matplotlib import pyplot as plt
np.random.seed(0)

# timesteps
T = 1000
#hyperparameters


def simulate(k):
    mu = 0.5
    N = 500
    N_mal = 0.5
    state = [0 for _ in range(int(N - N*N_mal))]
    state.extend([1 for _ in range(int(N*N_mal))])
    np.random.shuffle(state)
    alpha = np.random.normal(mu, 0.1, N)
    r = []
    for t in range(T):
        # get current state estimates
        if state.count(1) == 0:
            break
        r.append(state.count(0)/ state.count(1))
        # simulate predictions

        
        # Remove some bad nodes
        for _ in range(k):
            if state.count(1) == 0:
                break
            ind = state.index(1)
            state.pop(ind)
            alpha = np.delete(alpha, len(alpha)-ind-1, 0)
    
    # Calculate Next State using random walk
        temp = []
        for a in alpha:
            v = np.random.uniform(0, 1)
            if v <= a:
                temp.append(1)
            else:
                temp.append(0)
        # update state
        state = temp
        # Update alphas and betas
        temp = []
        for (ind,a) in enumerate(alpha):
            temp.append(a - np.random.normal(0.5, 1) + state.count(1)/(N*0.92))
        alpha = temp
    return r

stop = 30
x = np.arange(stop, step=1)

styles = ['solid', (0, (5, 1)), (0, (5, 5)), (0, (1, 1)), (5, (10, 3))]
for (i, k) in enumerate([1, 3, 5, 8, 10]):
    r = simulate(k)
    plt.plot(x, r[:stop], label=f'k={k}', linestyle=styles[i])

plt.hlines(y=1,xmin=x[0], xmax=x[-1], colors='red', linestyles='dashed')
plt.grid(visible=True, which='both')
plt.ylabel('r', fontsize=12, rotation=0)
plt.xlabel("Round")
plt.legend()
plt.savefig('k_diff.jpg', dpi=400, bbox_inches='tight')
    


    


