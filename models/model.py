import jax
import jax.numpy as jnp
from jax.scipy.optimize import minimize
import optax
NUMBER_OF_PARAMS = 30

def f(params, t): # Function that we want to optimize
    A, w = params[:NUMBER_OF_PARAMS], params[NUMBER_OF_PARAMS:]
    w = w[:, None] # Add a dimension for broadcasting
    return jnp.sum(A[:, None] * jnp.sin(w * t), axis=0)

def loss(params, t, target):
    prediction = f(params, t)
    return jnp.mean((prediction - target) ** 2)


params = jnp.array([1.0] * NUMBER_OF_PARAMS * 2)
t = jnp.linspace(0, 10, 100)
target = jnp.sin(t)

# Compute the gradient of the loss function
grad_loss = jax.grad(loss)
print(grad_loss(params, t, target))


# ------------------------------
# Optimization example using jax.scipy.optimize
# Easy but unstable (with lots of parameters it doesn't work well)
# ------------------------------

for pred, targ in zip(f(params, t), target):
    print(pred, targ, "not optimized")

result = minimize(loss, params, args=(t, target), method='BFGS')
optimzed_params = result.x

for pred, targ in zip(f(optimzed_params, t), target):
    print(pred, targ, "optimized")


# ------------------------------
# Optimization example using optax
# More stable and efficient
# ------------------------------


# Set up the optimizer
optimizer = optax.adam(0.01)  # Using Adam with a learning rate of 0.01
opt_state = optimizer.init(params)

# Define a single optimization step
@jax.jit # jit: compiles the function for faster execution
def step(params, opt_state, t, target):
    grads = jax.grad(loss)(params, t, target)
    updates, opt_state = optimizer.update(grads, opt_state)
    new_params = optax.apply_updates(params, updates)
    return new_params, opt_state

# Run the optimization loop
num_steps = 1000
for step_num in range(num_steps):
    params, opt_state = step(params, opt_state, t, target)
    if step_num % 100 == 0:
        current_loss = loss(params, t, target)
        print(f"Step {step_num}, Loss: {current_loss}")

# After optimization
optimized_params = params

# Comparing before and after optimization
print("Before optimization:")
for pred, targ in zip(f(jnp.array([1.0] * NUMBER_OF_PARAMS * 2), t), target):
    print(f"Prediction: {pred}, Target: {targ}")

print("After optimization:")
for pred, targ in zip(f(optimized_params, t), target):
    print(f"Prediction: {pred}, Target: {targ}")

print(f'Optimized parameters: {optimized_params}')