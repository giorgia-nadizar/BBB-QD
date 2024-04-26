import plotly.express as px
import jax.numpy as jnp

base_path = f"../results/me/evo-body-10x10-floor-all_0"
centroids = jnp.load(f"{base_path}/r3_centroids.npy")
fitnesses = jnp.load(f"{base_path}/r3_fitnesses.npy")

fig = px.scatter(x=centroids[:, 0], y=centroids[:, 1], color=fitnesses, text=jnp.arange(len(fitnesses)))
fig.show()
