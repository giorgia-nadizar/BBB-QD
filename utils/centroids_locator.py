import plotly.express as px
import jax.numpy as jnp

base_path = f"../results/evo-body-5x5-me-s3_0"
centroids = jnp.load(f"{base_path}/r3_centroids.npy")
fitnesses = jnp.load(f"{base_path}/r3_fitnesses.npy")
fit_mask = fitnesses > -jnp.inf
non_empty_centroids = centroids[fit_mask]
hover_data = jnp.arange(len(centroids)) * fit_mask
print(len(centroids[:, 0]))
print(len(hover_data))

fig = px.scatter(x=centroids[:, 0], y=centroids[:, 1], hover_data=[hover_data], color=fitnesses)
fig.show()
