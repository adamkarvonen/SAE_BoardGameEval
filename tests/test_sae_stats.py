from circuits.sae_stats_collection import compute_all_ae_stats

compute_all_ae_stats("autoencoders/group0/", n_inputs=100, top_k=10, max_dims=100, batch_size=10)
