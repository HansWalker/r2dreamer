MAMBA_CACHE_KEYS = (
    "mamba_angle_state",
    "mamba_ssm_state",
    "mamba_k_state",
    "mamba_v_state",
)


def replay_cache_keys(keys):
    keys = set(keys)
    if all(key in keys for key in MAMBA_CACHE_KEYS):
        return MAMBA_CACHE_KEYS
    return ()
