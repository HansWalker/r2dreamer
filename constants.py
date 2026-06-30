MAMBA_CACHE_KEYS = (
    "mamba_angle_state",
    "mamba_ssm_state",
    "mamba_k_state",
    "mamba_v_state",
)

STORM_CACHE_PREFIX = "storm_cache_"
STORM_POSITION_KEY = "storm_cache_position"


def storm_cache_keys(count):
    return (STORM_POSITION_KEY,) + tuple(f"{STORM_CACHE_PREFIX}{idx}" for idx in range(int(count)))


def replay_cache_keys(keys):
    keys = set(keys)
    if all(key in keys for key in MAMBA_CACHE_KEYS):
        return MAMBA_CACHE_KEYS
    storm_keys = [key for key in keys if key.startswith(STORM_CACHE_PREFIX) and key != STORM_POSITION_KEY]
    if storm_keys and STORM_POSITION_KEY in keys:
        return (STORM_POSITION_KEY,) + tuple(
            sorted(storm_keys, key=lambda key: int(key.removeprefix(STORM_CACHE_PREFIX)))
        )
    return ()
