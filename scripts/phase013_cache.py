import os, pickle
def cache_path(cache_dir, dataset_tag, config_name, K, subject):
    d = os.path.join(cache_dir, dataset_tag, f"{config_name}_K{K}")
    os.makedirs(d, exist_ok=True)
    safe = str(subject).replace(os.sep, "_").replace(":", "_")
    return os.path.join(d, f"{safe}.pkl")
def load(path):
    if os.path.exists(path):
        try:
            with open(path, "rb") as f: return pickle.load(f)
        except Exception: return None
    return None
def save(path, obj):
    tmp = path + ".tmp"
    with open(tmp, "wb") as f: pickle.dump(obj, f)
    os.replace(tmp, path)
