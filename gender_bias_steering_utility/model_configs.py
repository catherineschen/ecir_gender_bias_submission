MODEL_CONFIGS = {
    "sentence-transformers/msmarco-distilbert-dot-v5": {
        "pool": "mean",
        "sim_func": "dot",
        "heads": [(5, 3), (5, 11)],
        "last_layer": 5,
        "n_heads": 12,
    },
    "sebastian-hofstaetter/distilbert-dot-tas_b-b256-msmarco": {
        "pool": "cls",
        "sim_func": "dot",
        "heads": [(5, 3), (5, 11)],
        "last_layer": 5,
        "n_heads": 12,
    },
    "sentence-transformers/multi-qa-distilbert-dot-v1": {
        "pool": "cls",
        "sim_func": "dot",
        "heads": [(5, 3), (5, 11)],
        "last_layer": 5,
        "n_heads": 12,
    },
    "sentence-transformers/multi-qa-MiniLM-L6-dot-v1": {
        "pool": "cls",
        "sim_func": "dot",
        "heads": [(5, 1), (5, 4), (5, 6), (5, 11)],
        "last_layer": 5,
        "n_heads": 12,
    },
    "sentence-transformers/msmarco-bert-base-dot-v5": {
        "pool": "mean",
        "sim_func": "dot",
        "heads": [(11, 3), (11, 11)],
        "last_layer": 11,
        "n_heads": 12,
    },
}