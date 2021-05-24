def print_metrics(metrics):
    print("-" * 100)
    maxlen_name_len = max(len(name) for name in metrics)
    for name, value in metrics.items():
        blankspace = " " * (maxlen_name_len - len(name) + 1)
        if name.startswith("loss_"):
            print(f"{name}: {blankspace}{value:>0.2f}")
        else:
            print(f"{name}: {blankspace}{value:>0.1f}")
    print("-" * 100)
