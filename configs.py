
def get_ssl_config():
    return {
        # SSL settings
        "num_train_samples_per_class": 25,
        "mu_ratio": 7,
        "ema_decay": 0.999,
    }


def get_backend_config():
    return {
        "device": "cuda",  # possible values "cuda" or "xla"
        "distributed": False,

        # AMP
        "with_nv_amp_level": None,  # if "O1" or "O2" -> train with apex/amp, otherwise fp32 (None)
    }


def get_default_config():
    batch_size = 64

    config = {
        "seed": 12,
        "data_path": "/tmp/cifar10",
        "output_path": "/tmp/output-fixmatch-cifar10",
        "model": "WRN-28-2",
        "momentum": 0.9,
        "weight_decay": 0.0005,

        "batch_size": batch_size,
        "num_workers": 12,

        "num_epochs": 1024,
        "epoch_length": 2 ** 16 // batch_size,  # epoch_length * num_epochs == 2 ** 20
        "learning_rate": 0.03,

        "validate_every": 1,
        # Logging:
        "display_iters": True,
        "checkpoint_every": 200,
        "debug": False,

        # online platform logging:
        "online_logging": None  # "Neptune" or "WandB"
    }

    config.update(get_ssl_config())
    config.update(get_backend_config())

    return config
