from ml_collections import config_dict

from configs.wandb_pa645zxs import get_config as get_base_config


def get_config() -> config_dict.ConfigDict:
    cfg = get_base_config()

    cfg.msg_probe_fingerprint = "morgan"
    cfg.msg_probe_tune_metric = "msg_probe/test/auc_morgan_mean"
    cfg.msg_probe_select_metric = cfg.msg_probe_tune_metric
    cfg.msg_probe_pairwise_alignment_num_pairs = 20_000
    cfg.msg_probe_pairwise_alignment_plot = True
    cfg.run_name_suffix = f"{cfg.run_name_suffix}-morgan4096r2-probe"

    return cfg
