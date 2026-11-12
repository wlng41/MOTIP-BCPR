"""Integration checks for BCPR bias injection into MOTIP's ID decoder."""

import copy

import torch

from models.motip.id_decoder import IDDecoder


def _decoder():
    return IDDecoder(
        feature_dim=16,
        id_dim=8,
        ffn_dim_ratio=2,
        num_layers=2,
        head_dim=8,
        num_id_vocabulary=6,
        rel_pe_length=4,
        use_aux_loss=False,
        use_shared_aux_head=True,
        bcpr_config={
            "ENABLED": True,
            "PROJ_DIM": 4,
            "HIDDEN_DIM": 8,
            "LAMBDA": 1.0,
            "TAU": 1.0,
            "USE_COMPETITOR": True,
            "CHUNK_QUERIES": 0,
            "USE_QH": True,
            "USE_OWN": True,
            "USE_HC": True,
            "USE_QC": True,
            "ALPHA": 1.0,
        },
    )


def _seq_info():
    generator = torch.Generator().manual_seed(11)
    shape = (1, 1, 3, 3)
    trajectory_features_raw = torch.randn(*shape, 16, generator=generator)
    trajectory_features = torch.randn(*shape, 16, generator=generator)
    unknown_features = torch.randn(*shape, 16, generator=generator)
    trajectory_masks = torch.tensor([[[
        [False, False, False],
        [False, False, False],
        [False, False, True],
    ]]])
    unknown_masks = torch.zeros(shape, dtype=torch.bool)
    trajectory_times = torch.arange(3).view(1, 1, 3, 1).expand(shape).clone()
    unknown_times = trajectory_times.clone()
    trajectory_ids = torch.arange(3).view(1, 1, 1, 3).expand(shape).clone()
    unknown_ids = trajectory_ids.clone()
    return {
        "trajectory_features": trajectory_features,
        "trajectory_features_pre_adapter": trajectory_features_raw,
        "unknown_features": unknown_features,
        "trajectory_id_labels": trajectory_ids,
        "unknown_id_labels": unknown_ids,
        "trajectory_times": trajectory_times,
        "unknown_times": unknown_times,
        "trajectory_masks": trajectory_masks,
        "unknown_masks": unknown_masks,
    }


def test_alpha_zero_is_exact_decoder_noop():
    decoder = _decoder().eval()
    info = _seq_info()
    decoder._bcpr_alpha = 0.0
    with torch.no_grad():
        logits_off, labels_off, masks_off = decoder(copy.deepcopy(info), False)
        decoder._bcpr_alpha = 1.0
        logits_on, labels_on, masks_on = decoder(copy.deepcopy(info), False)
    # The BCPR output head starts at zero, so fresh OFF and ON endpoints match exactly.
    assert torch.equal(logits_off, logits_on)
    assert labels_off is labels_on is None
    assert torch.equal(masks_off, masks_on)


def test_trained_bias_changes_on_but_not_off_endpoint():
    decoder = _decoder().eval()
    info = _seq_info()
    with torch.no_grad():
        torch.nn.init.normal_(decoder.bcpr.mlp[-1].weight, std=0.5)
        torch.nn.init.normal_(decoder.bcpr.mlp[-1].bias, std=0.5)
        decoder._bcpr_alpha = 0.0
        logits_off_a, _, _ = decoder(copy.deepcopy(info), False)
        decoder._bcpr_alpha = 1.0
        logits_on, _, _ = decoder(copy.deepcopy(info), False)
        decoder._bcpr_alpha = 0.0
        logits_off_b, _, _ = decoder(copy.deepcopy(info), False)
    assert torch.equal(logits_off_a, logits_off_b)
    assert not torch.equal(logits_off_a, logits_on)
