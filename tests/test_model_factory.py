import pytest
import timm

torch = pytest.importorskip("torch")
nn = pytest.importorskip("torch.nn")

from src.models.factory import TIMM_MODEL_IDS, get_model, resolve_model_training_options


@pytest.mark.parametrize("model_name", ["resnet18", "deit_tiny", "mlp_mixer_base", "mlp_mixer_small"])
def test_model_factory_builds_supported_lightweight_models_without_pretrained_weights(model_name):
    model = get_model(model_name, num_classes=2, pretrained=False)

    output = model(next(model.parameters()).new_zeros((1, 3, 224, 224)))

    assert output.shape == (1, 2)


def test_model_factory_exposes_expected_timm_pretrained_model_ids():
    assert TIMM_MODEL_IDS == {
        "deit_tiny": "deit_tiny_patch16_224.fb_in1k",
        "mlp_mixer_base": "mixer_b16_224.goog_in21k_ft_in1k",
        "mlp_mixer_small": "mixer_s16_224",
    }


def test_model_factory_freezes_backbone_by_default_for_resnet18():
    model = get_model("resnet18", num_classes=2, pretrained=False)

    trainable_names = {
        name
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }

    assert trainable_names == {"fc.weight", "fc.bias"}


def test_model_factory_builds_trainable_resnet18_mlp_head_with_frozen_backbone():
    model = get_model(
        "resnet18",
        num_classes=2,
        pretrained=False,
        freeze_backbone=True,
        classification_head="mlp",
    )

    output = model(torch.zeros((2, 3, 224, 224)))
    trainable_names = {
        name
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }

    assert output.shape == (2, 2)
    assert isinstance(model.fc, nn.Sequential)
    assert trainable_names == {
        "fc.0.weight",
        "fc.0.bias",
        "fc.1.weight",
        "fc.1.bias",
        "fc.4.weight",
        "fc.4.bias",
    }


def test_model_factory_freezes_pretrained_mlp_mixer_base_head_only():
    model = get_model("mlp_mixer_base", num_classes=2, pretrained=False, freeze_backbone=True)

    trainable_names = {
        name
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }

    assert trainable_names == {"head.weight", "head.bias"}


def test_model_factory_rejects_removed_models():
    with pytest.raises(ValueError, match="Unsupported model"):
        get_model("unsupported_model", num_classes=2, pretrained=False)


def test_model_factory_trains_mlp_mixer_from_scratch_when_pretrained_weights_are_missing():
    model_id = TIMM_MODEL_IDS["mlp_mixer_small"]
    if timm.is_model_pretrained(model_id):
        pytest.skip("This timm installation provides pretrained MLP-Mixer Small weights")

    with pytest.warns(RuntimeWarning, match="initialized from scratch"):
        model = get_model("mlp_mixer_small", num_classes=2, pretrained=True, freeze_backbone=True)

    output = model(next(model.parameters()).new_zeros((1, 3, 224, 224)))

    assert output.shape == (1, 2)
    assert all(parameter.requires_grad for parameter in model.parameters())


def test_resolve_model_training_options_unfreezes_missing_pretrained_timm_model(monkeypatch):
    monkeypatch.setattr(timm, "is_model_pretrained", lambda model_id: False)

    with pytest.warns(RuntimeWarning, match="fully trained"):
        options = resolve_model_training_options(
            "mlp_mixer_small",
            pretrained=True,
            freeze_backbone=True,
        )

    assert options.pretrained is False
    assert options.freeze_backbone is False
