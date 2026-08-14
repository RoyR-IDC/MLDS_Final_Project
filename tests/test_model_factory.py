import pytest
import timm

torch = pytest.importorskip("torch")
nn = pytest.importorskip("torch.nn")

import src.models.factory as factory_module
from src.models.factory import (
    TIMM_MODEL_IDS,
    create_model,
    get_imagenet_pretrained_model,
    get_model,
    resolve_model_training_options,
)
from src.models.registry import ACTIVE_MODEL_NAMES


@pytest.mark.parametrize("model_name", ["mobilenetv3_small", "deit_tiny", "gmlp_s16"])
def test_create_model_builds_active_models_without_pretrained_weights(model_name):
    model = create_model(model_name, num_classes=2, pretrained=False)

    output = model(next(model.parameters()).new_zeros((1, 3, 224, 224)))

    assert output.shape == (1, 2)


def test_model_factory_exposes_expected_timm_pretrained_model_ids():
    assert TIMM_MODEL_IDS == {
        "mobilenetv3_small": "mobilenetv3_small_100",
        "deit_tiny": "deit_tiny_patch16_224.fb_in1k",
        "gmlp_s16": "gmlp_s16_224.ra3_in1k",
        "resnet18": "resnet18",
    }
    assert ACTIVE_MODEL_NAMES == ("mobilenetv3_small", "deit_tiny", "gmlp_s16")


def test_create_model_rejects_legacy_resnet18():
    with pytest.raises(ValueError, match="active models only"):
        create_model("resnet18", num_classes=2, pretrained=False)


def test_get_model_keeps_legacy_resnet18_constructible():
    model = get_model("resnet18", num_classes=2, pretrained=False)

    output = model(next(model.parameters()).new_zeros((1, 3, 224, 224)))
    trainable_names = {
        name
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }

    assert output.shape == (1, 2)
    assert trainable_names == {"fc.weight", "fc.bias"}


def test_get_model_builds_trainable_mlp_head_with_frozen_legacy_resnet18():
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


def test_get_model_freezes_active_timm_head_only():
    model = get_model("gmlp_s16", num_classes=2, pretrained=False, freeze_backbone=True)

    trainable_names = {
        name
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }

    assert trainable_names == {"head.weight", "head.bias"}


def test_model_factory_rejects_removed_mixer_models():
    for model_name in ("mlp_mixer_base", "mlp_mixer_small"):
        with pytest.raises(ValueError, match="Unsupported model"):
            get_model(model_name, num_classes=2, pretrained=False)


def test_resolve_model_training_options_rejects_missing_pretrained_timm_model(monkeypatch):
    monkeypatch.setattr(timm, "is_model_pretrained", lambda model_id: False)

    with pytest.raises(ValueError, match="pretrained ImageNet-1K weights"):
        resolve_model_training_options(
            "gmlp_s16",
            pretrained=True,
            freeze_backbone=True,
        )


def test_pretrained_validation_rejects_non_imagenet1k_model_id(monkeypatch):
    monkeypatch.setitem(factory_module.TIMM_MODEL_IDS, "gmlp_s16", "mixer_b16_224.goog_in21k_ft_in1k")

    with pytest.raises(ValueError, match="ImageNet-1K-only"):
        get_model("gmlp_s16", num_classes=2, pretrained=True)


def test_get_imagenet_pretrained_model_preserves_native_head(monkeypatch):
    class FakeModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.head = nn.Linear(4, 1000)
            self.pretrained_cfg = {"num_classes": 1000, "tag": "in1k"}

        def forward(self, images):
            return self.head(images.new_zeros((images.shape[0], 4)))

    monkeypatch.setattr(timm, "is_model_pretrained", lambda model_id: True)
    monkeypatch.setattr(timm, "create_model", lambda *args, **kwargs: FakeModel())

    model = get_imagenet_pretrained_model("gmlp_s16", freeze_backbone=True)

    output = model(torch.zeros((1, 3, 224, 224)))

    assert output.shape == (1, 1000)
    assert model.head.out_features == 1000
    assert not any(parameter.requires_grad for parameter in model.parameters())
