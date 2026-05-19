import pytest
import timm

from src.models.factory import TIMM_MODEL_IDS, get_model


@pytest.mark.parametrize("model_name", ["resnet18", "deit_tiny", "mlp_mixer_small"])
def test_model_factory_builds_supported_lightweight_models_without_pretrained_weights(model_name):
    model = get_model(model_name, num_classes=2, pretrained=False)

    output = model(next(model.parameters()).new_zeros((1, 3, 224, 224)))

    assert output.shape == (1, 2)


def test_model_factory_exposes_expected_timm_pretrained_model_ids():
    assert TIMM_MODEL_IDS == {
        "deit_tiny": "deit_tiny_patch16_224.fb_in1k",
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


def test_model_factory_rejects_removed_models():
    with pytest.raises(ValueError, match="Unsupported model"):
        get_model("unsupported_model", num_classes=2, pretrained=False)


def test_model_factory_does_not_fall_back_when_mlp_mixer_small_pretrained_weights_are_missing():
    model_id = TIMM_MODEL_IDS["mlp_mixer_small"]
    if timm.is_model_pretrained(model_id):
        pytest.skip("This timm installation provides pretrained MLP-Mixer Small weights")
    else:
        with pytest.raises(ValueError, match="no fallback model is used"):
            get_model("mlp_mixer_small", num_classes=2, pretrained=True)
