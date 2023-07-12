custom resnet parmas

```yaml
backbone:
  name: "resnet_custom"
  kwargs:
    model_config:
      backbone_type: "resnet18"
      only_features: True
      batchnorm_layers: True
      width_scale: 1
      skips: False
```