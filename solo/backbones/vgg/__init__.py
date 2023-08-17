from .vgg import vgg19 as default_vgg19
from .vgg import vgg19_bn as default_vgg19_bn


def vgg19(method, *args, **kwargs):
    return default_vgg19(*args, **kwargs)


def vgg19_bn(method, *args, **kwargs):
    return default_vgg19_bn(*args, **kwargs)


__all__ = ["vgg19", "vgg19_bn"]
