from .vgg import vgg19 as default_vgg19


def vgg19(method, *args, **kwargs):
    return default_vgg19(*args, **kwargs)


__all__ = ["vgg19"]
