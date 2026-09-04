from . import em, ez_mia, mia

REGISTRY = {
    em.FAMILY: em,
    mia.FAMILY: mia,
    ez_mia.FAMILY: ez_mia,
}

__all__ = ["REGISTRY", "em", "mia", "ez_mia"]
