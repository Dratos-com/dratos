from __future__ import annotations

import typing

if typing.TYPE_CHECKING:
    from beta.models.types.triton_model import TritonModel


class Models:
    @staticmethod
    def triton(model_name, engine, is_async=False):
        from beta.models.types.triton_model import TritonModel

        return TritonModel(model_name, engine, is_async)
