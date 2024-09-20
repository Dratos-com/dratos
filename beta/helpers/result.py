from typing import TypeVar, Generic, Union, Callable

T = TypeVar('T')
E = TypeVar('E')

class Result(Generic[T, E]):
    def __init__(self, value: Union[T, E], is_error: bool = False):
        self.value = value
        self.is_error = is_error

    @staticmethod
    def Ok(value: T) -> 'Result[T, E]':
        return Result(value)

    @staticmethod
    def Error(error: E) -> 'Result[T, E]':
        return Result(error, is_error=True)

    def bind(self, func: Callable[[T], 'Result']) -> 'Result':
        if self.is_error:
            return self
        try:
            return func(self.value)
        except Exception as e:
            return Result.Error(e)

    def map(self, func: Callable[[T], U]) -> 'Result[U, E]':
        if self.is_error:
            return Result.Error(self.value)
        try:
            return Result.Ok(func(self.value))
        except Exception as e:
            return Result.Error(e)