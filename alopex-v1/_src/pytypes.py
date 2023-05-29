"""Types of Alopex."""
import typing as tp
import chex


TrainState = chex.ArrayTree
Batch = chex.ArrayTree
Summary = tp.Mapping[str, chex.Array]
Scalars = tp.Mapping[str, chex.Array]
Prediction = chex.ArrayTree

LoggerState = tp.Any

TrainFun = tp.Callable[[TrainState, Batch], tp.Tuple[TrainState, Scalars]]
EvalFun = tp.Callable[[TrainState, Batch], Scalars]
PredFun = tp.Callable[[TrainState, Batch], Prediction]
