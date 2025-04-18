import typing as t

MODEL_TYPE = t.Literal["lstm", "gpt2"]
ESTIMATION_TYPE = t.Literal["100hpy", "500hpy", "1000hpy"]
DEVICE_TYPE = t.Literal["cuda", "cpu", "mps"]
