import typing as t

MODEL_TYPE = t.Literal["lstm", "gpt2"]
ESTIMATION_TYPE = t.Literal["100hpy", "500hpy", "1000hpy"]
TARGET_MONTH = [6, 12, 18, 24, 30, 36]
DEVICE_TYPE = t.Literal["cuda", "cpu", "mps"]
