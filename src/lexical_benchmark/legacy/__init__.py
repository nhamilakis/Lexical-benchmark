import warnings

# This helps warn whenever a legacy code-block is imported
warnings.warn(
    "You are running legacy code, please migrate to newer sections of the project",
    category=DeprecationWarning,
    stacklevel=1,
)
