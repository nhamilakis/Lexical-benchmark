import bisect
import collections

import numpy as np
import polars as pl

from lexical_benchmark import lb_types, metadata, settings


def get_tokens_per_month(month_range: tuple[int, int] = settings.MONTH_RANGE, lang: str = "EN") -> dict[int, int]:
    """Load the token per month list (Requires CHILDES to be installed)."""
    childes_md: metadata.CHILDESMetaDir = metadata.get_config("childes", lang=lang)
    if not childes_md.speech_quantities.is_file():
        raise ValueError("Child speech quantities not found !!")
    df = pl.read_csv(childes_md.speech_quantities, separator=";")
    return dict(
        df.filter(pl.col("age_months").is_between(month_range[0], month_range[1]))
        .select("age_months", "total_words")
        .rows()
    )


def _get_percentage_prev(n: int, prev_key: int, next_key: int) -> float:
    total_distance = next_key - prev_key
    x_distance = n - prev_key
    return x_distance / total_distance


def _get_prev_next(n: int, mapping: dict[int, int]) -> tuple[int, int]:
    """Get previous & next item from a month2model mapping."""
    keys = sorted(mapping.keys())
    idx = bisect.bisect_left(keys, n)
    prev_key = keys[idx - 1] if idx > 0 else None
    next_key = keys[idx] if idx < len(keys) else None
    return prev_key, next_key


def proportions(estimation_type: lb_types.ESTIMATION_TYPE, month_range: tuple[int, int] = settings.MONTH_RANGE) -> dict:
    """Compute model proportions for given month range."""
    estimate_mapping = settings.MONTH_2_PSEUDO_MONTH[estimation_type]
    model2month = {}
    for month in range(month_range[0], month_range[1] + 1):
        if month in estimate_mapping:
            model2month[month] = {estimate_mapping[month]: 1}
        else:
            prev_m, next_m = _get_prev_next(month, estimate_mapping)
            if prev_m < month < next_m:
                prev_m, next_m = _get_prev_next(month, estimate_mapping)
                distance_prev = _get_percentage_prev(month, prev_m, next_m)
                distance_next = 1 - distance_prev
                model2month[month] = {estimate_mapping[prev_m]: distance_prev, estimate_mapping[next_m]: distance_next}
            else:
                raise ValueError("Failed to find prev & next !!")
    return model2month


MODEL_PROPORTION: dict[lb_types.ESTIMATION_TYPE, dict[int, float]] = {
    estimation: proportions(estimation_type=estimation) for estimation in settings.MONTH_ESTIMATES
}


def get_token_count_dict(
    model_size: int,
    lang: str,
    *,
    month_range: tuple[int, int] = settings.MONTH_RANGE,
    month_estimates: tuple[lb_types.ESTIMATION_TYPE, ...] = settings.MONTH_ESTIMATES,
) -> dict[tuple[str, int], int]:
    """Get token count for a corresponding model size."""
    f_mapping = {}
    token_count_index = get_tokens_per_month(month_range=month_range, lang=lang)
    model_proportion: dict[lb_types.ESTIMATION_TYPE, dict[int, float]] = {
        estimation: proportions(estimation_type=estimation) for estimation in month_estimates
    }

    for estimation, prop_obj in model_proportion.items():
        for month, proportion in prop_obj.items():
            if model_size in proportion:
                tk_count = np.round(proportion[model_size] * token_count_index[month])
                f_mapping[(estimation, month)] = int(tk_count)

    return f_mapping


def get_month_to_model_size(model_sizes: list[int], lang: str, month_estimate: lb_types.ESTIMATION_TYPE) -> dict:
    """Build a month to model size mapping."""
    build_thing = collections.defaultdict(list)
    for ml_size in model_sizes:
        source = get_token_count_dict(model_size=ml_size, lang=lang, month_estimates=(month_estimate,))
        for _, month in source:
            build_thing[month].append(ml_size)
    return build_thing
