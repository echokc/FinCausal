from typing import Dict


def single_feature_schema_variants(
    *,
    canonical_name: str,
    canonical_time_col: str,
    canonical_feature_col: str,
    canonical_target_col: str,
    canonical_role: str,
    aliases: Dict[str, Dict[str, str]],
) -> Dict[str, Dict[str, str]]:
    variants = {
        canonical_name: {
            "time_col": canonical_time_col,
            "feature_col": canonical_feature_col,
            "target_col": canonical_target_col,
            "role": canonical_role,
        }
    }
    variants.update(aliases)
    return variants


def trade_news_schema_variants(
    *,
    canonical_name: str,
    trade_file_role: str,
    news_file_role: str,
    trade_time_col: str,
    news_time_col: str,
    trade_id_col: str,
    news_id_col: str,
    content_col: str,
    aliases: Dict[str, Dict[str, str]],
) -> Dict[str, Dict[str, str]]:
    variants = {
        canonical_name: {
            "trade_file_role": trade_file_role,
            "news_file_role": news_file_role,
            "trade_time_col": trade_time_col,
            "news_time_col": news_time_col,
            "trade_id_col": trade_id_col,
            "news_id_col": news_id_col,
            "content_col": content_col,
        }
    }
    variants.update(aliases)
    return variants


def price_series_schema_variants(
    *,
    canonical_name: str = "canonical_price_series",
    time_col: str = "timestamp",
    price_col: str = "close",
    volume_col: str | None = "volume",
    role: str = "price_series",
    aliases: Dict[str, Dict[str, str]] | None = None,
) -> Dict[str, Dict[str, str]]:
    canonical = {
        "time_col": time_col,
        "price_col": price_col,
        "role": role,
    }
    if volume_col is not None:
        canonical["volume_col"] = volume_col
    variants = {canonical_name: canonical}
    variants.update(aliases or {})
    return variants


def daily_return_schema_variants(
    *,
    canonical_name: str = "canonical_daily_returns",
    day_col: str = "day",
    return_col: str = "daily_return",
    role: str = "daily_return_series",
    aliases: Dict[str, Dict[str, str]] | None = None,
) -> Dict[str, Dict[str, str]]:
    variants = {
        canonical_name: {
            "role": role,
            "day_col": day_col,
            "return_col": return_col,
        }
    }
    variants.update(aliases or {})
    return variants


def return_panel_schema_variants(
    *,
    canonical_name: str = "wide_return_panel",
    asset_prefix: str = "STK_",
    n_assets: int = 50,
    n_observations: int = 30,
    role: str = "asset_return_panel",
    aliases: Dict[str, Dict[str, str]] | None = None,
) -> Dict[str, Dict[str, str]]:
    variants = {
        canonical_name: {
            "role": role,
            "layout": "wide_matrix",
            "asset_prefix": asset_prefix,
            "n_assets": str(n_assets),
            "n_observations": str(n_observations),
        }
    }
    variants.update(aliases or {})
    return variants


def trade_tape_schema_variants(
    *,
    canonical_name: str = "canonical_trade_tape",
    price_col: str = "price",
    volume_col: str = "volume",
    buyer_col: str = "buyer",
    seller_col: str = "seller",
    trade_id_col: str = "trade_id",
    role: str = "public_trade_tape",
    aliases: Dict[str, Dict[str, str]] | None = None,
) -> Dict[str, Dict[str, str]]:
    variants = {
        canonical_name: {
            "role": role,
            "price_col": price_col,
            "volume_col": volume_col,
            "buyer_col": buyer_col,
            "seller_col": seller_col,
            "trade_id_col": trade_id_col,
        }
    }
    variants.update(aliases or {})
    return variants


def execution_log_schema_variants(
    *,
    canonical_name: str = "canonical_execution_log",
    time_col: str = "timestamp",
    side_col: str = "side",
    qty_col: str = "qty",
    price_col: str = "price",
    volatility_col: str = "vol_1min",
    role: str = "execution_log",
    aliases: Dict[str, Dict[str, str]] | None = None,
) -> Dict[str, Dict[str, str]]:
    variants = {
        canonical_name: {
            "role": role,
            "time_col": time_col,
            "side_col": side_col,
            "qty_col": qty_col,
            "price_col": price_col,
            "volatility_col": volatility_col,
        }
    }
    variants.update(aliases or {})
    return variants
