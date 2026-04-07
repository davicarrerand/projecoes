"""
Motor de projecao mensal para producao diaria de produtos financeiros.

Este arquivo foi desenhado para rodar de forma standalone em ambiente corporativo:
ele contem o pipeline completo de normalizacao, engenharia de features, backtest,
calibracao de pesos por produto, tratamento de intermitencia, reconciliacao
hierarquica e uma camada de apresentacao em Streamlit. A entrada esperada e um
CSV com as colunas `anomesdia`, `familia`, `produto` e `valor`.

No contexto bancario, a previsao mensal de producao precisa ser auditavel e
estavel. Por isso o codigo evita dependencias complexas e privilegia metodos
interpretaveis: curva intra-mensal, run rate por dia da semana, regressao Ridge
e Croston simplificado para produtos intermitentes. A combinacao nao usa pesos
fixos: os pesos sao calibrados via backtest historico por produto e janela de
treino, simulando cortes nos dias 5, 10, 15 e 20 de meses passados.

Como executar em modo app:
    streamlit run motor_projecao_mensal.py

Como executar em modo batch:
    python motor_projecao_mensal.py --csv caminho/arquivo.csv --anomes 202604 --output-dir saida

Dependencias:
    pandas, numpy, scikit-learn, plotly, streamlit
"""

from __future__ import annotations

import argparse
import calendar
import math
import os
import warnings
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore", category=RuntimeWarning)


# =============================================================================
# CONFIGURACAO E TIPOS
# =============================================================================
#
# Este bloco concentra parametros globais e estruturas tipadas usadas pelo motor.
# Em ambiente bancario, manter a configuracao em um unico ponto reduz risco
# operacional: janelas de treino, dias simulados no backtest e limiares de
# classificacao ficam explicitos e rastreaveis, em vez de espalhados pelo codigo.
#
# A classe dataclass tambem ajuda a tornar o comportamento auditavel. Cada
# produto recebe uma configuracao com janela escolhida, pesos calibrados e
# indicadores de volume/intermitencia, o que facilita explicar a decisao do
# modelo para areas de negocio, controle e governanca.


REQUIRED_COLUMNS = ["anomesdia", "familia", "produto", "valor"]
TRAIN_WINDOWS_MONTHS = [3, 6, 12]
BACKTEST_CUTOFF_DAYS = [5, 10, 15, 20]
METHODS_CORE = ["curva", "run_rate", "ridge"]
METHODS_WITH_CROSTON = ["curva", "run_rate", "ridge", "croston"]
EPS = 1e-9


@dataclass
class ProductCalibration:
    """Guarda a configuracao calibrada de um produto."""

    familia: str
    produto: str
    is_intermittent: bool
    volume_class: str
    selected_window: int
    weights: Dict[str, float]
    method_mape: Dict[str, float]
    ensemble_mape: float
    dominant_method: str
    explanation: str = ""
    backtest_rows: pd.DataFrame = field(default_factory=pd.DataFrame)


@dataclass
class ProjectionArtifacts:
    """Container com todas as saidas do motor."""

    final_df: pd.DataFrame
    daily_df: pd.DataFrame
    product_metrics: pd.DataFrame
    family_metrics: pd.DataFrame
    backtest_df: pd.DataFrame
    explanations: pd.DataFrame


# =============================================================================
# FUNCOES DE DATA E VALIDACAO
# =============================================================================
#
# Este bloco padroniza manipulacao de datas, parsing de anomes e validacao de
# schema. A previsao mensal depende fortemente de cortes temporais corretos:
# um erro de interpretacao de `yyyymmdd` ou de ultimo dia do mes pode gerar
# vazamento de informacao e superestimar a acuracia em backtest.
#
# No contexto de producao bancaria, validacao defensiva e essencial porque CSVs
# operacionais frequentemente chegam com tipos inconsistentes, duplicidades ou
# valores faltantes. Ao falhar cedo com mensagens claras, o motor evita produzir
# uma projecao aparentemente precisa sobre uma base mal formada.


def parse_anomes(anomes: int | str) -> pd.Period:
    """Converte `yyyymm` para Period mensal, validando formato e calendario."""
    text = str(anomes).strip()
    if len(text) != 6 or not text.isdigit():
        raise ValueError("O parametro anomes deve estar no formato yyyymm, por exemplo 202604.")
    year, month = int(text[:4]), int(text[4:6])
    if month < 1 or month > 12:
        raise ValueError("Mes invalido em anomes. Use valores entre 01 e 12.")
    return pd.Period(f"{year}-{month:02d}", freq="M")


def month_start(period: pd.Period) -> pd.Timestamp:
    """Retorna o primeiro dia de um Period mensal."""
    return period.to_timestamp(how="start")


def month_end(period: pd.Period) -> pd.Timestamp:
    """Retorna o ultimo dia de um Period mensal."""
    return period.to_timestamp(how="end").normalize()


def yyyymmdd_to_date(series: pd.Series) -> pd.Series:
    """Converte coluna `yyyymmdd` para datetime, com erro explicito se invalida."""
    text = series.astype(str).str.replace(r"\.0$", "", regex=True).str.zfill(8)
    parsed = pd.to_datetime(text, format="%Y%m%d", errors="coerce")
    if parsed.isna().any():
        bad = series[parsed.isna()].head(5).tolist()
        raise ValueError(f"Foram encontrados valores invalidos em anomesdia. Exemplos: {bad}")
    return parsed


def validate_input_schema(df: pd.DataFrame) -> None:
    """Valida se o CSV contem as colunas obrigatorias."""
    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"Colunas obrigatorias ausentes: {missing}. Esperado: {REQUIRED_COLUMNS}")


# =============================================================================
# NORMALIZACAO E FEATURES
# =============================================================================
#
# Este bloco transforma a base transacional em uma malha diaria continua por
# familia e produto. Dias sem observacao sao preenchidos com zero porque, em
# producao diaria de produtos financeiros, ausencia de linha geralmente deve ser
# tratada como ausencia de producao, nao como valor desconhecido.
#
# As features temporais sao simples e interpretaveis: dia do mes, dia da semana,
# semana do mes, mes e ano. Elas capturam sazonalidades operacionais recorrentes
# como concentracao em dias uteis, viradas de semana e padroes de fechamento,
# sem depender de variaveis de negocio externas.


def add_calendar_features(df: pd.DataFrame, date_col: str = "data") -> pd.DataFrame:
    """Adiciona features temporais diarias usadas por todos os metodos."""
    out = df.copy()
    out[date_col] = pd.to_datetime(out[date_col])
    out["ano"] = out[date_col].dt.year
    out["mes"] = out[date_col].dt.month
    out["anomes"] = out[date_col].dt.to_period("M").astype(str).str.replace("-", "").astype(int)
    out["dia_mes"] = out[date_col].dt.day
    out["dia_semana"] = out[date_col].dt.dayofweek
    out["semana_mes"] = ((out["dia_mes"] - 1) // 7 + 1).astype(int)
    out["dias_no_mes"] = out[date_col].dt.days_in_month
    out["is_fim_semana"] = (out["dia_semana"] >= 5).astype(int)
    return out


def load_and_normalize(csv_path: str) -> pd.DataFrame:
    """
    Le e normaliza o CSV de entrada em serie diaria continua por produto.

    A funcao agrega duplicidades no mesmo dia/produto antes de expandir a malha,
    pois bases de producao podem conter multiplas linhas por produto em razao de
    canais, sistemas origem ou reprocessamentos. A agregacao garante que cada
    combinacao familia-produto-data tenha uma unica observacao de valor.
    """
    raw = pd.read_csv(csv_path)
    validate_input_schema(raw)

    df = raw[REQUIRED_COLUMNS].copy()
    df["data"] = yyyymmdd_to_date(df["anomesdia"])
    df["familia"] = df["familia"].astype(str).str.strip()
    df["produto"] = df["produto"].astype(str).str.strip()
    df["valor"] = pd.to_numeric(df["valor"], errors="coerce").fillna(0.0)

    grouped = (
        df.groupby(["familia", "produto", "data"], as_index=False)["valor"]
        .sum()
        .sort_values(["familia", "produto", "data"])
    )

    min_date, max_date = grouped["data"].min(), grouped["data"].max()
    full_dates = pd.date_range(min_date, max_date, freq="D")
    products = grouped[["familia", "produto"]].drop_duplicates()
    full_index = products.merge(pd.DataFrame({"data": full_dates}), how="cross")

    normalized = full_index.merge(grouped, on=["familia", "produto", "data"], how="left")
    normalized["valor"] = normalized["valor"].fillna(0.0)
    normalized["anomesdia"] = normalized["data"].dt.strftime("%Y%m%d").astype(int)
    normalized = add_calendar_features(normalized)
    return normalized.sort_values(["familia", "produto", "data"]).reset_index(drop=True)


# =============================================================================
# CLASSIFICACAO DE PRODUTOS
# =============================================================================
#
# Este bloco identifica intermitencia e volume. Produtos com muitos zeros se
# comportam de forma diferente de produtos recorrentes: a media diaria comum pode
# superestimar a producao, enquanto um metodo de demanda intermitente como
# Croston separa tamanho de evento e intervalo entre eventos.
#
# A classificacao de volume tambem protege o motor contra overfitting. Produtos
# de baixo volume tendem a ser mais ruidosos e menos previsiveis; nesses casos,
# metodos robustos baseados em mediana, run rate e Croston recebem espaco natural
# na calibracao porque o backtest penaliza modelos instaveis.


def classify_product(product_df: pd.DataFrame) -> Tuple[bool, str, Dict[str, float]]:
    """Classifica produto por intermitencia e volume usando apenas historico."""
    values = product_df["valor"].astype(float)
    zero_ratio = float((values <= EPS).mean())
    non_zero = values[values > EPS]
    avg_daily = float(values.mean())
    median_non_zero = float(non_zero.median()) if len(non_zero) else 0.0

    is_intermittent = zero_ratio > 0.40
    monthly_totals = product_df.groupby(product_df["data"].dt.to_period("M"))["valor"].sum()
    median_monthly = float(monthly_totals.median()) if len(monthly_totals) else 0.0
    p60_daily = float(values.quantile(0.60)) if len(values) else 0.0

    if median_monthly <= max(1.0, p60_daily * 10.0):
        volume_class = "baixo volume"
    else:
        volume_class = "alto volume"

    diagnostics = {
        "zero_ratio": zero_ratio,
        "avg_daily": avg_daily,
        "median_non_zero": median_non_zero,
        "median_monthly": median_monthly,
    }
    return is_intermittent, volume_class, diagnostics


# =============================================================================
# CURVA INTRA-MENSAL
# =============================================================================
#
# Este bloco implementa a curva intra-mensal por produto. Para cada mes historico,
# calcula-se o percentual acumulado do total mensal ate cada dia; a curva final e
# a mediana desses percentuais, o que reduz impacto de meses atipicos e eventos
# extremos de producao.
#
# Em negocio, essa abordagem responde a pergunta "quanto do mes normalmente ja
# teria acontecido ate hoje?". Ela e especialmente util quando existe forte
# padrao de fechamento dentro do mes e quando o realizado parcial do mes alvo ja
# contem informacao relevante sobre o total final.


def build_intramonth_curve(train_df: pd.DataFrame) -> pd.Series:
    """Calcula curva mediana de percentual acumulado por dia do mes."""
    if train_df.empty:
        return pd.Series(dtype=float)

    tmp = train_df.copy()
    tmp["periodo"] = tmp["data"].dt.to_period("M")
    month_total = tmp.groupby("periodo")["valor"].transform("sum")
    tmp["cum_value"] = tmp.groupby("periodo")["valor"].cumsum()
    tmp["cum_pct"] = np.where(month_total > EPS, tmp["cum_value"] / month_total, np.nan)

    curve = tmp.dropna(subset=["cum_pct"]).groupby("dia_mes")["cum_pct"].median().clip(0.0, 1.0)
    curve = curve.sort_index().cummax()
    if len(curve):
        curve.loc[curve.index.max()] = min(1.0, max(curve.iloc[-1], curve.max()))
    return curve


def forecast_curve_method(train_df: pd.DataFrame, observed_df: pd.DataFrame, future_dates: pd.DatetimeIndex) -> pd.Series:
    """Projeta dias futuros distribuindo o restante pela curva intra-mensal mediana."""
    if len(future_dates) == 0:
        return pd.Series(dtype=float)

    curve = build_intramonth_curve(train_df)
    realized = float(observed_df["valor"].sum())
    cutoff_day = int(observed_df["dia_mes"].max()) if not observed_df.empty else 0
    days_in_month = int(future_dates[-1].days_in_month)

    if curve.empty or realized <= EPS:
        fallback = max(float(train_df["valor"].mean()) if not train_df.empty else 0.0, 0.0)
        return pd.Series(fallback, index=future_dates)

    cutoff_pct = float(curve.reindex([cutoff_day], method="ffill").iloc[0]) if cutoff_day in range(1, 32) else np.nan
    if not np.isfinite(cutoff_pct) or cutoff_pct <= EPS:
        cutoff_pct = max(cutoff_day / max(days_in_month, 1), EPS)

    projected_total = realized / min(max(cutoff_pct, EPS), 0.995)

    all_days = pd.Index(range(1, days_in_month + 1), name="dia_mes")
    full_curve = curve.reindex(all_days).interpolate(limit_direction="both").fillna(method="ffill").fillna(0.0).clip(0, 1)
    full_curve.iloc[-1] = 1.0
    full_curve = full_curve.cummax()

    prev_pct = cutoff_pct
    future_values = []
    for date in future_dates:
        pct = float(full_curve.loc[date.day]) if date.day in full_curve.index else 1.0
        incremental = max(pct - prev_pct, 0.0) * projected_total
        future_values.append(max(incremental, 0.0))
        prev_pct = max(prev_pct, pct)

    return pd.Series(future_values, index=future_dates)


# =============================================================================
# RUN RATE INTELIGENTE
# =============================================================================
#
# Este bloco implementa uma media movel recente ajustada por padrao de dia da
# semana. A media movel captura o ritmo atual do produto, enquanto os fatores de
# dia da semana evitam projetar um domingo como se fosse uma segunda-feira, algo
# importante em operacoes bancarias com calendario operacional assimetrico.
#
# O metodo e propositalmente simples e robusto. Ele reage a aceleracoes ou
# desaceleracoes recentes sem exigir muitos parametros, e o backtest decide
# quanto peso esse sinal deve receber para cada produto.


def forecast_run_rate_method(train_df: pd.DataFrame, observed_df: pd.DataFrame, future_dates: pd.DatetimeIndex, window_days: int = 5) -> pd.Series:
    """Projeta dias futuros com media movel recente ajustada por dia da semana."""
    if len(future_dates) == 0:
        return pd.Series(dtype=float)

    combined = pd.concat([train_df, observed_df], ignore_index=True).sort_values("data")
    recent = combined.tail(max(window_days, 1))
    recent_mean = float(recent["valor"].mean()) if not recent.empty else 0.0
    global_mean = float(train_df["valor"].mean()) if not train_df.empty else recent_mean

    dow_mean = train_df.groupby("dia_semana")["valor"].mean()
    dow_factor = (dow_mean / max(global_mean, EPS)).replace([np.inf, -np.inf], np.nan).fillna(1.0)
    dow_factor = dow_factor.clip(lower=0.25, upper=3.00)

    preds = []
    for date in future_dates:
        factor = float(dow_factor.get(date.dayofweek, 1.0))
        preds.append(max(recent_mean * factor, 0.0))
    return pd.Series(preds, index=future_dates)


# =============================================================================
# MODELO ESTATISTICO RIDGE
# =============================================================================
#
# Este bloco implementa uma regressao Ridge com features temporais. A Ridge e
# adequada para um motor corporativo porque e rapida, estavel, interpretavel em
# termos de variaveis e menos sensivel a multicolinearidade do que regressao
# linear simples.
#
# O objetivo aqui nao e competir com modelos complexos de machine learning, mas
# capturar padroes temporais recorrentes sem exigir variaveis externas. Quando a
# serie tem volume suficiente, o backtest tende a dar mais peso ao modelo; quando
# a serie e ruidosa, o proprio erro historico reduz sua influencia.


FEATURE_COLUMNS = ["dia_mes", "dia_semana", "semana_mes", "mes", "ano", "is_fim_semana"]


def fit_ridge_model(train_df: pd.DataFrame) -> Optional[Pipeline]:
    """Treina Ridge para prever valor diario com features temporais."""
    clean = train_df.dropna(subset=["valor"]).copy()
    if clean.empty or clean["valor"].sum() <= EPS or clean["data"].nunique() < 20:
        return None

    model = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("ridge", Ridge(alpha=2.0, random_state=42)),
        ]
    )
    model.fit(clean[FEATURE_COLUMNS], clean["valor"].astype(float))
    return model


def forecast_ridge_method(train_df: pd.DataFrame, future_df: pd.DataFrame) -> pd.Series:
    """Projeta dias futuros com regressao Ridge e trava previsoes negativas em zero."""
    if future_df.empty:
        return pd.Series(dtype=float)

    model = fit_ridge_model(train_df)
    if model is None:
        fallback = max(float(train_df["valor"].mean()) if not train_df.empty else 0.0, 0.0)
        return pd.Series(fallback, index=pd.to_datetime(future_df["data"]))

    preds = model.predict(future_df[FEATURE_COLUMNS])
    preds = np.maximum(preds, 0.0)
    return pd.Series(preds, index=pd.to_datetime(future_df["data"]))


# =============================================================================
# CROSTON SIMPLIFICADO PARA INTERMITENCIA
# =============================================================================
#
# Este bloco implementa Croston simplificado. Para produtos intermitentes, a
# serie e composta por muitos zeros e poucos eventos positivos; prever a media
# diaria diretamente pode diluir demais a informacao de tamanho dos eventos ou
# reagir de forma exagerada ao ultimo evento.
#
# Croston estima separadamente o tamanho medio das demandas positivas e o
# intervalo medio entre elas. O resultado e uma taxa diaria esperada mais
# apropriada para produtos esparsos, que e integrada como mais um metodo no
# ensemble quando o produto tem muitos zeros.


def croston_daily_rate(values: Iterable[float], alpha: float = 0.10) -> float:
    """Calcula taxa diaria esperada pelo metodo de Croston simplificado."""
    demand_size = None
    interval = None
    periods_since = 0

    for value in values:
        periods_since += 1
        if value > EPS:
            if demand_size is None:
                demand_size = float(value)
                interval = float(periods_since)
            else:
                demand_size = demand_size + alpha * (float(value) - demand_size)
                interval = interval + alpha * (float(periods_since) - interval)
            periods_since = 0

    if demand_size is None or interval is None or interval <= EPS:
        return 0.0
    return max(demand_size / interval, 0.0)


def forecast_croston_method(train_df: pd.DataFrame, observed_df: pd.DataFrame, future_dates: pd.DatetimeIndex) -> pd.Series:
    """Projeta dias futuros com taxa diaria Croston estimada ate a data de corte."""
    if len(future_dates) == 0:
        return pd.Series(dtype=float)

    combined = pd.concat([train_df, observed_df], ignore_index=True).sort_values("data")
    rate = croston_daily_rate(combined["valor"].astype(float).tolist())
    return pd.Series(rate, index=future_dates)


# =============================================================================
# BACKTEST E CALIBRACAO
# =============================================================================
#
# Este bloco e o coracao de melhoria continua do motor. Em vez de escolher pesos
# fixos, o codigo simula previsoes historicas nos dias 5, 10, 15 e 20, mede MAPE
# por metodo e usa esses erros para escolher janela de treino e calibrar pesos
# por produto.
#
# Isso e importante porque produtos financeiros podem ter regimes muito
# diferentes: alguns dependem fortemente da curva do mes, outros respondem melhor
# ao ritmo recente, e outros possuem sazonalidade semanal capturada por regressao.
# A calibracao por produto evita uma regra unica para toda a carteira.


def safe_mape(actual: float, predicted: float) -> float:
    """Calcula MAPE robusto para totais mensais, tratando meses de valor zero."""
    actual = float(actual)
    predicted = float(predicted)
    if abs(actual) <= EPS:
        return 0.0 if abs(predicted) <= EPS else 1.0
    return abs(actual - predicted) / abs(actual)


def get_train_window(df: pd.DataFrame, cutoff_date: pd.Timestamp, months: int) -> pd.DataFrame:
    """Seleciona janela de treino encerrada antes do mes simulado."""
    start_period = cutoff_date.to_period("M") - months
    end_period = cutoff_date.to_period("M") - 1
    mask = (df["data"].dt.to_period("M") >= start_period) & (df["data"].dt.to_period("M") <= end_period)
    return df.loc[mask].copy()


def make_future_frame(future_dates: pd.DatetimeIndex) -> pd.DataFrame:
    """Cria DataFrame futuro com features de calendario para modelos diarios."""
    future_df = pd.DataFrame({"data": future_dates})
    return add_calendar_features(future_df)


def forecast_methods(
    train_df: pd.DataFrame,
    observed_df: pd.DataFrame,
    future_dates: pd.DatetimeIndex,
    include_croston: bool,
) -> Dict[str, pd.Series]:
    """Executa todos os metodos disponiveis para um produto e corte temporal."""
    future_df = make_future_frame(future_dates)
    forecasts = {
        "curva": forecast_curve_method(train_df, observed_df, future_dates),
        "run_rate": forecast_run_rate_method(train_df, observed_df, future_dates),
        "ridge": forecast_ridge_method(train_df, future_df),
    }
    if include_croston:
        forecasts["croston"] = forecast_croston_method(train_df, observed_df, future_dates)
    return forecasts


def weights_from_method_errors(method_errors: Dict[str, float]) -> Dict[str, float]:
    """Converte erros historicos em pesos normalizados por inverso do erro."""
    usable = {k: v for k, v in method_errors.items() if np.isfinite(v)}
    if not usable:
        return {}
    inv = {k: 1.0 / max(v, 0.01) for k, v in usable.items()}
    total = sum(inv.values())
    if total <= EPS:
        return {k: 1.0 / len(inv) for k in inv}
    return {k: v / total for k, v in inv.items()}


def combine_forecasts(forecasts: Dict[str, pd.Series], weights: Dict[str, float]) -> pd.Series:
    """Combina previsoes diarias usando pesos calibrados."""
    if not forecasts:
        return pd.Series(dtype=float)
    index = next(iter(forecasts.values())).index
    combined = pd.Series(0.0, index=index)
    valid_weight_sum = 0.0
    for method, series in forecasts.items():
        weight = float(weights.get(method, 0.0))
        if weight <= 0:
            continue
        combined = combined.add(series.reindex(index).fillna(0.0) * weight, fill_value=0.0)
        valid_weight_sum += weight
    if valid_weight_sum <= EPS:
        equal = 1.0 / len(forecasts)
        for series in forecasts.values():
            combined = combined.add(series.reindex(index).fillna(0.0) * equal, fill_value=0.0)
        return combined.clip(lower=0.0)
    return (combined / valid_weight_sum).clip(lower=0.0)


def run_backtest_for_product(
    product_df: pd.DataFrame,
    is_intermittent: bool,
    volume_class: str,
    target_period: pd.Period,
) -> Tuple[ProductCalibration, pd.DataFrame]:
    """Executa backtest por produto, escolhe janela e calibra pesos."""
    familia = str(product_df["familia"].iloc[0])
    produto = str(product_df["produto"].iloc[0])
    methods = METHODS_WITH_CROSTON if is_intermittent else METHODS_CORE
    if volume_class == "baixo volume" and not is_intermittent:
        methods = ["curva", "run_rate"]

    historical_months = sorted(product_df.loc[product_df["data"].dt.to_period("M") < target_period, "data"].dt.to_period("M").unique())
    rows = []

    for train_months in TRAIN_WINDOWS_MONTHS:
        for period in historical_months:
            period = pd.Period(period, freq="M")
            if period - train_months < product_df["data"].min().to_period("M"):
                continue
            actual_month_df = product_df[product_df["data"].dt.to_period("M") == period]
            actual_total = float(actual_month_df["valor"].sum())
            for cutoff_day in BACKTEST_CUTOFF_DAYS:
                cutoff_day = min(cutoff_day, period.days_in_month)
                cutoff_date = pd.Timestamp(year=period.year, month=period.month, day=cutoff_day)
                observed_df = product_df[(product_df["data"] >= month_start(period)) & (product_df["data"] <= cutoff_date)].copy()
                train_df = get_train_window(product_df, cutoff_date, train_months)
                future_dates = pd.date_range(cutoff_date + pd.Timedelta(days=1), month_end(period), freq="D")
                if train_df.empty or observed_df.empty or len(future_dates) == 0:
                    continue

                forecasts = forecast_methods(train_df, observed_df, future_dates, include_croston=is_intermittent)
                realized_to_cutoff = float(observed_df["valor"].sum())
                row = {
                    "familia": familia,
                    "produto": produto,
                    "janela_meses": train_months,
                    "periodo": int(period.strftime("%Y%m")),
                    "dia_corte": cutoff_day,
                    "realizado_mes": actual_total,
                }
                for method in methods:
                    pred_future = float(forecasts.get(method, pd.Series(dtype=float)).sum())
                    pred_total = realized_to_cutoff + pred_future
                    row[f"pred_{method}"] = pred_total
                    row[f"mape_{method}"] = safe_mape(actual_total, pred_total)
                rows.append(row)

    bt = pd.DataFrame(rows)
    if bt.empty:
        fallback_weights = {m: 1.0 / len(methods) for m in methods}
        calibration = ProductCalibration(
            familia=familia,
            produto=produto,
            is_intermittent=is_intermittent,
            volume_class=volume_class,
            selected_window=6,
            weights=fallback_weights,
            method_mape={m: np.nan for m in methods},
            ensemble_mape=np.nan,
            dominant_method=max(fallback_weights, key=fallback_weights.get),
            backtest_rows=bt,
        )
        return calibration, bt

    candidates = []
    for train_months, group in bt.groupby("janela_meses"):
        method_errors = {m: float(group[f"mape_{m}"].mean()) for m in methods if f"mape_{m}" in group.columns}
        weights = weights_from_method_errors(method_errors)
        ensemble_errors = []
        for _, row in group.iterrows():
            pred = 0.0
            weight_sum = 0.0
            for method, weight in weights.items():
                col = f"pred_{method}"
                if col in row and pd.notna(row[col]):
                    pred += weight * float(row[col])
                    weight_sum += weight
            pred = pred / weight_sum if weight_sum > EPS else np.nan
            ensemble_errors.append(safe_mape(float(row["realizado_mes"]), pred) if np.isfinite(pred) else np.nan)
        candidates.append(
            {
                "janela_meses": int(train_months),
                "weights": weights,
                "method_mape": method_errors,
                "ensemble_mape": float(np.nanmean(ensemble_errors)),
            }
        )

    best = min(candidates, key=lambda x: x["ensemble_mape"] if np.isfinite(x["ensemble_mape"]) else np.inf)
    weights = best["weights"] or {m: 1.0 / len(methods) for m in methods}
    dominant = max(weights, key=weights.get)

    calibration = ProductCalibration(
        familia=familia,
        produto=produto,
        is_intermittent=is_intermittent,
        volume_class=volume_class,
        selected_window=int(best["janela_meses"]),
        weights=weights,
        method_mape=best["method_mape"],
        ensemble_mape=float(best["ensemble_mape"]),
        dominant_method=dominant,
        backtest_rows=bt[bt["janela_meses"] == int(best["janela_meses"])].copy(),
    )
    return calibration, bt


# =============================================================================
# PROJECAO ATUAL POR PRODUTO
# =============================================================================
#
# Este bloco aplica a calibracao historica ao mes alvo. Para cada produto, o
# motor identifica o ultimo dia disponivel dentro do mes alvo e projeta somente
# os dias restantes ate o fechamento do mes, preservando o realizado ja observado.
#
# A separacao entre realizado e projetado e indispensavel para uso executivo: o
# usuario precisa enxergar o que ja aconteceu e o que ainda e estimativa. O codigo
# tambem gera a base diaria completa, permitindo auditoria da ponte entre valores
# diarios e total mensal.


def recent_trend_label(product_df: pd.DataFrame, target_period: pd.Period) -> Tuple[str, float]:
    """Classifica comportamento recente como aceleracao, desaceleracao ou estabilidade."""
    target_start = month_start(target_period)
    recent = product_df[product_df["data"] < target_start].tail(14)
    previous = product_df[product_df["data"] < target_start].tail(28).head(14)
    recent_mean = float(recent["valor"].mean()) if not recent.empty else 0.0
    previous_mean = float(previous["valor"].mean()) if not previous.empty else 0.0
    if previous_mean <= EPS and recent_mean <= EPS:
        return "estavel em patamar baixo", 0.0
    change = (recent_mean - previous_mean) / max(abs(previous_mean), EPS)
    if change > 0.10:
        return "aceleracao recente", change
    if change < -0.10:
        return "desaceleracao recente", change
    return "estabilidade recente", change


def generate_product_explanation(
    calibration: ProductCalibration,
    product_df: pd.DataFrame,
    target_period: pd.Period,
    realized_month: float,
    projected_total: float,
) -> str:
    """Gera texto de negocio explicando a projecao do produto."""
    trend_label, trend_change = recent_trend_label(product_df, target_period)
    hist_months = product_df[product_df["data"].dt.to_period("M") < target_period].groupby(product_df["data"].dt.to_period("M"))["valor"].sum()
    hist_median = float(hist_months.tail(6).median()) if len(hist_months) else 0.0
    delta_hist = (projected_total - hist_median) / max(abs(hist_median), EPS) if hist_median > EPS else np.nan

    dominant_weight = calibration.weights.get(calibration.dominant_method, 0.0)
    intermittent_text = "foi tratado como intermitente por apresentar muitos dias zerados" if calibration.is_intermittent else "nao foi tratado como intermitente"
    if np.isfinite(delta_hist):
        direction = "acima" if delta_hist >= 0 else "abaixo"
        hist_text = f"A projecao esta {abs(delta_hist):.1%} {direction} da mediana dos ultimos meses historicos"
    else:
        hist_text = "Nao havia historico mensal positivo suficiente para comparar com a mediana recente"

    explanation = (
        f"Produto {calibration.produto}: o metodo com maior peso foi {calibration.dominant_method} "
        f"({dominant_weight:.1%} do ensemble), escolhido por desempenho no backtest. "
        f"A janela selecionada foi de {calibration.selected_window} meses, com MAPE historico estimado de "
        f"{calibration.ensemble_mape:.1%} quando disponivel. O produto {intermittent_text} e foi classificado "
        f"como {calibration.volume_class}. O comportamento recente indica {trend_label} "
        f"({trend_change:.1%} contra a quinzena anterior). {hist_text}. "
        f"O realizado do mes e {realized_month:,.2f} e o total projetado e {projected_total:,.2f}; "
        f"a diferenca decorre da combinacao entre curva intra-mensal, ritmo recente, padrao semanal e, quando aplicavel, Croston."
    )
    return explanation


def project_current_month_for_product(
    product_df: pd.DataFrame,
    calibration: ProductCalibration,
    target_period: pd.Period,
) -> Tuple[pd.DataFrame, Dict[str, object]]:
    """Gera projecao diaria e resumo mensal para um produto no mes alvo."""
    period_start = month_start(target_period)
    period_end = month_end(target_period)
    month_df = product_df[(product_df["data"] >= period_start) & (product_df["data"] <= period_end)].copy()

    if month_df.empty:
        month_df = pd.DataFrame({"data": pd.date_range(period_start, period_end, freq="D")})
        month_df["familia"] = calibration.familia
        month_df["produto"] = calibration.produto
        month_df["valor"] = 0.0
        month_df = add_calendar_features(month_df)

    observed = month_df[month_df["valor"].notna()].copy()
    available_observed = observed[observed["data"] <= product_df["data"].max()].copy()
    available_observed = available_observed[available_observed["data"].dt.to_period("M") == target_period]
    if available_observed.empty:
        cutoff_date = period_start - pd.Timedelta(days=1)
        observed_df = month_df.iloc[0:0].copy()
    else:
        cutoff_date = available_observed["data"].max()
        observed_df = month_df[(month_df["data"] >= period_start) & (month_df["data"] <= cutoff_date)].copy()

    future_dates = pd.date_range(max(cutoff_date + pd.Timedelta(days=1), period_start), period_end, freq="D")
    train_df = get_train_window(product_df, max(cutoff_date, period_start), calibration.selected_window)
    if train_df.empty:
        train_df = product_df[product_df["data"] < period_start].tail(365).copy()

    forecasts = forecast_methods(train_df, observed_df, future_dates, include_croston=calibration.is_intermittent)
    forecasts = {method: series for method, series in forecasts.items() if method in calibration.weights}
    combined_future = combine_forecasts(forecasts, calibration.weights)

    daily = month_df[["familia", "produto", "data", "valor"]].copy()
    daily["tipo"] = np.where(daily["data"] <= cutoff_date, "realizado", "projetado")
    daily["valor_realizado"] = np.where(daily["tipo"] == "realizado", daily["valor"], np.nan)
    daily["valor_projetado"] = np.where(daily["tipo"] == "projetado", daily["data"].map(combined_future).fillna(0.0), np.nan)
    daily["valor_final"] = np.where(daily["tipo"] == "realizado", daily["valor_realizado"], daily["valor_projetado"])
    daily["anomes"] = int(target_period.strftime("%Y%m"))

    realized_month = float(daily.loc[daily["tipo"] == "realizado", "valor_realizado"].sum())
    projected_future = float(daily.loc[daily["tipo"] == "projetado", "valor_projetado"].sum())
    projected_total = realized_month + projected_future

    explanation = generate_product_explanation(calibration, product_df, target_period, realized_month, projected_total)
    calibration.explanation = explanation

    summary = {
        "familia": calibration.familia,
        "produto": calibration.produto,
        "realizado_mes": realized_month,
        "projetado": projected_total,
        "projetado_restante": projected_future,
        "erro_historico_backtest": calibration.ensemble_mape,
        "metodo_dominante": calibration.dominant_method,
        "janela_escolhida_meses": calibration.selected_window,
        "intermitente": calibration.is_intermittent,
        "classe_volume": calibration.volume_class,
        "explicacao": explanation,
    }
    return daily, summary


# =============================================================================
# HIERARQUIA E METRICAS
# =============================================================================
#
# Este bloco garante consistencia hierarquica entre produto e familia. Primeiro
# projetamos produtos, que sao o menor nivel de decisao; depois agregamos por
# familia. Essa estrategia garante automaticamente que a soma dos produtos seja
# igual ao total da familia.
#
# A reconciliacao explicita e relevante em rotinas executivas porque divergencias
# entre dashboards de produto e familia geram perda de confianca. Mesmo quando a
# familia e exibida como visao propria, ela deve ser a soma auditavel dos itens
# que a compoem.


def reconcile_family(daily_df: pd.DataFrame, final_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Agrega produto para familia mantendo soma reconciliada."""
    family_daily = (
        daily_df.groupby(["familia", "data", "anomes", "tipo"], as_index=False)[["valor", "valor_realizado", "valor_projetado", "valor_final"]]
        .sum()
    )
    family_daily["produto"] = "__TOTAL_FAMILIA__"
    daily_reconciled = pd.concat([daily_df, family_daily[daily_df.columns]], ignore_index=True, sort=False)

    family_final = (
        final_df.groupby("familia", as_index=False)
        .agg(
            realizado_mes=("realizado_mes", "sum"),
            projetado=("projetado", "sum"),
            projetado_restante=("projetado_restante", "sum"),
            erro_historico_backtest=("erro_historico_backtest", "mean"),
        )
    )
    family_final["produto"] = "__TOTAL_FAMILIA__"
    family_final["metodo_dominante"] = "reconciliado_por_soma_produtos"
    family_final["janela_escolhida_meses"] = np.nan
    family_final["intermitente"] = False
    family_final["classe_volume"] = "familia"
    family_final["explicacao"] = "Total de familia reconciliado como soma exata das projecoes dos produtos."

    final_reconciled = pd.concat([final_df, family_final[final_df.columns]], ignore_index=True, sort=False)
    return daily_reconciled, final_reconciled


def build_metrics(backtest_df: pd.DataFrame, final_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Gera metricas de acuracia por produto e por familia."""
    if backtest_df.empty:
        product_metrics = final_df[final_df["produto"] != "__TOTAL_FAMILIA__"][["familia", "produto", "erro_historico_backtest"]].copy()
        product_metrics = product_metrics.rename(columns={"erro_historico_backtest": "mape"})
        family_metrics = final_df[final_df["produto"] == "__TOTAL_FAMILIA__"][["familia", "erro_historico_backtest"]].copy()
        family_metrics = family_metrics.rename(columns={"erro_historico_backtest": "mape"})
        return product_metrics, family_metrics

    product_rows = []
    for (familia, produto), group in backtest_df.groupby(["familia", "produto"]):
        mape_cols = [col for col in group.columns if col.startswith("mape_")]
        product_rows.append({"familia": familia, "produto": produto, "mape": float(group[mape_cols].mean(axis=1).mean())})
    product_metrics = pd.DataFrame(product_rows)

    family_rows = []
    pred_cols = [col for col in backtest_df.columns if col.startswith("pred_")]
    if pred_cols:
        method = pred_cols[0].replace("pred_", "")
        fam_group_cols = ["familia", "periodo", "dia_corte", "janela_meses"]
        fam = backtest_df.groupby(fam_group_cols, as_index=False).agg(realizado_mes=("realizado_mes", "sum"), pred=(f"pred_{method}", "sum"))
        fam["mape"] = fam.apply(lambda r: safe_mape(r["realizado_mes"], r["pred"]), axis=1)
        family_metrics = fam.groupby("familia", as_index=False)["mape"].mean()
    else:
        family_metrics = pd.DataFrame(family_rows, columns=["familia", "mape"])
    return product_metrics, family_metrics


# =============================================================================
# ORQUESTRACAO DO MOTOR
# =============================================================================
#
# Este bloco conecta todas as etapas: normalizacao, classificacao, backtest,
# calibracao, projecao e reconciliacao. Ele e a principal funcao para uso em
# batch, testes e Streamlit, reduzindo duplicidade de logica entre interfaces.
#
# A orquestracao por produto torna o motor escalavel para mais de 20 produtos e
# evita ajustes manuais. Cada produto aprende sua propria configuracao historica,
# mas as saidas seguem um schema unico para facilitar consumo em relatorios e
# dashboards.


def run_projection_engine(csv_path: str, anomes: int | str) -> ProjectionArtifacts:
    """Executa o motor completo e retorna todos os artefatos de saida."""
    target_period = parse_anomes(anomes)
    normalized = load_and_normalize(csv_path)

    daily_outputs = []
    summary_outputs = []
    backtests = []
    explanations = []

    for (familia, produto), product_df in normalized.groupby(["familia", "produto"], sort=True):
        product_df = product_df.sort_values("data").reset_index(drop=True)
        is_intermittent, volume_class, _ = classify_product(product_df[product_df["data"].dt.to_period("M") < target_period])
        calibration, bt = run_backtest_for_product(product_df, is_intermittent, volume_class, target_period)
        daily, summary = project_current_month_for_product(product_df, calibration, target_period)

        daily_outputs.append(daily)
        summary_outputs.append(summary)
        if not bt.empty:
            backtests.append(bt)
        explanations.append({"familia": familia, "produto": produto, "explicacao": calibration.explanation})

    daily_df = pd.concat(daily_outputs, ignore_index=True) if daily_outputs else pd.DataFrame()
    final_df = pd.DataFrame(summary_outputs)
    backtest_df = pd.concat(backtests, ignore_index=True) if backtests else pd.DataFrame()

    if not daily_df.empty and not final_df.empty:
        daily_df, final_df = reconcile_family(daily_df, final_df)

    product_metrics, family_metrics = build_metrics(backtest_df, final_df)
    explanations_df = pd.DataFrame(explanations)

    return ProjectionArtifacts(
        final_df=final_df,
        daily_df=daily_df,
        product_metrics=product_metrics,
        family_metrics=family_metrics,
        backtest_df=backtest_df,
        explanations=explanations_df,
    )


# =============================================================================
# EXPORTACAO
# =============================================================================
#
# Este bloco salva as principais saidas em CSV quando o script e executado em
# modo batch. Em ambiente corporativo, arquivos tabulares simples facilitam a
# integracao com Excel, Power BI, pipelines legados e rotinas de validacao.
#
# A exportacao separa resumo mensal, base diaria e metricas de acuracia. Essa
# separacao preserva granularidade para auditoria sem obrigar consumidores
# executivos a lidar com tabelas excessivamente detalhadas.


def export_artifacts(artifacts: ProjectionArtifacts, output_dir: str) -> None:
    """Exporta artefatos do motor para CSV."""
    os.makedirs(output_dir, exist_ok=True)
    artifacts.final_df.to_csv(os.path.join(output_dir, "projecao_final.csv"), index=False)
    artifacts.daily_df.to_csv(os.path.join(output_dir, "base_diaria_realizado_projetado.csv"), index=False)
    artifacts.product_metrics.to_csv(os.path.join(output_dir, "mape_por_produto.csv"), index=False)
    artifacts.family_metrics.to_csv(os.path.join(output_dir, "mape_por_familia.csv"), index=False)
    artifacts.backtest_df.to_csv(os.path.join(output_dir, "backtest_detalhado.csv"), index=False)
    artifacts.explanations.to_csv(os.path.join(output_dir, "explicacoes_produto.csv"), index=False)


# =============================================================================
# VISUALIZACOES STREAMLIT
# =============================================================================
#
# Este bloco define graficos executivos com Plotly. As regras visuais sao
# mantidas de forma consistente: realizado em preto forte, projetado em cinza
# claro e backtest em azul tracejado. O fundo branco e o grid leve reduzem ruido
# visual e favorecem leitura em comites.
#
# A camada visual nao recalibra o modelo; ela apenas consome os artefatos gerados
# pelo motor. Essa separacao e importante porque a previsao deve ser reproduzivel
# independentemente de estar sendo rodada via batch ou pela interface Streamlit.


def apply_executive_layout(fig: go.Figure, title: str) -> go.Figure:
    """Aplica layout clean e executivo aos graficos."""
    fig.update_layout(
        title=title,
        template="plotly_white",
        paper_bgcolor="white",
        plot_bgcolor="white",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=30, r=30, t=60, b=30),
    )
    fig.update_xaxes(showgrid=True, gridcolor="rgba(0,0,0,0.06)")
    fig.update_yaxes(showgrid=True, gridcolor="rgba(0,0,0,0.08)", zeroline=False)
    return fig


def daily_line_chart(daily_df: pd.DataFrame, title: str, backtest_df: Optional[pd.DataFrame] = None) -> go.Figure:
    """Cria grafico diario com realizado, projetado e linha de backtest quando disponivel."""
    fig = go.Figure()
    realized = daily_df[daily_df["tipo"] == "realizado"]
    projected = daily_df[daily_df["tipo"] == "projetado"]

    fig.add_trace(go.Scatter(x=realized["data"], y=realized["valor_realizado"], mode="lines+markers", name="Realizado", line=dict(color="black", width=3)))
    fig.add_trace(go.Scatter(x=projected["data"], y=projected["valor_projetado"], mode="lines+markers", name="Projetado", line=dict(color="lightgray", width=3)))

    if backtest_df is not None and not backtest_df.empty:
        bt = backtest_df.copy()
        latest_period = bt["periodo"].max()
        bt = bt[bt["periodo"] == latest_period]
        if not bt.empty:
            x = [pd.Timestamp(str(int(latest_period)) + "01") + pd.Timedelta(days=int(d) - 1) for d in bt["dia_corte"]]
            y_cols = [col for col in bt.columns if col.startswith("pred_")]
            if y_cols:
                fig.add_trace(go.Scatter(x=x, y=bt[y_cols[0]], mode="lines+markers", name="Backtest M-1", line=dict(color="royalblue", width=2, dash="dash")))

    return apply_executive_layout(fig, title)


def monthly_line_chart(normalized_df: pd.DataFrame, daily_projection: pd.DataFrame, title: str) -> go.Figure:
    """Cria grafico mensal historico com inclusao do mes projetado."""
    hist = normalized_df.copy()
    hist["periodo"] = hist["data"].dt.to_period("M").dt.to_timestamp()
    monthly = hist.groupby("periodo", as_index=False)["valor"].sum().rename(columns={"valor": "realizado"})

    proj_period = pd.to_datetime(daily_projection["data"]).dt.to_period("M").iloc[0].to_timestamp()
    proj_total = float(daily_projection["valor_final"].sum())
    monthly_proj = pd.DataFrame({"periodo": [proj_period], "projetado": [proj_total]})

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=monthly["periodo"], y=monthly["realizado"], mode="lines+markers", name="Realizado", line=dict(color="black", width=3)))
    fig.add_trace(go.Scatter(x=monthly_proj["periodo"], y=monthly_proj["projetado"], mode="markers+lines", name="Projetado", line=dict(color="lightgray", width=3)))
    return apply_executive_layout(fig, title)


def error_ranking_chart(product_metrics: pd.DataFrame) -> go.Figure:
    """Cria grafico de erro por produto e ranking de previsibilidade."""
    df = product_metrics.dropna(subset=["mape"]).sort_values("mape", ascending=True).copy()
    fig = go.Figure()
    fig.add_trace(go.Bar(x=df["produto"], y=df["mape"], marker_color="royalblue", name="MAPE"))
    fig.update_yaxes(tickformat=".1%")
    return apply_executive_layout(fig, "Erro historico por produto (MAPE)")


# =============================================================================
# APP STREAMLIT
# =============================================================================
#
# Este bloco implementa a interface requerida. O app permite carregar CSV,
# informar o anomes alvo, rodar o motor e navegar por visao familia, visao
# produto, explicabilidade e acuracia visual.
#
# A linguagem exibida no app e orientada a negocio: alem de tabelas e graficos,
# cada produto traz uma explicacao textual sobre metodo dominante, janela
# escolhida, intermitencia e motivo da diferenca contra historico. Isso reduz a
# dependencia de leitura tecnica do codigo para interpretar a projecao.


def run_streamlit_app() -> None:
    """Executa app Streamlit embutido no mesmo arquivo."""
    import streamlit as st

    st.set_page_config(page_title="Motor de Projecao Mensal", layout="wide")
    st.title("Motor de Projecao Mensal")
    st.caption("Projecao baseada exclusivamente em historico, com backtest, pesos calibrados e reconciliacao hierarquica.")

    with st.sidebar:
        st.header("Parametros")
        uploaded = st.file_uploader("CSV de entrada", type=["csv"])
        anomes = st.text_input("Anomes alvo (yyyymm)", value="202604")
        run_button = st.button("Rodar projecao", type="primary")

    if not uploaded:
        st.info("Carregue um CSV com as colunas anomesdia, familia, produto e valor para iniciar.")
        return

    temp_path = os.path.join("/tmp", f"input_projecao_{abs(hash(uploaded.name))}.csv")
    with open(temp_path, "wb") as f:
        f.write(uploaded.getbuffer())

    if run_button or "artifacts" not in st.session_state:
        with st.spinner("Executando motor de projecao, backtest e reconciliacao..."):
            st.session_state["artifacts"] = run_projection_engine(temp_path, anomes)
            st.session_state["normalized"] = load_and_normalize(temp_path)

    artifacts: ProjectionArtifacts = st.session_state["artifacts"]
    normalized = st.session_state["normalized"]

    tab_familia, tab_produto, tab_acuracia, tab_tabelas = st.tabs(["Visao familia", "Visao produto", "Acuracia", "Tabelas"])

    with tab_familia:
        familias = sorted(artifacts.final_df["familia"].dropna().unique().tolist())
        familia = st.selectbox("Familia", familias, key="familia_select")
        fam_daily = artifacts.daily_df[(artifacts.daily_df["familia"] == familia) & (artifacts.daily_df["produto"] == "__TOTAL_FAMILIA__")]
        fam_hist = normalized[normalized["familia"] == familia]
        st.plotly_chart(daily_line_chart(fam_daily, f"Diario - Familia {familia}"), use_container_width=True)
        st.plotly_chart(monthly_line_chart(fam_hist, fam_daily, f"Mensal - Familia {familia}"), use_container_width=True)

    with tab_produto:
        base_prod = artifacts.final_df[artifacts.final_df["produto"] != "__TOTAL_FAMILIA__"]
        produtos = sorted(base_prod["produto"].dropna().unique().tolist())
        produto = st.selectbox("Produto", produtos, key="produto_select")
        prod_row = base_prod[base_prod["produto"] == produto].iloc[0]
        prod_daily = artifacts.daily_df[artifacts.daily_df["produto"] == produto]
        prod_hist = normalized[normalized["produto"] == produto]
        prod_bt = artifacts.backtest_df[artifacts.backtest_df["produto"] == produto] if not artifacts.backtest_df.empty else pd.DataFrame()

        c1, c2, c3 = st.columns(3)
        c1.metric("Realizado no mes", f"{prod_row['realizado_mes']:,.2f}")
        c2.metric("Projetado total", f"{prod_row['projetado']:,.2f}")
        c3.metric("MAPE backtest", "n/d" if pd.isna(prod_row["erro_historico_backtest"]) else f"{prod_row['erro_historico_backtest']:.1%}")

        st.plotly_chart(daily_line_chart(prod_daily, f"Diario - Produto {produto}", prod_bt), use_container_width=True)
        st.plotly_chart(monthly_line_chart(prod_hist, prod_daily, f"Mensal - Produto {produto}"), use_container_width=True)

        st.subheader("Explicacao da projecao")
        st.write(str(prod_row["explicacao"]))

    with tab_acuracia:
        st.plotly_chart(error_ranking_chart(artifacts.product_metrics), use_container_width=True)
        ranked = artifacts.product_metrics.dropna(subset=["mape"]).sort_values("mape")
        col_a, col_b = st.columns(2)
        col_a.subheader("Mais previsiveis")
        col_a.dataframe(ranked.head(10), use_container_width=True)
        col_b.subheader("Menos previsiveis")
        col_b.dataframe(ranked.tail(10).sort_values("mape", ascending=False), use_container_width=True)
        st.subheader("MAPE por familia")
        st.dataframe(artifacts.family_metrics.sort_values("mape"), use_container_width=True)

    with tab_tabelas:
        st.subheader("DataFrame final")
        st.dataframe(artifacts.final_df, use_container_width=True)
        st.subheader("Base diaria completa")
        st.dataframe(artifacts.daily_df, use_container_width=True)
        st.subheader("Backtest detalhado")
        st.dataframe(artifacts.backtest_df, use_container_width=True)


# =============================================================================
# CLI E PONTO DE ENTRADA
# =============================================================================
#
# Este bloco permite rodar o mesmo arquivo tanto com Streamlit quanto em batch.
# Quando chamado por `streamlit run`, a aplicacao visual e aberta; quando chamado
# por `python`, argumentos de linha de comando permitem gerar CSVs de saida.
#
# Ter uma unica entrada reduz risco de divergencia entre ambiente analitico e
# ambiente executivo. O mesmo motor que alimenta o dashboard e o que exporta as
# bases, preservando consistencia de numeros e explicacoes.


def parse_args() -> argparse.Namespace:
    """Parseia argumentos do modo batch."""
    parser = argparse.ArgumentParser(description="Motor de projecao mensal com backtest e Streamlit.")
    parser.add_argument("--csv", required=False, help="Caminho do CSV de entrada.")
    parser.add_argument("--anomes", required=False, default="202604", help="Mes alvo no formato yyyymm.")
    parser.add_argument("--output-dir", required=False, default="saida_projecao", help="Diretorio para salvar CSVs de saida.")
    return parser.parse_args()


def running_inside_streamlit() -> bool:
    """Detecta execucao dentro do runtime do Streamlit."""
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx

        return get_script_run_ctx() is not None
    except Exception:
        return False


def main() -> None:
    """Ponto de entrada do arquivo."""
    if running_inside_streamlit():
        run_streamlit_app()
        return

    args = parse_args()
    if not args.csv:
        print("Uso batch: python motor_projecao_mensal.py --csv caminho/arquivo.csv --anomes 202604 --output-dir saida")
        print("Uso app:   streamlit run motor_projecao_mensal.py")
        return

    artifacts = run_projection_engine(args.csv, args.anomes)
    export_artifacts(artifacts, args.output_dir)
    print(f"Projecao concluida. Arquivos salvos em: {args.output_dir}")
    print(artifacts.final_df.head(20).to_string(index=False))


if __name__ == "__main__":
    main()
