# What's missing:
# - [x] select the metric.
# - [ ] sort out pre-filtering logic.
# first, check the data for percentage is correct
# then, sort out the filter.

import os

import plotly.express as px  # type: ignore[attr-defined]
import polars as pl
import psutil
import streamlit as st

st.set_page_config(layout="wide")


def get_memory_usage():
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    return mem_info.rss / (1024**2)  # RSS in MB


# Load the data
# @st.cache_resource
def load_data():
    return (
        pl.scan_parquet("big_files/matches.parquet")  # pyright: ignore[reportUnknownMemberType]
        .select(
            "is_open_data",
            "is_open_code",
            "affiliation_country",
            "journal",
            "funder",
            "year",
        )
        .filter(pl.col("year") >= 2000)
        .collect()
    )


@st.cache_resource
def load_data_for_funder():
    return (
        pl.scan_parquet("big_files/matches.parquet")  # pyright: ignore[reportUnknownMemberType]
        .select(
            "is_open_data",
            "is_open_code",
            "affiliation_country",
            "journal",
            "funder",
            "year",
        )
        .filter(pl.col("year") >= 2000)
        .with_columns(pl.col("funder").list.unique())
        .explode("funder")
        .filter(pl.col("funder").str.len_chars() > 0, pl.col("funder").is_not_null())
        .collect()
    )


@st.cache_resource
def load_data_for_country():
    return (
        pl.scan_parquet("big_files/matches.parquet")  # pyright: ignore[reportUnknownMemberType]
        .select(
            "is_open_data",
            "is_open_code",
            "affiliation_country",
            "journal",
            "funder",
            "year",
        )
        .filter(pl.col("year") >= 2000, pl.col("affiliation_country").is_not_null())
        .with_columns(pl.col("affiliation_country").str.split("; ").list.unique())
        .explode("affiliation_country")
        .filter(
            pl.col("affiliation_country").str.len_chars() > 0,
            pl.col("affiliation_country").is_not_null(),
        )
        .collect()
    )


data = load_data()
data_for_country = load_data_for_country()
data_for_funder = load_data_for_funder()


# Dropdown selection
group_option = st.selectbox(
    "Splitting variable",
    options=[None, "journal", "affiliation_country", "funder"],
    index=3,
)

if group_option != "journal":
    unique_journals = data["journal"].unique(maintain_order=True).to_list()
    journals = st.multiselect("Journal", options=unique_journals)
else:
    journals = []

if group_option != "affiliation_country":
    unique_countries = (
        data_for_country["affiliation_country"].unique(maintain_order=True).to_list()
    )
    countries = st.multiselect("Country", options=unique_countries)
else:
    countries = []

unique_funders = data_for_funder["funder"].unique(maintain_order=True).to_list()
if group_option == "funder":
    default = (
        data_for_funder.group_by("funder")
        .len()
        .select(pl.col("funder").top_k_by("len", 10))["funder"]
        .to_list()
    )
else:
    default = []
funders = st.multiselect("Funder", options=unique_funders, default=default)

max_year: int = data["year"].max()  # type: ignore[assignment]
years: tuple[int, int] = st.slider(  # type: ignore[assignment]
    "Years", min_value=2000, max_value=max_year, value=(2000, max_year)
)

aggregation_name = st.selectbox(
    "Aggregation",
    options=[
        "data_sharing_percent",
        "data_sharing",
        "count",
        "code_sharing_percent",
        "code_sharing",
    ],
)

FORMULAE = {
    "data_sharing_percent": pl.col("is_open_data").mean() * 100,
    "data_sharing": pl.col("is_open_data").sum(),
    "count": pl.col("is_open_data").len(),
    "code_sharing_percent": pl.col("is_open_code").mean() * 100,
    "code_sharing": pl.col("is_open_code").sum(),
}
formula = FORMULAE[aggregation_name]


def filter(df: pl.DataFrame) -> pl.DataFrame:
    df = df.filter(pl.col("year").is_between(*years, closed="both"))

    if journals:
        df = df.filter(pl.col("journal").is_in(journals))
    if countries:
        df = df.filter(
            pl.any_horizontal(
                [
                    pl.col("affiliation_country").str.split("; ").list.contains(x)
                    for x in countries
                ]
            )
        )
    if funders:
        if group_option == "funder":
            df = df.filter(pl.col("funder").is_in(funders))
        else:
            df = df.filter(
                pl.any_horizontal([pl.col("funder").list.contains(x) for x in funders])
            )
    return df


def keep_and_sort_top_data(df: pl.DataFrame, group_option: str) -> pl.DataFrame:
    top = (
        df.group_by(group_option)
        .agg(pl.col(aggregation_name).last())
        .sort(aggregation_name)
        .tail(20)
    )
    if group_option == "funder":
        # Special logic to ensure Howard Medical School always appears
        df = df.filter(
            pl.col("funder").is_in(
                top["funder"].unique().to_list() + ["Howard Hughes Medical Institute"]
            )
        )
    else:
        df = df.join(top, on=group_option, how="semi").sort("year", aggregation_name)
    df = df.sort("year", group_option)

    return df


# Plotting logic
if group_option is None:
    df = data

    df = filter(df)

    summary = df.group_by("year").agg(formula.alias(aggregation_name)).sort("year")
    fig = px.line(  # pyright: ignore[reportUnknownMemberType]
        summary,
        x="year",
        y=aggregation_name,
        title="Open Data Over Time",
    )

elif group_option == "journal":
    df = data

    df = filter(df)

    summary = df.group_by(group_option, "year").agg(formula.alias(aggregation_name))

    summary = keep_and_sort_top_data(summary, group_option)

    fig = px.line(  # pyright: ignore[reportUnknownMemberType]
        summary,
        x="year",
        y=aggregation_name,
        color=group_option,
        title=f"Open Data by {group_option.title()} Over Time",
    )
else:
    if group_option == "funder":
        df = data_for_funder
    else:
        df = data_for_country

    df = filter(df)

    summary = (
        df.select("is_open_data", "year", group_option)
        .group_by(group_option, "year")
        .agg(formula.alias(aggregation_name))
    )
    summary = keep_and_sort_top_data(summary, group_option)

    fig = px.line(  # pyright: ignore[reportUnknownMemberType]
        summary,
        x="year",
        y=aggregation_name,
        color=group_option,
        title=f"Open Data by {group_option.title()} Over Time",
    )

st.plotly_chart(fig, use_container_width=True)  # pyright: ignore[reportUnknownMemberType]

mem_usage = get_memory_usage()
st.write(f"**Memory usage:** {mem_usage:.2f} MB")
