import os

import plotly.express as px
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
        pl.scan_parquet("big_files/matches.parquet")
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
        pl.scan_parquet("big_files/matches.parquet")
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
        pl.scan_parquet("big_files/matches.parquet")
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

if group_option != "journals":
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


def filter(df):
    if journals:
        df = df.filter(pl.col("journal").is_in(journals))
    if countries:
        df = df.filter(
            pl.any_horizontal(
                [pl.col("affiliation_country").str.contains(x) for x in countries]
            )
        )
    return df


# Plotting logic
if group_option is None:
    df = data

    df = filter(df)

    summary = (
        df.group_by("year")
        .agg(pl.col("is_open_data").sum().alias("open_data_count"))
        .sort("year")
    )
    fig = px.line(
        summary,
        x="year",
        y="open_data_count",
        title="Open Data Over Time",
    )

elif group_option == "journal":
    df = data

    df = filter(df)

    summary = (
        df.group_by(group_option, "year")
        .agg(data_sharing=pl.col("is_open_data").sum())
        .sort("year")
    )
    top = (
        summary.group_by(group_option)
        .agg(pl.col("data_sharing").last())
        .sort("data_sharing")
        .tail(10)
    )
    summary = summary.join(top, on=group_option, how="semi").sort(
        "year", "data_sharing"
    )
    fig = px.line(
        summary,
        x="year",
        y="data_sharing",
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
        .agg(data_sharing=pl.col("is_open_data").sum())
        .sort(group_option, "year")
    )
    top = (
        summary.group_by(group_option)
        .agg(pl.col("data_sharing").last())
        .sort("data_sharing")
        .tail(10)
    )

    if group_option == "funder":
        summary = summary.filter(
            pl.col("funder").is_in(
                top["funder"].unique().to_list() + ["Howard Hughes Medical Institute"]
            )
        )
    else:
        summary = summary.filter(
            pl.col(group_option).is_in(top[group_option].unique().to_list())
        )

    summary = summary.sort(pl.col("data_sharing").last().over(group_option))
    fig = px.line(
        summary,
        x="year",
        y="data_sharing",
        color=group_option,
        title=f"Open Data by {group_option.title()} Over Time",
        category_orders={
            group_option: summary[group_option]
            .unique(maintain_order=True)
            .to_list()[::-1]
        },
    )

st.plotly_chart(fig, use_container_width=True)

mem_usage = get_memory_usage()
st.write(f"**Memory usage:** {mem_usage:.2f} MB")
