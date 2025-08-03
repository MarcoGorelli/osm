import plotly.express as px
import polars as pl
import streamlit as st

st.set_page_config(layout="wide")


# Load the data
@st.cache_resource
def load_data():
    return (
        pl.scan_parquet("big_files/matches.parquet")
        .filter(pl.col("year") >= 2000)
        .collect()
    )


df = load_data()

# Dropdown selection
group_option = st.selectbox(
    "Splitting variable", options=[None, "journal", "affiliation_country", "funder"]
)

# Plotting logic
if group_option is None:
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
    summary = (
        df.select("is_open_data", "year", group_option)
        .filter(
            pl.col(group_option).list.eval(pl.element().str.len_chars()).list.max() > 0
        )
        .with_columns(pl.col(group_option).list.unique())
        .explode(group_option)
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
    summary = summary.join(top, on=group_option, how="semi").sort(
        "year", "data_sharing"
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
