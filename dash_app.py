import plotly.express as px  # type: ignore[attr-defined]
import polars as pl
import streamlit as st

st.set_page_config(layout="wide")


@st.cache_resource
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


splitting_variable = st.selectbox(
    "Splitting variable",
    options=[None, "journal", "affiliation_country", "funder"],
    index=3,  # default to 'funder'
)

unique_journals = data["journal"].unique(maintain_order=True).to_list()
if splitting_variable == "journal":
    default_journals = (
        data.group_by("journal")
        .len()
        .select(pl.col("journal").top_k_by("len", 10))["journal"]
        .to_list()
    )
else:
    default_journals = []
journals = st.multiselect("Journal", options=unique_journals, default=default_journals)

unique_countries = (
    data_for_country["affiliation_country"].unique(maintain_order=True).to_list()
)
if splitting_variable == "affiliation_country":
    default_countries = (
        data_for_country.group_by("affiliation_country")
        .len()
        .select(pl.col("affiliation_country").top_k_by("len", 10))[
            "affiliation_country"
        ]
        .to_list()
    )
else:
    default_countries = []
countries = st.multiselect(
    "Country", options=unique_countries, default=default_countries
)

unique_funders = data_for_funder["funder"].unique(maintain_order=True).to_list()
if splitting_variable == "funder":
    # Ensure that Howard Hughes Medical Institute always appears.
    default_funders = data_for_funder.group_by("funder").len().select(
        pl.col("funder").top_k_by("len", 9)
    )["funder"].to_list() + ["Howard Hughes Medical Institute"]
else:
    default_funders = []
funders = st.multiselect("Funder", options=unique_funders, default=default_funders)

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
        if splitting_variable == "affiliation_country":
            # 'affiliation_country' has already been preprocessed, so we can just use `is_in`.
            df = df.filter(pl.col("affiliation_country").is_in(countries))
        else:
            df = df.filter(
                pl.any_horizontal(
                    [
                        pl.col("affiliation_country").str.split("; ").list.contains(x)
                        for x in countries
                    ]
                )
            )
    if funders:
        if splitting_variable == "funder":
            # 'funder' has already been preprocessed, so we can just use `is_in`.
            df = df.filter(pl.col("funder").is_in(funders))
        else:
            df = df.filter(
                pl.any_horizontal([pl.col("funder").list.contains(x) for x in funders])
            )
    return df


if splitting_variable is None:
    df = filter(data)
    summary = df.group_by("year").agg(formula.alias(aggregation_name)).sort("year")
    fig = px.line(  # pyright: ignore[reportUnknownMemberType]
        summary,
        x="year",
        y=aggregation_name,
        title="Open Data Over Time",
    )

elif splitting_variable == "journal":
    df = filter(data)
    summary = (
        df.group_by(splitting_variable, "year")
        .agg(formula.alias(aggregation_name))
        .sort("year", splitting_variable)
    )
    fig = px.line(  # pyright: ignore[reportUnknownMemberType]
        summary,
        x="year",
        y=aggregation_name,
        color=splitting_variable,
        title=f"Open Data by {splitting_variable.title()} Over Time",
    )
else:
    if splitting_variable == "funder":
        df = data_for_funder
    else:
        df = data_for_country
    df = filter(df)
    summary = (
        df.select("is_open_data", "year", "is_open_code", splitting_variable)
        .group_by(splitting_variable, "year")
        .agg(formula.alias(aggregation_name))
        .sort("year", splitting_variable)
    )
    fig = px.line(  # pyright: ignore[reportUnknownMemberType]
        summary,
        x="year",
        y=aggregation_name,
        color=splitting_variable,
        title=f"Open Data by {splitting_variable.title()} Over Time",
    )

fig.update_layout(hovermode="x unified")  # pyright: ignore[reportUnknownMemberType]
fig.update_traces(hovertemplate="%{y}")  # pyright: ignore[reportUnknownMemberType]
st.plotly_chart(fig, use_container_width=True)  # pyright: ignore[reportUnknownMemberType]
