from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

# Mengatur tampilan halaman dashboard
st.set_page_config(page_title="Dashboard E-Commerce", layout="wide")

# Mengatur gaya dan warna visualisasi, sama seperti di notebook
# Tanda $ ditulis \\$ agar matplotlib tidak membacanya sebagai rumus
sns.set_style("white")
plt.rcParams["axes.spines.top"] = False
plt.rcParams["axes.spines.right"] = False
main_color = "#2a78d6"
second_color = "#eb6834"
base_color = "#D3D3D3"
line_color = "#52514e"


def format_number(value, decimals=0):
    """Mengubah angka ke format Indonesia, misalnya 7289 menjadi 7.289."""
    text = f"{value:,.{decimals}f}"
    return text.replace(",", "X").replace(".", ",").replace("X", ".")


# Helper function untuk menyiapkan DataFrame setiap visualisasi
# create_monthly_orders_df() menyiapkan jumlah pesanan dan revenue per bulan (Pertanyaan 1)
def create_monthly_orders_df(df):
    monthly_orders_df = df.drop_duplicates(subset="order_id").resample(rule="ME", on="order_purchase_timestamp").agg({
        "order_id": "nunique",
        "payment_value": "sum"
    })
    monthly_orders_df.index = monthly_orders_df.index.strftime("%Y-%m")
    monthly_orders_df = monthly_orders_df.reset_index()
    monthly_orders_df.rename(columns={
        "order_purchase_timestamp": "order_month",
        "order_id": "order_count",
        "payment_value": "revenue"
    }, inplace=True)
    return monthly_orders_df


# create_category_df() menyiapkan unit terjual, revenue, dan harga rata-rata per kategori (Pertanyaan 2)
def create_category_df(df):
    category_df = df.groupby(by="product_category_name_english").agg({
        "order_id": "nunique",
        "order_item_id": "count",
        "price": ["sum", "mean"]
    })
    category_df.columns = ["order_count", "unit_sold", "revenue", "avg_price"]
    category_df = category_df.reset_index().sort_values(by=["unit_sold", "product_category_name_english"], ascending=[False, True])
    return category_df


# create_bystate_df() menyiapkan jumlah pelanggan, lama pengiriman, ongkos kirim, dan nilai pesanan per negara bagian (Pertanyaan 3)
def create_bystate_df(df):
    order_level = df.drop_duplicates(subset="order_id")
    bystate_df = order_level.groupby(by="customer_state").agg({
        "customer_unique_id": "nunique",
        "delivery_time": "mean",
        "payment_value": "mean"
    })
    bystate_df["avg_freight_per_item"] = df.groupby(by="customer_state").freight_value.mean()
    bystate_df.rename(columns={
        "customer_unique_id": "customer_count",
        "delivery_time": "avg_delivery_time",
        "payment_value": "avg_order_value"
    }, inplace=True)
    bystate_df["customer_share"] = bystate_df["customer_count"] / df["customer_unique_id"].nunique() * 100
    bystate_df = bystate_df.sort_values(by="customer_count", ascending=False).reset_index()
    return bystate_df


# create_seller_state_df() menyiapkan jumlah seller aktif per negara bagian (Pertanyaan 3)
def create_seller_state_df(df):
    active_sellers_df = df.drop_duplicates(subset="seller_id")
    seller_state_df = active_sellers_df.groupby(by="seller_state").seller_id.nunique().reset_index()
    seller_state_df.rename(columns={"seller_id": "seller_count"}, inplace=True)
    seller_state_df["seller_share"] = seller_state_df["seller_count"] / len(active_sellers_df) * 100
    return seller_state_df


# create_rfm_df() menyiapkan recency, frequency, dan monetary setiap pelanggan (Pertanyaan 4, 5, dan 6)
def create_rfm_df(df, recent_date):
    rfm_df = df.drop_duplicates(subset="order_id").groupby(by="customer_unique_id", as_index=False).agg({
        "order_purchase_timestamp": "max",
        "order_id": "nunique",
        "payment_value": "sum"
    })
    rfm_df.columns = ["customer_unique_id", "max_order_timestamp", "frequency", "monetary"]
    rfm_df["max_order_timestamp"] = rfm_df["max_order_timestamp"].dt.date
    rfm_df["recency"] = rfm_df["max_order_timestamp"].apply(lambda x: (recent_date - x).days)
    rfm_df.drop("max_order_timestamp", axis=1, inplace=True)
    return rfm_df


# Memuat all_data.csv dari folder yang sama dengan dashboard.py
# st.cache_data menyimpan hasil pembacaan agar data tidak dibaca ulang setiap filter diubah
@st.cache_data
def load_data():
    df = pd.read_csv(Path(__file__).parent / "all_data.zip")
    df["order_purchase_timestamp"] = pd.to_datetime(df["order_purchase_timestamp"])
    df.sort_values(by="order_purchase_timestamp", inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


all_df = load_data()

# Filter rentang waktu di sidebar, sesuai periode analisis
min_date = pd.Timestamp("2017-01-01").date()
max_date = pd.Timestamp("2018-08-31").date()

with st.sidebar:
    st.title("E-Commerce Public Dataset")
    st.write("Pesanan delivered dari 1 Januari 2017 sampai 31 Agustus 2018.")
    date_range = st.date_input(
        label="Rentang Waktu",
        min_value=min_date,
        max_value=max_date,
        value=[min_date, max_date],
        format="DD/MM/YYYY"
    )

if len(date_range) != 2:
    st.warning("Pilih tanggal awal dan tanggal akhir pada filter Rentang Waktu.")
    st.stop()

start_date, end_date = date_range
next_day = pd.Timestamp(end_date) + pd.Timedelta(days=1)
main_df = all_df[(all_df["order_purchase_timestamp"] >= pd.Timestamp(start_date)) &
                 (all_df["order_purchase_timestamp"] < next_day)]

if main_df.empty:
    st.warning("Tidak ada pesanan pada rentang waktu ini.")
    st.stop()

# Menyiapkan DataFrame dari data yang sudah difilter
# Recency dihitung sampai sehari setelah tanggal akhir filter, yaitu 1 September 2018 untuk periode penuh
monthly_orders_df = create_monthly_orders_df(main_df)
category_df = create_category_df(main_df)
bystate_df = create_bystate_df(main_df)
seller_state_df = create_seller_state_df(main_df)
rfm_df = create_rfm_df(main_df, next_day.date())

# Header dan metrik utama
st.header("Dashboard E-Commerce Public Dataset")
st.caption(f"Periode: {start_date.strftime('%d-%m-%Y')} sampai {end_date.strftime('%d-%m-%Y')}")

col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Total pesanan", value=format_number(main_df["order_id"].nunique()))
with col2:
    total_revenue = main_df.drop_duplicates(subset="order_id")["payment_value"].sum()
    st.metric("Total revenue", value="R$ " + format_number(total_revenue))
with col3:
    st.metric("Pelanggan unik", value=format_number(main_df["customer_unique_id"].nunique()))

tab1, tab2, tab3, tab4 = st.tabs(["Tren Bulanan", "Kategori Produk", "Pelanggan per Wilayah", "RFM Pelanggan"])

# Pertanyaan 1: tren jumlah pesanan dan revenue per bulan
with tab1:
    st.subheader("Jumlah Pesanan dan Revenue per Bulan")
    peak_order = monthly_orders_df.loc[monthly_orders_df["order_count"].idxmax()]
    peak_revenue = monthly_orders_df.loc[monthly_orders_df["revenue"].idxmax()]
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Bulan dengan pesanan tertinggi", value=peak_order["order_month"])
        st.caption(format_number(peak_order["order_count"]) + " pesanan")
    with col2:
        st.metric("Bulan dengan revenue tertinggi", value=peak_revenue["order_month"])
        st.caption("R$ " + format_number(peak_revenue["revenue"]))

    fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(14, 9), sharex=True)
    ax[0].plot(monthly_orders_df["order_month"], monthly_orders_df["order_count"], marker="o", linewidth=2, color=main_color)
    ax[0].set_title("Jumlah Pesanan per Bulan", loc="left", fontsize=14)
    ax[0].set_ylabel("Jumlah pesanan")
    ax[0].set_ylim(0, monthly_orders_df["order_count"].max() * 1.15)
    ax[0].yaxis.set_major_formatter(lambda x, pos: format_number(x))

    ax[1].plot(monthly_orders_df["order_month"], monthly_orders_df["revenue"] / 1_000_000, marker="o", linewidth=2, color=main_color)
    ax[1].set_title("Revenue per Bulan", loc="left", fontsize=14)
    ax[1].set_ylabel("Revenue (juta R\\$)")
    ax[1].set_ylim(0, monthly_orders_df["revenue"].max() / 1_000_000 * 1.15)
    ax[1].yaxis.set_major_formatter(lambda x, pos: format_number(x, 1))

    for axis in ax:
        axis.grid(axis="y", color="#EEEEEE")
    ax[1].tick_params(axis="x", rotation=45)
    plt.tight_layout()
    st.pyplot(fig)

# Pertanyaan 2: kategori produk paling banyak dan paling sedikit terjual
with tab2:
    st.subheader("Kategori Paling Banyak dan Paling Sedikit Terjual")
    top_category = category_df.iloc[0]
    bottom_category = category_df.sort_values(by=["unit_sold", "product_category_name_english"]).iloc[0]
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Kategori terlaris", value=top_category["product_category_name_english"])
        st.caption(format_number(top_category["unit_sold"]) + " unit")
    with col2:
        st.metric("Kategori paling sedikit", value=bottom_category["product_category_name_english"])
        st.caption(format_number(bottom_category["unit_sold"]) + " unit")

    top_category_df = category_df.head(7)
    bottom_category_df = category_df.sort_values(by=["unit_sold", "product_category_name_english"]).head(7)

    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(16, 7))
    sns.barplot(x="unit_sold", y="product_category_name_english", data=top_category_df,
                hue="product_category_name_english",
                palette={name: (main_color if i == 0 else base_color) for i, name in enumerate(top_category_df["product_category_name_english"])},
                legend=False, ax=ax[0])
    ax[0].set_title("Kategori Paling Banyak Terjual", loc="center", fontsize=15)
    ax[0].set_xlabel("Unit terjual")
    ax[0].set_ylabel(None)
    ax[0].tick_params(axis="y", labelsize=12)

    sns.barplot(x="unit_sold", y="product_category_name_english", data=bottom_category_df,
                hue="product_category_name_english",
                palette={name: (main_color if i == 0 else base_color) for i, name in enumerate(bottom_category_df["product_category_name_english"])},
                legend=False, ax=ax[1])
    ax[1].set_title("Kategori Paling Sedikit Terjual", loc="center", fontsize=15)
    ax[1].set_xlabel("Unit terjual")
    ax[1].set_ylabel(None)
    ax[1].invert_xaxis()
    ax[1].yaxis.set_label_position("right")
    ax[1].yaxis.tick_right()
    ax[1].spines["left"].set_visible(False)
    ax[1].spines["right"].set_visible(True)
    ax[1].tick_params(axis="y", labelsize=12)

    for axis in ax:
        axis.xaxis.set_major_formatter(lambda x, pos: format_number(x))
        for container in axis.containers:
            axis.bar_label(container, labels=[format_number(value) for value in container.datavalues], padding=4, fontsize=11)
    plt.tight_layout()
    st.pyplot(fig)

    st.subheader("Revenue dan Harga Rata-rata Lima Kategori dengan Revenue Terbesar")
    top_revenue_df = category_df.sort_values(by="revenue", ascending=False).head(5).copy()
    top_revenue_df["revenue_million"] = top_revenue_df["revenue"] / 1_000_000
    top_revenue_name = top_revenue_df["product_category_name_english"].iloc[0]
    top_price_name = top_revenue_df.sort_values(by="avg_price", ascending=False)["product_category_name_english"].iloc[0]

    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(16, 6), sharey=True)
    sns.barplot(x="revenue_million", y="product_category_name_english", data=top_revenue_df,
                hue="product_category_name_english",
                palette={name: (main_color if name == top_revenue_name else base_color) for name in top_revenue_df["product_category_name_english"]},
                legend=False, ax=ax[0])
    ax[0].set_title("Revenue (juta R\\$)", loc="center", fontsize=15)
    ax[0].set_xlabel(None)
    ax[0].set_ylabel(None)
    ax[0].tick_params(axis="y", labelsize=12)
    ax[0].xaxis.set_major_formatter(lambda x, pos: format_number(x, 1))

    sns.barplot(x="avg_price", y="product_category_name_english", data=top_revenue_df,
                hue="product_category_name_english",
                palette={name: (main_color if name == top_price_name else base_color) for name in top_revenue_df["product_category_name_english"]},
                legend=False, ax=ax[1])
    ax[1].set_title("Harga Rata-rata per Unit (R\\$)", loc="center", fontsize=15)
    ax[1].set_xlabel(None)
    ax[1].set_ylabel(None)

    for container in ax[0].containers:
        ax[0].bar_label(container, labels=[format_number(value, 2) for value in container.datavalues], padding=4, fontsize=11)
    for container in ax[1].containers:
        ax[1].bar_label(container, labels=[format_number(value, 2) for value in container.datavalues], padding=4, fontsize=11)
    plt.tight_layout()
    st.pyplot(fig)

# Pertanyaan 3: sebaran pelanggan per negara bagian
with tab3:
    st.subheader("Jumlah Pelanggan per Negara Bagian")
    top_state = bystate_df.iloc[0]
    delivery_freight_corr = bystate_df["avg_delivery_time"].corr(bystate_df["avg_freight_per_item"])
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Pelanggan terbanyak", value=top_state["customer_state"])
        st.caption(format_number(top_state["customer_count"]) + " pelanggan")
    with col2:
        st.metric("Pangsa pelanggan " + top_state["customer_state"], value=format_number(top_state["customer_share"], 1) + "%")
    with col3:
        st.metric("Korelasi lama kirim dan ongkir",
                  value="-" if pd.isna(delivery_freight_corr) else format_number(delivery_freight_corr, 2))

    fig, ax = plt.subplots(figsize=(12, 9))
    sns.barplot(x="customer_count", y="customer_state", data=bystate_df,
                hue="customer_state",
                palette={state: (main_color if i == 0 else base_color) for i, state in enumerate(bystate_df["customer_state"])},
                legend=False, ax=ax)
    for index, row in bystate_df.head(3).iterrows():
        ax.text(row["customer_count"] + bystate_df["customer_count"].max() * 0.01, index,
                f'{format_number(row["customer_count"])} ({format_number(row["customer_share"], 1)}%)', va="center", fontsize=11)
    ax.set_xlabel("Jumlah pelanggan unik")
    ax.set_ylabel(None)
    ax.xaxis.set_major_formatter(lambda x, pos: format_number(x))
    ax.grid(axis="x", color="#EEEEEE")
    st.pyplot(fig)

    st.subheader("Pangsa Pelanggan dan Seller Aktif")
    share_df = pd.merge(
        left=bystate_df[["customer_state", "customer_share"]].head(5),
        right=seller_state_df[["seller_state", "seller_share"]],
        how="left",
        left_on="customer_state",
        right_on="seller_state"
    )
    share_df["seller_share"] = share_df["seller_share"].fillna(0)
    share_df = share_df.melt(id_vars="customer_state", value_vars=["customer_share", "seller_share"], var_name="group", value_name="share")
    share_df["group"] = share_df["group"].replace({"customer_share": "Pelanggan", "seller_share": "Seller aktif"})

    fig, ax = plt.subplots(figsize=(12, 6))
    sns.barplot(x="customer_state", y="share", hue="group", data=share_df,
                palette={"Pelanggan": main_color, "Seller aktif": second_color}, ax=ax)
    for container in ax.containers:
        ax.bar_label(container, labels=[format_number(value, 1) + "%" for value in container.datavalues], padding=3, fontsize=11)
    ax.set_xlabel(None)
    ax.set_ylabel("Persentase (%)")
    ax.set_ylim(0, share_df["share"].max() * 1.2)
    ax.legend(title=None)
    st.pyplot(fig)

    st.subheader("Hubungan Lama Pengiriman dan Ongkos Kirim")
    fig, ax = plt.subplots(figsize=(12, 7))
    sns.regplot(x="avg_delivery_time", y="avg_freight_per_item", data=bystate_df, color=line_color,
                scatter_kws={"s": 80, "color": "#B0B0B0", "edgecolor": "white", "linewidths": 2}, line_kws={"linewidth": 1.5}, ax=ax)
    fastest_state = bystate_df.loc[bystate_df["avg_delivery_time"].idxmin()]
    slowest_state = bystate_df.loc[bystate_df["avg_delivery_time"].idxmax()]
    highlight_df = bystate_df[bystate_df["customer_state"].isin([fastest_state["customer_state"], slowest_state["customer_state"]])]
    ax.scatter(highlight_df["avg_delivery_time"], highlight_df["avg_freight_per_item"], s=110, color=main_color, edgecolor="white", linewidth=2, zorder=3)
    for _, row in highlight_df.iterrows():
        offset = (0, -22) if row["customer_state"] == fastest_state["customer_state"] else (0, 10)
        ax.annotate(f'{row["customer_state"]}: {format_number(row["avg_delivery_time"], 1)} hari, R\\${format_number(row["avg_freight_per_item"], 2)}',
                    xy=(row["avg_delivery_time"], row["avg_freight_per_item"]), xytext=offset, textcoords="offset points", ha="center", fontsize=11)
    ax.set_xlabel("Rata-rata lama pengiriman (hari)")
    ax.set_ylabel("Rata-rata ongkos kirim per item (R\\$)")
    ax.set_xlim(0, bystate_df["avg_delivery_time"].max() * 1.2)
    ax.set_ylim(0, bystate_df["avg_freight_per_item"].max() * 1.25)
    ax.grid(color="#EEEEEE")
    st.pyplot(fig)

# Pertanyaan 4, 5, dan 6: RFM analysis
with tab4:
    st.subheader("Recency, Frequency, dan Monetary Pelanggan")
    inactive_count = (rfm_df["recency"] > 180).sum()
    inactive_share = (rfm_df["recency"] > 180).mean() * 100
    repeat_share = (rfm_df["frequency"] > 1).mean() * 100
    monetary_mean = rfm_df["monetary"].mean()
    monetary_median = rfm_df["monetary"].median()

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Recency lebih dari 180 hari", value=format_number(inactive_share, 1) + "%")
        st.caption(format_number(inactive_count) + " pelanggan")
    with col2:
        st.metric("Berbelanja lebih dari satu kali", value=format_number(repeat_share, 1) + "%")
        st.caption(format_number((rfm_df["frequency"] > 1).sum()) + " pelanggan")
    with col3:
        st.metric("Median total belanja", value="R$ " + format_number(monetary_median, 2))
        st.caption("Rata-rata R$ " + format_number(monetary_mean, 2))

    # Pertanyaan 4: sebaran recency
    st.subheader("Sebaran Recency Pelanggan")
    bins = np.arange(1, rfm_df["recency"].max() + 31, 30)
    fig, ax = plt.subplots(figsize=(14, 6))
    low_counts, _, _ = ax.hist(rfm_df.loc[rfm_df["recency"] <= 180, "recency"], bins=bins, color=base_color, edgecolor="white", linewidth=2)
    high_counts, _, _ = ax.hist(rfm_df.loc[rfm_df["recency"] > 180, "recency"], bins=bins, color=main_color, edgecolor="white", linewidth=2)
    max_count = max(low_counts.max(), high_counts.max())
    ax.axvline(x=181, color=line_color, linewidth=1)
    ax.text(186, max_count * 1.05, "Lebih dari 180 hari", fontsize=12)
    ax.set_xlabel(f"Recency (hari sejak pesanan terakhir sampai {next_day.strftime('%d-%m-%Y')})")
    ax.set_ylabel("Jumlah pelanggan")
    ax.set_ylim(0, max_count * 1.15)
    ax.yaxis.set_major_formatter(lambda x, pos: format_number(x))
    ax.grid(axis="y", color="#EEEEEE")
    st.pyplot(fig)

    # Pertanyaan 5: frekuensi belanja
    st.subheader("Pelanggan Berdasarkan Frekuensi Belanja")
    rfm_df["frequency_group"] = rfm_df["frequency"].apply(lambda x: "1 kali" if x == 1 else ("2 kali" if x == 2 else "3 kali atau lebih"))
    byfrequency_df = rfm_df.groupby(by="frequency_group").customer_unique_id.nunique().reset_index()
    byfrequency_df.rename(columns={"customer_unique_id": "customer_count"}, inplace=True)
    order_reviews = main_df.drop_duplicates(subset="order_id")["review_score"].dropna()
    st.caption(f"Rata-rata skor ulasan per pesanan {format_number(order_reviews.mean(), 2)}, "
               f"dan {format_number((order_reviews >= 4).mean() * 100, 1)}% pesanan diberi skor 4 atau 5.")

    fig, ax = plt.subplots(figsize=(12, 6))
    sns.barplot(x="frequency_group", y="customer_count", data=byfrequency_df, hue="frequency_group",
                palette={"1 kali": base_color, "2 kali": main_color, "3 kali atau lebih": main_color}, legend=False, ax=ax)
    for container in ax.containers:
        ax.bar_label(container, labels=[format_number(value) for value in container.datavalues], padding=3, fontsize=11)
    ax.set_xlabel(None)
    ax.set_ylabel("Jumlah pelanggan")
    ax.yaxis.set_major_formatter(lambda x, pos: format_number(x))
    ax.grid(axis="y", color="#EEEEEE")
    st.pyplot(fig)

    # Pertanyaan 6: sebaran total belanja
    st.subheader("Sebaran Total Belanja Pelanggan")
    above_limit = (rfm_df["monetary"] > 1000).sum()
    fig, ax = plt.subplots(figsize=(14, 6))
    counts, _, _ = ax.hist(rfm_df.loc[rfm_df["monetary"] <= 1000, "monetary"], bins=np.arange(0, 1025, 25), color=base_color, edgecolor="white", linewidth=2)
    ax.set_ylim(0, counts.max() * 1.15)
    ax.axvline(x=monetary_median, color=main_color, linewidth=2)
    ax.axvline(x=monetary_mean, color=line_color, linewidth=2)
    ax.text(monetary_median - 8, counts.max() * 1.07, f"Median R\\${format_number(monetary_median, 2)}", color=main_color, ha="right", fontsize=12)
    ax.text(monetary_mean + 8, counts.max() * 1.07, f"Rata-rata R\\${format_number(monetary_mean, 2)}", color=line_color, ha="left", fontsize=12)
    ax.set_xlabel(f"Total belanja per pelanggan (R\\$), {format_number(above_limit)} pelanggan di atas R\\$1.000 tidak ditampilkan")
    ax.set_ylabel("Jumlah pelanggan")
    ax.xaxis.set_major_formatter(lambda x, pos: format_number(x))
    ax.yaxis.set_major_formatter(lambda x, pos: format_number(x))
    ax.grid(axis="y", color="#EEEEEE")
    st.pyplot(fig)

plt.close("all")
st.caption("Copyright (c) Novi Nur Qomariya 2026")
