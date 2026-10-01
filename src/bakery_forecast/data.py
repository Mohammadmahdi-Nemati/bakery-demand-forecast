"""Daten laden: aus den CSV-Dateien von bakery-sales-db eine Tagestabelle je Filiale und Produkt bauen."""

from pathlib import Path

import pandas as pd


def load_daily_panel(data_dir: str | Path) -> pd.DataFrame:
    """Eine Zeile je (Filiale, Produkt, Tag) mit allem, was die Bäckerei an dem Tag weiß.

    Spalten:
        branch_id, product_id, category_id, date
        baked          produzierte Menge
        sold           verkaufte Menge
        wasted         abgeschriebene Menge
        stockout       True, wenn alles verkauft wurde (echte Nachfrage dann unbekannt)
        last_sale_time Uhrzeit des letzten Verkaufs als Dezimalzahl, z. B. 10.5 = 10:30 (für die Ausverkaufs-Korrektur)
        price, unit_cost, is_holiday
    """
    d = Path(data_dir)
    production = pd.read_csv(d / "daily_production.csv", parse_dates=["prod_date"])
    receipts = pd.read_csv(d / "receipt.csv", parse_dates=["sold_at"])
    items = pd.read_csv(d / "receipt_item.csv")
    waste = pd.read_csv(d / "waste.csv", parse_dates=["waste_date"])
    products = pd.read_csv(d / "product.csv")
    holidays = pd.read_csv(d / "public_holiday.csv", parse_dates=["holiday_date"])

    sales = items.merge(receipts[["receipt_id", "branch_id", "sold_at"]], on="receipt_id")
    sales["date"] = sales["sold_at"].dt.normalize()
    sales["hour"] = sales["sold_at"].dt.hour
    sales["time"] = sales["hour"] + sales["sold_at"].dt.minute / 60
    daily_sales = (
        sales.groupby(["branch_id", "product_id", "date"])
        .agg(sold=("quantity", "sum"), last_sale_time=("time", "max"))
        .reset_index()
    )

    daily_waste = (
        waste.groupby(["branch_id", "product_id", "waste_date"])["quantity_wasted"]
        .sum()
        .rename("wasted")
        .reset_index()
        .rename(columns={"waste_date": "date"})
    )

    panel = (
        production.rename(columns={"prod_date": "date", "quantity_baked": "baked"})
        .merge(daily_sales, on=["branch_id", "product_id", "date"], how="left")
        .merge(daily_waste, on=["branch_id", "product_id", "date"], how="left")
        .merge(products[["product_id", "category_id", "price", "unit_cost"]], on="product_id")
    )
    panel[["sold", "wasted"]] = panel[["sold", "wasted"]].fillna(0).astype(int)
    panel["stockout"] = (panel["baked"] > 0) & (panel["sold"] >= panel["baked"])
    panel["is_holiday"] = panel["date"].isin(holidays["holiday_date"])
    return panel.sort_values(["branch_id", "product_id", "date"]).reset_index(drop=True)


def load_hourly_sales(data_dir: str | Path) -> pd.DataFrame:
    """Verkaufte Menge je (Filiale, Produkt, Tag, Stunde). Grundlage für Verkaufsprofile über den Tag."""
    d = Path(data_dir)
    receipts = pd.read_csv(d / "receipt.csv", parse_dates=["sold_at"])
    items = pd.read_csv(d / "receipt_item.csv")
    sales = items.merge(receipts[["receipt_id", "branch_id", "sold_at"]], on="receipt_id")
    sales["date"] = sales["sold_at"].dt.normalize()
    sales["hour"] = sales["sold_at"].dt.hour
    return sales.groupby(["branch_id", "product_id", "date", "hour"])["quantity"].sum().reset_index()


def load_true_demand(data_dir: str | Path) -> pd.DataFrame:
    """Echte Nachfrage aus der Simulation. Nur für die Bewertung, nie für das Training."""
    td = pd.read_csv(Path(data_dir) / "true_demand.csv", parse_dates=["demand_date"])
    return td.rename(columns={"demand_date": "date", "demand": "true_demand"})
