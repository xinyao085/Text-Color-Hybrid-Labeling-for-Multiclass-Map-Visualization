import json
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.patches as mpatches
from pyproj import Geod
from sklearn.cluster import DBSCAN

_GEOD = Geod(ellps="WGS84")


def _geodesic_area(geom):
    """计算几何体的大地测量面积（平方米），支持 Polygon 和 MultiPolygon。"""
    if geom is None or geom.is_empty:
        return 0.0
    if geom.geom_type == "Polygon":
        area, _ = _GEOD.geometry_area_perimeter(geom)
        return abs(area)
    return sum(abs(_GEOD.geometry_area_perimeter(p)[0]) for p in geom.geoms)


def _cluster_and_filter(gdf_plot, cluster_eps_m, min_cluster_area):
    """
    对每个类别的地块按质心距离做 DBSCAN 聚类，
    过滤掉总面积不足 min_cluster_area 的聚类，
    再把同聚类内相邻地块 dissolve 合并。
    """
    eps_deg = cluster_eps_m / 111000.0  # 米转度（近似）
    clustered_parts = []

    for cat in gdf_plot["plot_category"].unique():
        cat_gdf = gdf_plot[gdf_plot["plot_category"] == cat].copy()

        # 用质心坐标聚类
        centroids = np.array([
            [g.centroid.x, g.centroid.y] for g in cat_gdf.geometry
        ])
        labels = DBSCAN(eps=eps_deg, min_samples=1, algorithm="ball_tree",
                        metric="haversine").fit(np.radians(centroids)).labels_
        cat_gdf["cluster_id"] = labels
        clustered_parts.append(cat_gdf)

    gdf_clustered = pd.concat(clustered_parts, ignore_index=True)

    # 计算每个聚类的总面积，过滤面积不足的整组
    group_area = (
        gdf_clustered
        .groupby(["plot_category", "cluster_id"])["geometry"]
        .apply(lambda gs: sum(_geodesic_area(g) for g in gs))
        .reset_index(name="cluster_area")
    )
    gdf_clustered = gdf_clustered.merge(group_area, on=["plot_category", "cluster_id"])
    gdf_kept = gdf_clustered[gdf_clustered["cluster_area"] > min_cluster_area].copy()

    # 同聚类内 dissolve（合并接触的多边形），再 explode 拆回单体
    gdf_final = (
        gdf_kept[["plot_category", "cluster_id", "geometry"]]
        .dissolve(by=["plot_category", "cluster_id"])
        .reset_index()
        [["plot_category", "geometry"]]
        .explode(index_parts=False)
        .reset_index(drop=True)
    )

    # 清理 dissolve 后的空/无效几何体
    gdf_final = gdf_final[
        ~gdf_final.geometry.is_empty & gdf_final.geometry.is_valid
    ].copy()

    return gdf_final


def generate_land_use_map(
    geojson_path,
    category_col="LU_DESC",
    area_col="SHAPE.AREA",
    area_threshold=50000,
    cluster_eps_m=5000,
    min_cluster_area=100000,
    top_n=12,
    map_path="land_use_map_with_legend.png",
    no_legend_path="land_use_map_NoLegend.png",
    color_map_path="land_use_color_map.json",
    title="Land Use Classification Map",
    figsize=(14, 10),
    dpi=300,
):
    """
    从 GeoJSON 生成分类地图，自动分配颜色并保存颜色映射。

    参数：
        area_threshold   : 初始面积过滤（平方米），直接丢弃极小地块
        cluster_eps_m    : DBSCAN 聚类半径（米），质心距离小于此值的同类地块归为一组
        min_cluster_area : 聚类总面积阈值（平方米），整组面积不足则整组丢弃
        top_n            : 保留出现次数最多的前 N 个类别，其余合并为 OTHER

    返回值：
        no_legend_path : 无图例地图路径（供词云使用）
        color_map      : {类别名: [R, G, B]} 字典
    """
    # 读取数据
    gdf = gpd.read_file(geojson_path)
    print(f"原始数据量：{len(gdf)}")

    gdf[category_col] = gdf[category_col].fillna("UNKNOWN")

    # 初始面积过滤（去掉极小地块）
    gdf_plot = gdf[gdf[area_col] > area_threshold].copy()
    print(f"初始过滤后：{len(gdf_plot)} 个地块")

    # 合并低频类别为 OTHER
    top_categories = gdf_plot[category_col].value_counts().head(top_n).index
    gdf_plot["plot_category"] = gdf_plot[category_col].apply(
        lambda x: x if x in top_categories else "OTHER"
    )

    # 聚类 + 过滤
    print(f"DBSCAN 聚类（半径 {cluster_eps_m}m），过滤总面积 < {min_cluster_area}m² 的聚类...")
    gdf_plot = _cluster_and_filter(gdf_plot, cluster_eps_m, min_cluster_area)
    print(f"聚类过滤后：{len(gdf_plot)} 个地块")
    print(gdf_plot["plot_category"].value_counts())

    # 分配颜色
    categories = sorted(gdf_plot["plot_category"].unique())
    n = len(categories)
    if n <= 10:
        palette = [cm.get_cmap("tab10")(i / 10) for i in range(n)]
    elif n <= 20:
        palette = [cm.get_cmap("tab20")(i / 20) for i in range(n)]
    else:
        palette = [cm.get_cmap("hsv")(i / n) for i in range(n)]

    cat_to_rgba = {cat: palette[i] for i, cat in enumerate(categories)}
    row_colors = [cat_to_rgba[cat] for cat in gdf_plot["plot_category"]]
    color_map = {
        cat: [int(rgba[0] * 255), int(rgba[1] * 255), int(rgba[2] * 255)]
        for cat, rgba in cat_to_rgba.items()
    }

    print("\n颜色分配：")
    for cat, rgb in color_map.items():
        print(f"  {tuple(rgb)} -> {cat}")

    # 绘图
    fig, ax = plt.subplots(figsize=figsize)
    gdf_plot.plot(color=row_colors, linewidth=0.05, edgecolor="white", ax=ax, aspect="equal")
    legend_patches = [
        mpatches.Patch(facecolor=cat_to_rgba[cat], label=cat)
        for cat in categories
    ]
    ax.legend(handles=legend_patches, loc="lower right", framealpha=0.9, fontsize=8)
    ax.set_title(title, fontsize=16)
    ax.axis("off")

    plt.savefig(map_path, dpi=dpi, bbox_inches="tight")
    ax.get_legend().remove()
    plt.savefig(no_legend_path, dpi=dpi, bbox_inches="tight", pad_inches=0)
    plt.show()

    with open(color_map_path, "w", encoding="utf-8") as f:
        json.dump({"map_path": no_legend_path, "color_map": color_map}, f, ensure_ascii=False, indent=2)

    print(f"\n带图例地图：{map_path}")
    print(f"无图例地图：{no_legend_path}")
    print(f"颜色映射：{color_map_path}")

    return no_legend_path, color_map
