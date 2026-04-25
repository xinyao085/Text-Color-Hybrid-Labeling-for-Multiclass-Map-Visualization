import json
import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.patches as mpatches

def generate_land_use_map(
    geojson_path,
    category_col="LU_DESC",
    area_col="SHAPE.AREA",
    area_threshold=5000,
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

    返回值：
        no_legend_path : 无图例地图的路径（供词云使用）
        color_map      : {类别名: [R, G, B]} 字典
    """
    # 读取数据
    gdf = gpd.read_file(geojson_path)
    print(f"数据量：{len(gdf)}，字段：{gdf.columns.tolist()}")

    gdf[category_col] = gdf[category_col].fillna("UNKNOWN")
    print(f"类别数量：{gdf[category_col].nunique()}")

    # 过滤小面积地块
    gdf_plot = gdf[gdf[area_col] > area_threshold].copy()
    print(f"过滤后数据量：{len(gdf_plot)}")

    # 合并低频类别
    top_categories = gdf_plot[category_col].value_counts().head(top_n).index
    gdf_plot["plot_category"] = gdf_plot[category_col].apply(
        lambda x: x if x in top_categories else "OTHER"
    )

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

    print("\n颜色分配预览：")
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

    # 保存带图例版本
    plt.savefig(map_path, dpi=dpi, bbox_inches="tight")

    # 保存无图例版本
    ax.get_legend().remove()
    plt.savefig(no_legend_path, dpi=dpi, bbox_inches="tight", pad_inches=0)
    plt.show()

    # 保存颜色映射 JSON
    with open(color_map_path, "w", encoding="utf-8") as f:
        json.dump({"map_path": no_legend_path, "color_map": color_map}, f, ensure_ascii=False, indent=2)

    print(f"\n带图例地图：{map_path}")
    print(f"无图例地图：{no_legend_path}")
    print(f"颜色映射：{color_map_path}")

    return no_legend_path, color_map
