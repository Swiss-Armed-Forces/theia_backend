from theia.terrain_fast_los import build_terrain_tree


lat_min = 46
lat_max = 49
lon_min = 7

lon_max = 11
subsample = 2

tree = build_terrain_tree(
    lat_min,
    lat_max,
    lon_min,
    lon_max,
    subsample_stride=subsample,
)
tree.save(
    f"tree_lat{lat_min}:{lat_max}_lon{lon_min}:{lon_max}_subsamplestride{subsample}.zip"
)
