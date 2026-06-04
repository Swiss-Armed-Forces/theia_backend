from theia.terrain_fast_los import build_terrain_tree


tree = build_terrain_tree(45, 49, 6, 11, subsample_stride=2)
tree.save("tree_lat45:49_lon6:11.zip")