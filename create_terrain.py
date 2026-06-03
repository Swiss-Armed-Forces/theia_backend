from theia.terrain_fast_los import build_terrain_tree


tree = build_terrain_tree(46, 49, 7, 9, subsample_stride=2)
tree.save("tree_lat46:49_lon7:9.zip")