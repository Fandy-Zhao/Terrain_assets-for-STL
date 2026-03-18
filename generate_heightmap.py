#!/usr/bin/env python3
"""
从 STL mesh 文件生成 heightmap 用于 Extreme Parkour 地形观测

用法:
    python generate_heightmap.py --stl_path T_step.STL --output heightmap.npy
"""

import numpy as np
import argparse
import os
try:
    import trimesh
except ImportError:
    print("正在安装 trimesh...")
    os.system("pip install trimesh")
    import trimesh


def generate_heightmap_from_stl(stl_path, output_path, terrain_size=12.0, resolution=0.02, z_offset=0.0):
    """
    从 STL mesh 文件生成 heightmap

    参数:
        stl_path: STL 文件路径
        output_path: 输出的 heightmap.npy 文件路径
        terrain_size: 地形尺寸 [meters]，默认 12m × 12m
        resolution: 分辨率 [meters]，默认 0.02m
        z_offset: z轴偏移量 [meters]，默认 0.0（mesh中心在z=0平面）
    """
    print(f"正在加载 STL 文件: {stl_path}")
    
    # 加载 STL mesh
    mesh = trimesh.load_mesh(stl_path)
    
    # 检查 mesh 是否 watertight
    if not mesh.is_watertight:
        print(f"警告: Mesh 不是 watertight，可能会有空洞")
    
    # 检查单位并自动转换
    bbox = mesh.bounding_box.bounds
    bbox_size = bbox[1] - bbox[0]
    max_dim = np.max(bbox_size)
    
    print(f"Mesh bounding box: {bbox}")
    print(f"Mesh size: {bbox_size}")
    print(f"Max dimension: {max_dim:.3f} meters")
    
    # 如果 max_dim > 100，假设单位是 mm，转换为 meters
    if max_dim > 100:
        print("检测到单位可能是 mm，自动转换为 meters")
        mesh.apply_scale(0.001)
        bbox = mesh.bounding_box.bounds
        bbox_size = bbox[1] - bbox[0]
        max_dim = np.max(bbox_size)
        print(f"转换后 Mesh bounding box: {bbox}")
        print(f"转换后 Mesh size: {bbox_size}")
    
    # 将 mesh 居中到地形中央
    mesh_center = mesh.bounding_box.centroid
    target_center = np.array([terrain_size / 2, terrain_size / 2, z_offset])
    translation = target_center - mesh_center
    mesh.apply_translation(translation)

    print(f"Mesh 已居中到地形中央: ({target_center[0]:.2f}, {target_center[1]:.2f}, {target_center[2]:.2f})")
    if z_offset != 0.0:
        print(f"z 轴偏移量: {z_offset:.3f} m")
    
    # 生成 ray grid
    grid_size = int(terrain_size / resolution)
    x = np.linspace(0, terrain_size, grid_size)
    y = np.linspace(0, terrain_size, grid_size)
    xx, yy = np.meshgrid(x, y)
    
    # 创建 ray origins (从上方投射)
    ray_origins = np.zeros((grid_size * grid_size, 3))
    ray_origins[:, 0] = xx.flatten()
    ray_origins[:, 1] = yy.flatten()
    ray_origins[:, 2] = 100.0  # 从高处投射
    
    # ray directions (向下)
    ray_directions = np.zeros((grid_size * grid_size, 3))
    ray_directions[:, 2] = -1.0
    
    print(f"正在 raycasting: {grid_size} × {grid_size} = {grid_size * grid_size} rays")
    
    # 执行 ray intersection
    locations, _, _ = mesh.ray.intersects_location(
        ray_origins=ray_origins,
        ray_directions=ray_directions
    )
    
    # 构建 heightmap
    heightmap = np.zeros((grid_size, grid_size), dtype=np.float32)
    
    # 对于每个 ray，找到最近的交点
    for i in range(grid_size):
        for j in range(grid_size):
            idx = i * grid_size + j
            # 找到所有来自这个 ray 的交点
            mask = (locations[:, 0] == xx[i, j]) & (locations[:, 1] == yy[i, j])
            ray_hits = locations[mask]
            
            if len(ray_hits) > 0:
                # 取最高点（z坐标最大的）
                heightmap[i, j] = np.max(ray_hits[:, 2])
            else:
                # 没有交点，设置为地面高度 (0)
                heightmap[i, j] = 0.0
    
    print(f"Heightmap 生成完成: shape={heightmap.shape}")
    print(f"Heightmap 范围: {heightmap.min():.3f} ~ {heightmap.max():.3f} meters")
    
    # 保存 heightmap
    np.save(output_path, heightmap)
    print(f"Heightmap 已保存到: {output_path}")
    
    return heightmap


def main():
    parser = argparse.ArgumentParser(description='从 STL 生成 heightmap')
    parser.add_argument('--stl_path', type=str, default='T_step.STL',
                        help='STL 文件路径')
    parser.add_argument('--output', type=str, default='heightmap.npy',
                        help='输出的 heightmap.npy 文件路径')
    parser.add_argument('--terrain_size', type=float, default=12.0,
                        help='地形尺寸 [meters] (默认: 12.0)')
    parser.add_argument('--resolution', type=float, default=0.02,
                        help='分辨率 [meters] (默认: 0.02)')
    parser.add_argument('--z_offset', type=float, default=0.0,
                        help='z轴偏移量 [meters] (默认: 0.0, mesh中心在z=0平面)')

    args = parser.parse_args()
    
    # 生成 heightmap
    heightmap = generate_heightmap_from_stl(
        stl_path=args.stl_path,
        output_path=args.output,
        terrain_size=args.terrain_size,
        resolution=args.resolution,
        z_offset=args.z_offset
    )


if __name__ == '__main__':
    main()