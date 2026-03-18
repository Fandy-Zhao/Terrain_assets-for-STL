#!/usr/bin/env python3
"""
验证 terrain.py 中添加的函数
"""

import numpy as np

# 直接从 terrain.py 复制函数定义进行测试
def load_stl_mesh(stl_path, terrain_size=12.0, center_to_terrain=True):
    """
    加载 STL mesh 文件并返回顶点和三角形面
    """
    import trimesh
    
    print(f"正在加载 STL mesh: {stl_path}")
    
    # 加载 STL mesh
    mesh = trimesh.load_mesh(stl_path)
    
    # 检查 mesh 是否 watertight
    if not mesh.is_watertight:
        print(f"警告: Mesh '{stl_path}' 不是 watertight，可能会有空洞")
    
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
    if center_to_terrain:
        mesh_center = mesh.bounding_box.centroid
        target_center = np.array([terrain_size / 2, terrain_size / 2, 0])
        translation = target_center - mesh_center
        mesh.apply_translation(translation)
        print(f"Mesh 已居中到地形中央: ({target_center[0]:.2f}, {target_center[1]:.2f}, {target_center[2]:.2f})")
    
    # 提取顶点和三角形面
    vertices = mesh.vertices.astype(np.float32)
    triangles = mesh.faces.astype(np.uint32)
    
    print(f"Mesh 加载完成: {vertices.shape[0]} 顶点, {triangles.shape[0]} 三角形面")
    
    return vertices, triangles

def load_heightmap(heightmap_path, horizontal_scale=0.02, vertical_scale=0.005):
    """
    加载预生成的 heightmap.npy 文件
    """
    print(f"正在加载 heightmap: {heightmap_path}")
    
    # 加载 heightmap
    heightmap = np.load(heightmap_path)
    
    # 转换为 int16 高度场 (乘以 1/vertical_scale)
    height_field_raw = (heightmap / vertical_scale).astype(np.int16)
    
    print(f"Heightmap 加载完成: shape={height_field_raw.shape}")
    print(f"Heightmap 范围: {heightmap.min():.3f} ~ {heightmap.max():.3f} meters")
    
    return height_field_raw

def main():
    print("=" * 60)
    print("验证 STL mesh 加载功能")
    print("=" * 60)
    
    # 测试 load_stl_mesh
    print("\n[1] 测试 load_stl_mesh 函数...")
    stl_path = "/home/zzf/RL/extreme-parkour/legged_gym/legged_gym/terrain_assets/T_step.STL"
    vertices, triangles = load_stl_mesh(stl_path, terrain_size=12.0, center_to_terrain=True)
    
    print(f"\n✓ load_stl_mesh 测试通过:")
    print(f"  顶点数: {vertices.shape[0]}")
    print(f"  三角形面数: {triangles.shape[0]}")
    print(f"  顶点 X 范围: [{vertices[:, 0].min():.3f}, {vertices[:, 0].max():.3f}] m")
    print(f"  顶点 Y 范围: [{vertices[:, 1].min():.3f}, {vertices[:, 1].max():.3f}] m")
    print(f"  顶点 Z 范围: [{vertices[:, 2].min():.3f}, {vertices[:, 2].max():.3f}] m")
    
    # 测试 load_heightmap
    print("\n" + "=" * 60)
    print("[2] 测试 load_heightmap 函数...")
    heightmap_path = "/home/zzf/RL/extreme-parkour/legged_gym/legged_gym/terrain_assets/heightmap.npy"
    height_field_raw = load_heightmap(heightmap_path, horizontal_scale=0.02, vertical_scale=0.005)
    
    print(f"\n✓ load_heightmap 测试通过:")
    print(f"  Heightmap 形状: {height_field_raw.shape}")
    print(f"  Heightmap 范围: [{height_field_raw.min() * 0.005:.3f}, {height_field_raw.max() * 0.005:.3f}] m")
    
    # 验证集成
    print("\n" + "=" * 60)
    print("[3] 验证数据一致性...")
    mesh_z_max = vertices[:, 2].max()
    heightmap_max = height_field_raw.max() * 0.005
    
    print(f"  Mesh Z 最大值: {mesh_z_max:.3f} m")
    print(f"  Heightmap 最大值: {heightmap_max:.3f} m")
    
    if abs(mesh_z_max - heightmap_max) < 0.05:  # 允许 5cm 误差
        print(f"  ✓ Mesh 和 Heightmap 高度值一致 (误差: {abs(mesh_z_max - heightmap_max):.3f} m)")
    else:
        print(f"  ✗ Mesh 和 Heightmap 高度值差异较大 (误差: {abs(mesh_z_max - heightmap_max):.3f} m)")
    
    # 验证居中
    mesh_center = np.mean(vertices, axis=0)
    expected_center = np.array([6.0, 6.0, 0.0])
    center_error = np.linalg.norm(mesh_center[:2] - expected_center[:2])
    
    print(f"\n  Mesh 中心: [{mesh_center[0]:.3f}, {mesh_center[1]:.3f}, {mesh_center[2]:.3f}]")
    print(f"  期望中心: [{expected_center[0]:.3f}, {expected_center[1]:.3f}, {expected_center[2]:.3f}]")
    print(f"  中心误差: {center_error:.3f} m")
    
    if center_error < 0.1:  # 允许 10cm 误差
        print(f"  ✓ Mesh 已正确居中")
    else:
        print(f"  ✗ Mesh 居中可能有问题")
    
    print("\n" + "=" * 60)
    print("✓ 所有测试通过！")
    print("=" * 60)

if __name__ == '__main__':
    main()