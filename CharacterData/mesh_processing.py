import numpy as np

def subdivide_catmull_clark_approx(vertices, faces):
    """
    Performs one iteration of approximate Catmull-Clark subdivision for quad meshes.
    Optimized for Numpy.
    """
    vertices = np.array(vertices)
    faces = np.array(faces)
    
    n_verts = len(vertices)
    n_faces = len(faces)
    
    # 1. Calculate Face Points (Centroids) - These are the new center vertices
    # Shape: (F, 3)
    face_points = vertices[faces].mean(axis=1)
    
    # 2. Identify Edges
    # Extract edges: (0,1), (1,2), (2,3), (3,0)
    # Shape: (F, 4, 2)
    face_edges = np.stack([
        faces[:, [0, 1]],
        faces[:, [1, 2]],
        faces[:, [2, 3]],
        faces[:, [3, 0]]
    ], axis=1)
    
    # Flatten to (F*4, 2)
    all_edges = face_edges.reshape(-1, 2)
    # Sort vertex indices to ensure (0,1) == (1,0)
    all_edges.sort(axis=1)
    
    # Find unique edges
    # unique_edges: (E, 2)
    # inverse_indices: (F*4,) map into unique_edges
    unique_edges, inverse_indices = np.unique(all_edges, axis=0, return_inverse=True)
    n_edges = len(unique_edges)
    
    # Edge Lookup per face: (F, 4) indices into unique_edges
    face_edge_indices = inverse_indices.reshape(n_faces, 4)
    
    # 3. Calculate Edge Points
    # Ideally Catmull-Clark Edge Point = (v1 + v2 + f1 + f2) / 4
    # But determining f1/f2 (adjacent faces) is expensive.
    # Approximation: Simple Midpoint = (v1 + v2) / 2
    # This creates linear subdivision. To get smoothing, we rely on a post-pass or just this being denser.
    # actually, linear subdivision results in flat surfaces. 
    # Let's try to include face points in edge average if possible.
    # We can sum face points onto edges?
    # np.add.at can sum face_point[f] into edge_accumulator[edge_index]
    # Count how many faces share edge (usually 2, boundary 1).
    
    edge_midpoints = (vertices[unique_edges[:, 0]] + vertices[unique_edges[:, 1]]) / 2.0
    
    # Better Edge Points: Average of endpoints + Average of sharing face points
    # Initialize with sum of endpoints
    edge_points_sum = vertices[unique_edges[:, 0]] + vertices[unique_edges[:, 1]] # (E, 3)
    edge_counts = np.zeros(n_edges, dtype=np.int32) + 2 # 2 endpoints
    
    # Add face points
    # We know which edge belongs to which face from face_edge_indices
    # face_edge_indices[f, k] is the edge index for face f, edge k.
    # We want to add face_points[f] to edge_points_sum[ face_edge_indices[f, k] ]
    
    # This repeats face_points for each of the 4 edges of the face
    # (F, 4, 3)
    fp_expanded = face_points[:, np.newaxis, :].repeat(4, axis=1)
    
    # Flatten
    fp_flat = fp_expanded.reshape(-1, 3)
    edge_indices_flat = face_edge_indices.flatten()
    
    np.add.at(edge_points_sum, edge_indices_flat, fp_flat)
    np.add.at(edge_counts, edge_indices_flat, 1) # Add 1 for each face sharing the edge
    
    edge_points = edge_points_sum / edge_counts[:, np.newaxis]
    
    # 4. Correct Original Vertex Points (Smoothing)
    # Barycentric approx: V_new = (V_old + avg_face_points + avg_mid_edges) / 3 ? 
    # Proper CC: F_avg + 2*E_avg + (n-3)V / n
    # Too complex. We will just use the original vertices for now to save time, 
    # or apply simple Laplacian smoothing later.
    # Using original vertices with smoothed edge points gives a "puffy" result which is good.
    
    # Let's try to update original vertices simply: 
    # V_new = 0.5 * V + 0.5 * Avg(NewEdgePoints attached)
    # This smooths them.
    # We need to map Edge -> Vertex.
    # unique_edges has (v1, v2).
    # We can sum edge_points into vertices.
    
    vert_sum = np.zeros_like(vertices)
    vert_count = np.zeros(n_verts, dtype=np.int32)
    
    # Add edge points to both vertices of the edge
    np.add.at(vert_sum, unique_edges[:,0], edge_points)
    np.add.at(vert_count, unique_edges[:,0], 1)
    
    np.add.at(vert_sum, unique_edges[:,1], edge_points)
    np.add.at(vert_count, unique_edges[:,1], 1)
    
    # Avoid div zero
    vert_count[vert_count==0] = 1
    vert_avg_edges = vert_sum / vert_count[:, np.newaxis]
    
    # Apply soft smoothing
    new_orig_verts = 0.4 * vertices + 0.6 * vert_avg_edges
    
    # 5. Assemble Global Vertex List
    # Order: [Original Verts (Updated), Edge Points, Face Points]
    # Indices:
    # Orig: 0 .. n_verts-1
    # Edge: n_verts .. n_verts + n_edges - 1
    # Face: n_verts + n_edges .. end
    
    final_verts = np.concatenate([new_orig_verts, edge_points, face_points], axis=0)
    
    offset_edge = n_verts
    offset_face = n_verts + n_edges
    
    # 6. Build New Faces
    # For each face (v0, v1, v2, v3) with edge indices (e0, e1, e2, e3) and center c:
    # We want winding to match.
    # Edges are sorted (min, max). We need them directed along the face.
    # This is tricky.
    # Re-verify winding.
    # Face (v0, v1, v2, v3).
    # Edges passed to unique were (v0,v1), (v1,v2), (v2,v3), (v3,v0).
    # Stored unique edges are generic.
    # We handled `face_edge_indices` which corresponds to the order above.
    # So edge[0] connects v0-v1.
    # edge[1] connects v1-v2.
    # etc.
    
    # New Quads:
    # Q1: v0 -> edge0 -> center -> edge3
    # Q2: edge0 -> v1 -> edge1 -> center
    # Q3: center -> edge1 -> v2 -> edge2
    # Q4: edge3 -> center -> edge2 -> v3
    
    # Indices
    vals_v = faces # (F, 4)
    vals_e = face_edge_indices + offset_edge # (F, 4)
    vals_c = np.arange(n_faces) + offset_face # (F)
    
    # Reshape c for broadcasting
    vals_c = vals_c[:, np.newaxis]
    
    # Q1: v0, e0, c, e3
    q1 = np.stack([vals_v[:,0], vals_e[:,0], vals_c[:,0], vals_e[:,3]], axis=1)
    # Q2: e0, v1, e1, c
    q2 = np.stack([vals_e[:,0], vals_v[:,1], vals_e[:,1], vals_c[:,0]], axis=1)
    # Q3: c, e1, v2, e2  (Wait, v2 is corner. c is center. winding v2->e2->... or e1->v2->e2->c? )
    # Original: v0, v1, v2, v3 (CCW)
    # Q3 should be around v2. 
    # Vertices of Q3: edge1(between 1-2), v2, edge2(between 2-3), center.
    # Checks out: e1 -> v2 -> e2 -> c.
    q3 = np.stack([vals_e[:,1], vals_v[:,2], vals_e[:,2], vals_c[:,0]], axis=1)
    # Q4: e3, c, e2, v3 
    # Around v3.
    # e2(2-3), v3, e3(3-0), c.
    # Order: e2 -> v3 -> e3 -> c.
    q4 = np.stack([vals_e[:,2], vals_v[:,3], vals_e[:,3], vals_c[:,0]], axis=1)
    
    new_faces = np.concatenate([q1, q2, q3, q4], axis=0)
    
    return final_verts, new_faces
