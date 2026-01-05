import numpy as np

class Multivector5D:
    """
    Python implementation of 5D Conformal Geometric Algebra Cl(4,1).
    Optimized for batch processing using NumPy.
    
    Basis order (32 components):
    0: 1 (Scalar)
    1-5: e1, e2, e3, e+, e- (Vectors)
    6-15: Bivectors (e12, e13, e1+, e1-, e23, e2+, e2-, e3+, e3-, e+-)
    16-25: Trivectors (e123, e12+, e12-, e13+, e13-, e1+-, e23+, e23-, e2+-, e3+-)
    26-30: Quadvectors (e123+, e123-, e12+-, e13+-, e23+-)
    31: e123+- (Pseudoscalar)
    """

    def __init__(self, lanes=None):
        if lanes is None:
            self.lanes = np.zeros(32, dtype=np.float32)
        else:
            self.lanes = np.array(lanes, dtype=np.float32)

    @classmethod
    def basis(cls, i):
        m = cls()
        if 1 <= i <= 5:
            m.lanes[i] = 1.0
        return m

    @classmethod
    def n_inf(cls):
        """Conformal Null Basis: n_infinity = e- + e+"""
        m = cls()
        m.lanes[5] = 1.0  # e-
        m.lanes[4] = 1.0  # e+
        return m

    @classmethod
    def n_o(cls):
        """Conformal Null Basis: n_o = 0.5 * (e- - e+)"""
        m = cls()
        m.lanes[5] = 0.5   # e-
        m.lanes[4] = -0.5  # e+
        return m

    @classmethod
    def point(cls, x, y, z=0.0):
        """Maps a 3D coordinate to a Conformal Point"""
        e1 = cls.basis(1)
        e2 = cls.basis(2)
        e3 = cls.basis(3)
        no = cls.n_o()
        ninf = cls.n_inf()
        
        sq_dist = x*x + y*y + z*z
        return no + (e1 * x) + (e2 * y) + (e3 * z) + (ninf * (0.5 * sq_dist))

    def reverse(self):
        res = Multivector5D(self.lanes.copy())
        for i in range(32):
            grade = bin(i).count('1')
            if (grade * (grade - 1) // 2) % 2 == 1:
                res.lanes[i] *= -1.0
        return res

    def __add__(self, other):
        return Multivector5D(self.lanes + other.lanes)

    def __sub__(self, other):
        return Multivector5D(self.lanes - other.lanes)

    def __mul__(self, other):
        if isinstance(other, (int, float)):
            return Multivector5D(self.lanes * other)
        return self.geometric_product(other)

    def geometric_product(self, other):
        res_lanes = np.zeros(32, dtype=np.float32)
        # Using precomputed GP_MAP (computed below)
        for sign, a, b, k in GP_MAP:
            res_lanes[k] += sign * self.lanes[a] * other.lanes[b]
        return Multivector5D(res_lanes)

# --- Precomputation Logic ---

def basis_product_logic(a, b):
    sign = 1.0
    a_bits = a
    for i in range(5):
        if (b >> i) & 1:
            # Check signs for swaps
            for j in range(i + 1, 5):
                if (a_bits >> j) & 1:
                    sign *= -1.0
            # Metric check
            if (a_bits >> i) & 1:
                if i == 4: # e5 (e-) metric is -1
                    sign *= -1.0
                a_bits &= ~(1 << i)
            else:
                a_bits |= (1 << i)
    return sign, a_bits

# Initialize Cayley Table
CAYLEY_TABLE = []
for a in range(32):
    row = []
    for b in range(32):
        row.append(basis_product_logic(a, b))
    CAYLEY_TABLE.append(row)

# Initialize GP_MAP (Result-centric table)
GP_MAP = []
for k in range(32):
    for a in range(32):
        for b in range(32):
            sign, res_k = CAYLEY_TABLE[a][b]
            if res_k == k:
                GP_MAP.append((sign, a, b, k))

# Grade-based masks for multivector components
SCALAR_IDX = 0
VECTOR_INDICES = [i for i in range(32) if bin(i).count('1') == 1]
BIVECTOR_INDICES = [i for i in range(32) if bin(i).count('1') == 2]
TRIVECTOR_INDICES = [i for i in range(32) if bin(i).count('1') == 3]
QUADVECTOR_INDICES = [i for i in range(32) if bin(i).count('1') == 4]
PSEUDOSCALAR_IDX = 31

def batch_geometric_product(A, B):
    """
    A, B are numpy arrays or torch tensors of shape (..., 32)
    Returns product of shape (..., 32)
    """
    is_torch = hasattr(A, 'device') or hasattr(B, 'device')
    
    if is_torch:
        import torch
        # Ensure both are tensors on same device
        if not hasattr(A, 'device'): A = torch.from_numpy(A)
        if hasattr(B, 'device'): 
            A = A.to(B.device)
        else:
            B = torch.from_numpy(B).to(A.device)
            
        out = torch.zeros(list(torch.broadcast_shapes(A.shape[:-1], B.shape[:-1])) + [32], 
                         device=A.device, dtype=A.dtype)
        # Optimization: Group by sign to reduce number of additions
        for sign, a, b, k in GP_MAP:
            if sign == 1.0:
                out[..., k] += A[..., a] * B[..., b]
            else:
                out[..., k] -= A[..., a] * B[..., b]
        return out
    else:
        broadcast_shape = list(np.broadcast(A[..., 0], B[..., 0]).shape) + [32]
        out = np.zeros(broadcast_shape, dtype=np.float32)
        for sign, a, b, k in GP_MAP:
            out[..., k] += sign * A[..., a] * B[..., b]
        return out

def batch_wedge_product(A, B):
    """
    A, B: (..., 32)
    Returns A ^ B: (..., 32)
    """
    is_torch = hasattr(A, 'device') or hasattr(B, 'device')
    
    if is_torch:
        import torch
        if not hasattr(A, 'device'): A = torch.from_numpy(A)
        if hasattr(B, 'device'): A = A.to(B.device)
        else: B = torch.from_numpy(B).to(A.device)
        
        out = torch.zeros(list(torch.broadcast_shapes(A.shape[:-1], B.shape[:-1])) + [32], 
                         device=A.device, dtype=A.dtype)
        for sign, a, b, k in GP_MAP:
            grade_a = bin(a).count('1')
            grade_b = bin(b).count('1')
            grade_k = bin(k).count('1')
            if grade_k == (grade_a + grade_b):
                if sign == 1.0:
                    out[..., k] += A[..., a] * B[..., b]
                else:
                    out[..., k] -= A[..., a] * B[..., b]
        return out
    else:
        broadcast_shape = list(np.broadcast(A[..., 0], B[..., 0]).shape) + [32]
        out = np.zeros(broadcast_shape, dtype=np.float32)
        for sign, a, b, k in GP_MAP:
            grade_a = bin(a).count('1')
            grade_b = bin(b).count('1')
            grade_k = bin(k).count('1')
            if grade_k == (grade_a + grade_b):
                out[..., k] += sign * A[..., a] * B[..., b]
        return out

def batch_inner_product(A, B):
    """
    Proper Bilinear Form for Cl(4,1).
    Calculates the Scalar part of (A * reverse(B)).
    """
    is_torch = hasattr(A, 'device') or hasattr(B, 'device')
    
    # 1. Reverse B
    if is_torch:
        import torch
        if not hasattr(B, 'device'): B = torch.from_numpy(B)
        B_rev = B.clone()
        for i in range(32):
            grade = bin(i).count('1')
            if (grade * (grade - 1) // 2) % 2 == 1:
                B_rev[..., i] *= -1.0
        
        if not hasattr(A, 'device'): A = torch.from_numpy(A).to(B.device)
        else: B_rev = B_rev.to(A.device)
        
        res_scalar = torch.zeros(torch.broadcast_shapes(A.shape[:-1], B.shape[:-1]), 
                                device=A.device, dtype=A.dtype)
        for sign, a, b, k in GP_MAP:
            if k == 0:
                if sign == 1.0:
                    res_scalar += A[..., a] * B_rev[..., b]
                else:
                    res_scalar -= A[..., a] * B_rev[..., b]
        return res_scalar
    else:
        B_rev = B.copy()
        for i in range(32):
            grade = bin(i).count('1')
            if (grade * (grade - 1) // 2) % 2 == 1:
                B_rev[..., i] *= -1.0
        res_scalar = np.zeros(np.broadcast(A[..., 0], B[..., 0]).shape, dtype=np.float32)
        for sign, a, b, k in GP_MAP:
            if k == 0:
                res_scalar += sign * A[..., a] * B_rev[..., b]
        return res_scalar

def exp_map(bivector_batch):
    """
    Maps a batch of bivectors to Rotors using the exponential map.
    R = exp(B) approx 1 + B + B^2/2! + ...
    """
    is_torch = hasattr(bivector_batch, 'device')
    if is_torch:
        import torch
        identity = torch.zeros_like(bivector_batch)
        identity[..., 0] = 1.0
        
        # STABILITY FIX: Clamp bivector magnitude
        # Taylor series diverges if ||B|| is large.
        # We treat large rotations as multiple small ones or just clamp.
        B = torch.clamp(bivector_batch, min=-1.0, max=1.0) 
        
        # Taylor expansion (3rd order for better precision)
        B2 = batch_geometric_product(B, B)
        B3 = batch_geometric_product(B2, B)
        return identity + B + (B2 * 0.5) + (B3 * (1.0/6.0))
    else:
        # Numpy implementation (keep as is)
        identity = np.zeros(bivector_batch.shape, dtype=np.float32)
        identity[..., 0] = 1.0
        B = bivector_batch
        B2 = batch_geometric_product(B, B)
        return identity + B + (B2 * 0.5)

def inverse(R):
    """
    Rotor inverse: R^-1 = Reverse(R) / (R * Reverse(R))_scalar
    For unit rotors, Inverse == Reverse.
    """
    is_torch = hasattr(R, 'device')
    if is_torch:
        import torch
        R_rev = R.clone()
        for i in range(32):
            grade = bin(i).count('1')
            if (grade * (grade - 1) // 2) % 2 == 1:
                R_rev[..., i] *= -1.0
        return R_rev # Assuming unit rotor for now
    else:
        R_rev = R.copy()
        for i in range(32):
            grade = bin(i).count('1')
            if (grade * (grade - 1) // 2) % 2 == 1:
                R_rev[..., i] *= -1.0
        return R_rev

def normalize_rotor(R):
    """Ensures rotor stays on the manifold (unit norm)"""
    is_torch = hasattr(R, 'device')
    if is_torch:
        import torch
        norm = torch.norm(R, dim=-1, keepdim=True)
        return R / (norm + 1e-8)
    else:
        norm = np.linalg.norm(R, axis=-1, keepdims=True)
        return R / (norm + 1e-8)

def hodge_dual(A):
    """
    Returns the Hodge Dual of a multivector in Cl(4,1).
    A* = A * Inverse(Pseudoscalar)
    """
    if hasattr(A, 'detach'): A = A.detach().numpy()
    
    pseudoscalar = np.zeros(32, dtype=np.float32)
    pseudoscalar[31] = 1.0 # e123+-
    
    # In Cl(4,1), the pseudoscalar squares to -1.
    # So Inverse(I) = -I.
    inv_ps = -pseudoscalar
    return batch_geometric_product(A, inv_ps[np.newaxis, :])

# --- Conformal Indices (Bit-based) ---
E1_IDX = 1
E2_IDX = 2
E3_IDX = 4
EP_IDX = 8  # e+
EM_IDX = 16 # e-

def batch_point(x, y, z):
    """
    Maps 3D coordinates to Conformal Points (Batch).
    P = no + x*e1 + y*e2 + z*e3 + 0.5*|x|^2*ninf
    """
    if hasattr(x, 'shape'): n = x.shape[0]
    else: n = 1; x=np.array([x]); y=np.array([y]); z=np.array([z])
        
    out = np.zeros((n, 32), dtype=np.float32)
    
    # n_o = 0.5*(e- - e+)
    out[..., EM_IDX] += 0.5
    out[..., EP_IDX] -= 0.5
    
    out[..., E1_IDX] = x
    out[..., E2_IDX] = y
    out[..., E3_IDX] = z
    
    # n_inf = e- + e+
    sq_dist = x**2 + y**2 + z**2
    term = 0.5 * sq_dist
    out[..., EM_IDX] += term
    out[..., EP_IDX] += term
    
    return out

def batch_sphere(center_x, center_y, center_z, radius):
    """
    S = Point(center) - 0.5 * radius^2 * ninf
    """
    p = batch_point(center_x, center_y, center_z)
    term = 0.5 * (radius**2)
    p[..., EM_IDX] -= term
    p[..., EP_IDX] -= term
    return p

def batch_plane(nx, ny, nz, distance):
    """
    L = nx*e1 + ny*e2 + nz*e3 + distance*ninf
    """
    if hasattr(nx, 'shape'): n = nx.shape[0]
    else: n = 1; nx=np.array([nx]); ny=np.array([ny]); nz=np.array([nz]); distance=np.array([distance])
        
    out = np.zeros((n, 32), dtype=np.float32)
    out[..., E1_IDX] = nx
    out[..., E2_IDX] = ny
    out[..., E3_IDX] = nz
    out[..., EM_IDX] += distance
    out[..., EP_IDX] += distance
    return out

def compute_cga_distance(A, B):
    """
    Calculates the squared distance between two conformal objects.
    d^2 = -2 * <A, B> (for normalized points)
    """
    return -2.0 * batch_inner_product(A, B)

def compute_gca_bias_batch(psi, Q_lifted, K_lifted):
    """
    psi: (num_heads, 32) - The Context Rotor
    Q_lifted: (batch, n_heads, seq_len, 32)
    K_lifted: (batch, n_heads, seq_len, 32)
    
    Returns: (batch, n_heads, seq_len, seq_len)
    Formula: <PSI, Q ^ K>
    """
    batch, n_heads, seq_len, _ = Q_lifted.shape
    
    # 1. Compute pairwise wedge product Q_i ^ K_j
    # This involves a large broadcast: (batch, n_heads, seq_len, 1, 32) ^ (batch, n_heads, 1, seq_len, 32)
    # Result: (batch, n_heads, seq_len, seq_len, 32)
    Q_exp = Q_lifted[:, :, :, np.newaxis, :]
    K_exp = K_lifted[:, :, np.newaxis, :, :]
    
    # Pairwise bivector relationship: Relationship_Plane(i, j)
    rel_plane = batch_wedge_product(Q_exp, K_exp)
    
    # 2. Inner product with PSI: <PSI, Relationship_Plane>
    # psi is (n_heads, 32), we broadcast to (batch, n_heads, seq_len, seq_len, 32)
    psi_exp = psi[np.newaxis, :, np.newaxis, np.newaxis, :]
    
    # The GCA bias essentially checks how much the current Context "agrees" with the 
    # plane of the relationship between two tokens.
    bias = batch_inner_product(psi_exp, rel_plane)
    
    return bias
