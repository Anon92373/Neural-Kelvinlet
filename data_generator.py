import numpy as np
from scipy.spatial import KDTree

class MeshDisplacementGenerator:
    def __init__(self, points, normals, M_to_N_indices,
                 min_neighbors=3,
                 max_neighbors=7,
                 displacement_range=(-0.01, 0.02),  # in meters (-1cm to 2cm)
                 max_cone_angle=np.pi/6):  # 30 degrees cone
        """
        Initialize the mesh displacement generator.
        
        Args:
            points: (M, 3) array of mesh vertex positions
            normals: (M, 3) array of vertex normals
            M_to_N_indices: indices mapping each M point to its corresponding N point
            min_neighbors: minimum number of neighbors to average
            max_neighbors: maximum number of neighbors to average
            displacement_range: (min, max) range for displacement magnitude in meters
            max_cone_angle: maximum angle of the cone for displacement direction
        """
        self.points = points
        self.normals = normals
        self.M_to_N_indices = M_to_N_indices
        self.min_neighbors = min_neighbors
        self.max_neighbors = max_neighbors
        self.displacement_range = displacement_range
        self.max_cone_angle = max_cone_angle
        
        # Initialize KD-tree for neighbor search
        self.tree = KDTree(points)
        
        # Get size of N mesh
        self.N = 10400
        self.M = len(points)
        
        # Validate inputs
        self._validate_inputs()
    
    def _validate_inputs(self):
        """Validate input data dimensions and parameters."""
        if self.points.shape[1] != 3:
            raise ValueError("Points must be a (M, 3) array")
        if self.normals.shape != self.points.shape:
            raise ValueError("Normals must have same shape as points")
        if len(self.M_to_N_indices) != self.M:
            raise ValueError("M_to_N_indices must have length M")
        if self.min_neighbors >= self.max_neighbors:
            raise ValueError("min_neighbors must be less than max_neighbors")
        if self.displacement_range[0] >= self.displacement_range[1]:
            raise ValueError("displacement_range[0] must be less than displacement_range[1]")
    
    def _generate_cone_direction(self, avg_normal):
        """Generate a random direction within a cone around the average normal."""
        theta = np.random.uniform(0, self.max_cone_angle)
        phi = np.random.uniform(0, 2 * np.pi)
        
        # Create perpendicular vectors
        v1 = np.array([1, 0, 0]) if not np.allclose(avg_normal, [1, 0, 0]) else np.array([0, 1, 0])
        perp1 = np.cross(avg_normal, v1)
        perp1 = perp1 / np.linalg.norm(perp1)
        perp2 = np.cross(avg_normal, perp1)
        
        # Calculate displacement direction
        return (avg_normal * np.cos(theta) + 
                perp1 * np.sin(theta) * np.cos(phi) +
                perp2 * np.sin(theta) * np.sin(phi))
    
    def generate_single_displacement(self):
        """Generate a single displacement pattern."""
        # 1. Randomly select a seed point
        seed_idx = np.random.randint(0, self.M)
        
        # 2. Get random number of neighbors
        n_neighbors = np.random.randint(self.min_neighbors, self.max_neighbors + 1)
        
        # 3. Find nearest neighbors
        distances, indices = self.tree.query(self.points[seed_idx], k=n_neighbors)
        # 4. Average normals of the selected region
        avg_normal = np.mean(self.normals[indices], axis=0)
        avg_normal = avg_normal / np.linalg.norm(avg_normal)
        
        # 5. Generate random displacement
        magnitude = np.random.uniform(*self.displacement_range)
        direction = self._generate_cone_direction(avg_normal)
        displacement = magnitude * direction
        
        # 6. Create displacement arrays
        displacement_M = np.zeros_like(self.points)
        displacement_M[indices] = displacement
        
        displacement_N = np.zeros((self.N, 3))
        active_flag_N = np.zeros(self.N, dtype=bool)
        
        # 7. Map to N points
        affected_N_indices = self.M_to_N_indices[indices]
        displacement_N[affected_N_indices] = displacement
        active_flag_N[affected_N_indices] = True
        
        return displacement_N, active_flag_N, indices
    
    def generate_batch(self, n_displacements):
        """
        Generate multiple displacement patterns.
        
        Args:
            n_displacements: number of displacement patterns to generate
            
        Returns:
            displacements_M: List of (M, 3) arrays with displacements for M mesh
            displacements_N: List of (N, 3) arrays with mapped displacements
            active_flags_N: List of (N,) boolean arrays indicating active nodes
            affected_indices_M: List of arrays containing affected vertex indices in M mesh
        """

        displacements_N = []
        active_flags_N = []
        
        for _ in range(n_displacements):
            disp_N, active_N, indices = self.generate_single_displacement()
            displacements_N.append(disp_N)
            active_flags_N.append(active_N)
        
        return np.array(displacements_N), np.array(active_flags_N)
    
    def get_mesh_info(self):
        """Return basic information about the meshes."""
        return {
            'M_points': self.M,
            'N_points': self.N,
            'min_neighbors': self.min_neighbors,
            'max_neighbors': self.max_neighbors,
            'displacement_range': self.displacement_range,
            'max_cone_angle_degrees': np.degrees(self.max_cone_angle)
        }
