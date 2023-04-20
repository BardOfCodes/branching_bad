import numpy as np
import torch as th
import math


class Transforms2D():
    """Transforms is a class to build 4x4 affine transformation matrices.
    It takes numpy arrays as input and returns torch tensors.
    """

    def __init__(self, device):
        self.device = device
        self.setup()

    def setup(self):
        self.homogeneous_identity = th.eye(3, device=self.device)
        self.zero_cube = th.zeros(2, 2, device=self.device)
        # For affine -> rotation quaternions
        # self.matrix_k = th.tensor(
        #     [[1, -1, -1], [-1, 1, -1], [-1, -1, 1]]).to(self.device)

    def get_affine_identity(self):
        """Return an identity affine matrix.
        """
        return self.homogeneous_identity.clone()

    def get_affine_translate(self, param):
        """Return an affine matrix for translation.

        Args:
            param (np.ndarray)
        """
        matrix = self.homogeneous_identity.clone()
        param = th.from_numpy(param.astype(np.float32)).to(self.device)
        matrix[:2, 2] = param
        return matrix

    def get_affine_rotate(self, param):
        """Return an affine matrix for rotation.

        Args:
            param (np.ndarray)
        """
        sin = math.sin(param[0])
        cos = math.cos(param[0])
        affine_matrix = np.array(
            [[cos, -sin, 0], [sin, cos, 0], [0, 0, 1]]).astype(np.float32)
        affine_matrix = th.from_numpy(affine_matrix).to(self.device)
        return affine_matrix

    def get_affine_scale(self, param):
        """Return an affine matrix for scale.

        Args:
            scale (_type_): _description_
        """
        param = th.from_numpy(param.astype(np.float32)).to(self.device)
        affine_matrix = self.homogeneous_identity.clone()
        affine_matrix[0, 0] = param[0]
        affine_matrix[1, 1] = param[1]
        return affine_matrix

    def decouple_affine(self, affine_matrix):
        """Decouple an affine matrix into translation, rotation and scale.

        Args:
            affine_matrix (np.ndarray): _description_
        """
        translation = affine_matrix[:3, 3]
        scale = th.sqrt(th.sum(affine_matrix[:3, :3]**2, dim=0))

        rotation = affine_matrix[:3, :3] / th.reshape(scale, (3, 1))
        q = th.zeros(4, device=self.device)
        q[1:] = self.matrix_k[th.argmax(th.diag(rotation))] * th.diag(rotation)
        q[0] = th.sqrt(th.maximum(0.0, 1.0 + th.sum(q[1:])))
        q = q / th.norm(q)

        return translation, q, scale

    def alternative_decouple_affine(self, affine_matrix):
        # TODO: Check if this is faster
        translation = affine_matrix[:3, 3]
        scale = th.sqrt(th.sum(affine_matrix[:3, :3]**2, dim=0))

        rotation = affine_matrix[:3, :3] / th.reshape(scale, (3, 1))
        q = th.empty(4, device=rotation.device)
        q[3] = 0.5 * th.sqrt(th.max(th.tensor([0.0], device=rotation.device),
                             1.0 + rotation[0, 0] + rotation[1, 1] + rotation[2, 2]))
        q[0] = 0.5 * th.sqrt(th.max(th.tensor([0.0], device=rotation.device),
                             1.0 + rotation[0, 0] - rotation[1, 1] - rotation[2, 2]))
        q[1] = 0.5 * th.sqrt(th.max(th.tensor([0.0], device=rotation.device),
                             1.0 - rotation[0, 0] + rotation[1, 1] - rotation[2, 2]))
        q[2] = 0.5 * th.sqrt(th.max(th.tensor([0.0], device=rotation.device),
                             1.0 - rotation[0, 0] - rotation[1, 1] + rotation[2, 2]))
        q[1] *= th.sign(rotation[2, 1] - rotation[1, 2])
        q[2] *= th.sign(rotation[0, 2] - rotation[2, 0])
        q[3] *= th.sign(rotation[1, 0] - rotation[0, 1])
        q = q / th.norm(q)

        return translation, q, scale


def transform_points(points, affine_matrix):
    """Transforms a sequence of points by an affine matrix.
    Args:
        points (th.Tensor): B, N, 3
        affine_matrix (th.Tensor): B, 4, 4
    """
    points = th.cat(
        [points, th.ones(points.size(0), points.size(1), 1)], dim=2)

    # Apply the affine transformation matrix to the points
    transformed_points = th.matmul(
        affine_matrix, points.unsqueeze(-1)).squeeze(-1)

    # Remove the homogeneous coordinate and return the transformed points
    transformed_points = transformed_points[..., :3]
    return transformed_points
