import concurrent.futures
import dataclasses
import math
import numpy as np
import unittest


@dataclasses.dataclass
class Matrix:
  vals: list[list[float]]

  @classmethod
  def from_ndarray(cls, x: np.ndarray) -> "Matrix":
    assert x.ndim == 2
    return Matrix(x.tolist())

  @classmethod
  def zeros(cls, num_rows: int, num_cols: int) -> "Matrix":
    assert num_rows > 0
    assert num_cols > 0
    # this was the first bug line
    return Matrix([[0.] * num_cols] * num_rows)
    # the fix I did (after surprisingly and embarrassingly long time)
    # return Matrix([[0. for _ in range(num_cols)] for _ in range(num_rows)])

  def to_ndarray(self) -> np.ndarray:
    return np.array(self.vals)

  @property
  def num_rows(self) -> int:
    return len(self.vals)

  @property
  def num_cols(self) -> int:
    return len(self.vals[0])

  def add(self, other: "Matrix") -> None:
    assert len(self.vals) == len(other.vals)
    assert len(self.vals[0]) == len(other.vals[0])
    for i in range(len(self.vals)):
      for j in range(len(self.vals[0])):
        self.vals[i][j] += other.vals[i][j]

def simple_matmul(A: Matrix, B: Matrix) -> Matrix:
  assert A.num_cols == B.num_rows
  C = Matrix.zeros(A.num_rows, B.num_cols)

  for i in range(A.num_rows):
    for j in range(B.num_cols):
      for k in range(A.num_cols):
        C.vals[i][j] += A.vals[i][k] * B.vals[k][j]
  return C


class SimpleMatmulTest(unittest.TestCase):

  def test_matrix(self):
    m = np.array([[1.0, 0.2], [3.0, 0.4]])
    np.testing.assert_almost_equal(Matrix.from_ndarray(m).to_ndarray(), m)

  def test_simple_matmul(self):
    A = np.random.rand(4, 2)
    B = np.random.rand(2, 3)
    result = simple_matmul(Matrix.from_ndarray(A), Matrix.from_ndarray(B))
    expected = np.matmul(A, B)
    np.testing.assert_almost_equal(result.to_ndarray(), expected)


# --------------------------------------------------------------------
import math

MAX_DIM = 16

@dataclasses.dataclass
class DistributedMatrix:
  num_rows: int
  num_cols: int
  shards: list[list[Matrix]]

  @classmethod
  def from_ndarray(cls, x: np.ndarray) -> "DistributedMatrix":
    assert x.ndim == 2
    dm = DistributedMatrix(x.shape[0], x.shape[1], [])

    # 3rd bug was using floor division by MAX_DIM like x.shape[0] // MAX_DIM - which is common across the class in multiple places - should be ceil division
    for block_row in range(math.ceil(x.shape[0] / MAX_DIM)):
      row_start = block_row * MAX_DIM
      dm.shards.append([])
      for block_col in range(math.ceil(x.shape[1] / MAX_DIM)):
        col_start = block_col * MAX_DIM
        m = Matrix.from_ndarray(x[row_start:(row_start + MAX_DIM), col_start:(col_start + MAX_DIM)])
        dm.shards[-1].append(m)

    return dm

  @classmethod
  def zeros(cls, num_rows: int, num_cols: int) -> "DistributedMatrix":
    assert num_rows > 0
    assert num_cols > 0
    return DistributedMatrix.from_ndarray(np.zeros((num_rows, num_cols)))

  def to_ndarray(self) -> np.ndarray:
    # second bug was axis=1 missing from the concat
    rows = [np.concatenate([j.to_ndarray() for j in i], axis=1) for i in self.shards]
    return np.concatenate(rows)[:self.num_rows, :self.num_cols]


def distributed_matmul(A: DistributedMatrix, B: DistributedMatrix, max_workers: int = 1) -> DistributedMatrix:
  assert A.num_cols == B.num_rows
  C = DistributedMatrix.zeros(A.num_rows, B.num_cols)

  
  with concurrent.futures.ProcessPoolExecutor(max_workers) as pool:
    for block_i in range(math.ceil(C.num_rows / MAX_DIM)):
      for block_j in range(math.ceil(C.num_cols / MAX_DIM)):
        for block_k in range(math.ceil(A.num_cols / MAX_DIM)):
          future = pool.submit(simple_matmul, A.shards[block_i][block_k], B.shards[block_k][block_j])
          future.add_done_callback(lambda future: C.shards[block_i][block_j].add(future.result()))


  
  return C


class DistributedMatmulTest(unittest.TestCase):

  def test_distributed_matrix(self):
    m = np.random.rand(32, 64)
    dm = DistributedMatrix.from_ndarray(m)
    np.testing.assert_almost_equal(dm.to_ndarray(), m)

    m = np.random.rand(17, 19)
    dm = DistributedMatrix.from_ndarray(m)
    np.testing.assert_almost_equal(dm.to_ndarray(), m)

  def test_distributed_matmul(self):
    A = np.random.rand(4, 3)
    B = np.random.rand(3, 2)
    result = distributed_matmul(DistributedMatrix.from_ndarray(A), DistributedMatrix.from_ndarray(B))
    expected = np.matmul(A, B)
    np.testing.assert_almost_equal(result.to_ndarray(), expected)

    A = np.random.rand(32, 16)
    B = np.random.rand(16, 32)
    result = distributed_matmul(DistributedMatrix.from_ndarray(A), DistributedMatrix.from_ndarray(B), max_workers=4)
    expected = np.matmul(A, B)
    np.testing.assert_almost_equal(result.to_ndarray(), expected)

    A = np.random.rand(133, 179)
    B = np.random.rand(179, 231)
    result = distributed_matmul(DistributedMatrix.from_ndarray(A), DistributedMatrix.from_ndarray(B), max_workers=10)
    expected = np.matmul(A, B)
    np.testing.assert_almost_equal(result.to_ndarray(), expected)

if __name__ == '__main__':
  unittest.main()