from typing import Tuple
import torch

def gen_torch_tensor_shape_assertion(target_name: str, target: torch.tensor, expected: Tuple) -> str:
    assert target.shape == expected, f'The shape of {target_name} is abnormal. {target_name}.shape:{target.shape}, expected:{expected}'