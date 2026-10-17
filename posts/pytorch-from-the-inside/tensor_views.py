import torch

x = torch.arange(12.0).reshape(3, 4)
print(f"x: shape {tuple(x.shape)}, strides {x.stride()}, contiguous {x.is_contiguous()}")
t = x.T                                            # a transpose is a view: new strides, same memory
print(f"x.T: shape {tuple(t.shape)}, strides {t.stride()}, contiguous {t.is_contiguous()}, "
      f"same memory {t.data_ptr() == x.data_ptr()}")
row, col = x[1], x[:, 1]                           # slicing is a view too, with an offset
print(f"x[1]: offset {row.storage_offset()}, strides {row.stride()};  "
      f"x[:, 1]: offset {col.storage_offset()}, strides {col.stride()}")
t[0, 1] = 100.0                                    # writing through a view writes the original
print(f"after t[0, 1] = 100: x[1, 0] = {x[1, 0].item()}")
c = t.contiguous()                                 # a real copy, laid out row by row
print(f"t.contiguous(): strides {c.stride()}, same memory {c.data_ptr() == x.data_ptr()}")
try:
    t.view(12)
except RuntimeError as e:
    print(f"t.view(12) fails: {str(e).split('.')[0]}")
print(f"t.reshape(12) works (it copies when it must): {t.reshape(12)[:4].tolist()}")