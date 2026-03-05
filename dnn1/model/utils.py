import torch

def amounts_to_class(amounts):
    i0, i1 = torch.nonzero(amounts)
    cnt_0 = amounts[i0]

    mask_encoding = i0 * (9 - i0) / 2 + i1 - 1
    return (9 * mask_encoding + cnt_0 - 1).to(torch.long)

def class_to_pair_encoding(cls):
    return cls // 9


def test_class_translation():
    class_to_shapes = {}

    s = set()
    for i in range(5):
        for j in range(i + 1, 6):
            for k in range(1, 10):
                # k times elemnt i and 10 - k times element j
                t = torch.zeros(6)
                t[i] = k
                t[j] = 10 - k
                cls = amounts_to_class(t)
                
                s.add(cls)
                class_to_shapes[cls.item()] = (i, j)

    assert len(s) == 135 and min(s) == 0 and max(s) == 134

    s_a = set(range(135))
    s = set(i.item() for i in s)

    assert s == s_a