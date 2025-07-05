import torch
import torch.nn.functional as F

def js_divergence(p, q, eps=1e-12):
    p = p + eps
    q = q + eps
    p = p / p.sum()
    q = q / q.sum()
    m = 0.5 * (p + q)

    kl_pm = F.kl_div(m.log(), q, reduction='none', log_target=False)
    kl_qm = F.kl_div(m.log(), p, reduction='none', log_target=False)

    js = 0.5 * (kl_pm + kl_qm)
    return js

p = torch.tensor([0.0391, 0.0452, 0.0395, 0.0430, 0.0630, 0.0561, 0.0395, 0.0584, 0.0620,
        0.0454, 0.0571, 0.0512, 0.0401, 0.0538, 0.0449, 0.0444, 0.0414, 0.0497,
        0.0368, 0.0371, 0.0522])
q = torch.tensor([0.0000e+00, 2.8026e-45, 1.5882e-33, 1.5614e-23, 2.3404e-15, 5.3483e-09,
        1.8634e-04, 9.8985e-02, 8.0166e-01, 9.8985e-02, 1.8634e-04, 5.3483e-09,
        2.3404e-15, 1.5614e-23, 1.5882e-33, 2.8026e-45, 0.0000e+00, 0.0000e+00,
        0.0000e+00, 0.0000e+00, 0.0000e+00])

js_val = js_divergence(p, q)
print(f"JS divergence: {js_val}")
