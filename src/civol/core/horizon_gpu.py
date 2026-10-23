"""GPU paired-twin integration (torch). Same law as horizon._batch.

Standalone script for the torch environment: reads a model .npz, writes batch
moments to an output .npz. Path simulation in float32, moment aggregation in
float64. Invoked by horizon.paired_moments(backend='gpu').
"""
import sys
import numpy as np
import torch


def run(path_in, path_out):
    m = np.load(path_in)
    H = int(m['horizon']); pairs = int(m['pairs']); batches = int(m['batches']); seed = int(m['seed'])
    drivers = [int(x) for x in m['drivers']]          # 0 = total shock, 1 = private shock; (transitory, level)
    dev = torch.device('cuda')
    f32 = dict(dtype=torch.float32, device=dev)
    mu = torch.tensor(m['mu'], **f32); b = torch.tensor(m['loadings'], **f32); d = torch.tensor(m['d'], **f32)
    v = torch.tensor(m['reference'], **f32); h0 = torch.tensor(m['h_next'], **f32); m0 = torch.tensor(m['m_next'], **f32)
    coeff = torch.tensor(m['coefficients'], **f32)
    alpha, beta, gamma, rho_m, phi = (coeff[:, j] for j in range(5))
    omega = torch.tensor(m['factor_omega'], **f32); af = torch.tensor(m['factor_alpha'], **f32); bf = torch.tensor(m['factor_beta'], **f32)
    g_bound = torch.tensor(m['g_bound'], **f32)
    g0 = torch.minimum(torch.tensor(m['g_next'], **f32), g_bound)
    N = mu.numel(); K = g0.numel(); count = pairs // batches
    g_contacts = 0
    bt = b.t().contiguous(); bb_t = (b * b).t().contiguous()
    out_q = np.empty((batches, N)); out_ipv = np.empty((batches, N)); out_mean = np.empty((batches, N)); out_second = np.empty((batches, N))
    lower = 0; upper = 0

    def update(total_shock, eps, h, lev):
        nonlocal lower, upper
        s_total = total_shock * total_shock / v
        r = h * eps * eps
        if drivers[0] == 0:
            s = s_total; neg = (total_shock < 0)
        else:
            s = r; neg = (eps < 0)
        l = s_total if drivers[1] == 0 else r
        pm = 1 - rho_m + rho_m * lev + phi * (l - h)
        lower += int((pm <= .05).sum()); upper += int((pm >= 25.).sum())
        pm = pm.clamp(.05, 25.)
        ph = pm + beta * (h - lev) + alpha * (s - lev) + gamma * (neg * s - .5 * lev)
        lower += int((ph <= .05).sum()); upper += int((ph >= 25.).sum())
        return ph.clamp(.05, 25.), pm

    for batch in range(batches):
        gen = torch.Generator(device=dev); gen.manual_seed(seed * 1000003 + batch)
        g = g0.expand(count, K).clone()
        ha = h0.expand(count, N).clone(); hb = ha.clone()
        ma_ = m0.expand(count, N).clone(); mb_ = ma_.clone()
        pa = torch.zeros(count, N, **f32); pb = torch.zeros(count, N, **f32)
        ipv = (d * (ha + hb) * .5).double()
        for day in range(H - 1):
            f = torch.sqrt(g) * torch.randn(count, K, generator=gen, **f32)
            g = omega + af * f * f + bf * g
            g_contacts += int((g >= g_bound).sum()); g = torch.minimum(g, g_bound)     # bounded common variance (finite moments)
            common = f @ bt
            ea = torch.randn(count, N, generator=gen, **f32); eb = torch.randn(count, N, generator=gen, **f32)
            sa = common + torch.sqrt(ha * d) * ea
            sb = common + torch.sqrt(hb * d) * eb
            pa += mu + sa; pb += mu + sb
            ha, ma_ = update(sa, ea, ha, ma_)
            hb, mb_ = update(sb, eb, hb, mb_)
            ipv += (d * (ha + hb) * .5).double()
        c = (g @ bb_t).double()
        da = (ha * d).double(); db = (hb * d).double(); mud = mu.double()
        ma = torch.exp(pa.double() + mud + .5 * da); mb = torch.exp(pb.double() + mud + .5 * db)
        e2c = torch.exp(2 * c)
        q = .5 * e2c * ((ma - mb) ** 2 + ma * ma * torch.expm1(da) + mb * mb * torch.expm1(db))
        mean = torch.exp(.5 * c) * .5 * (ma + mb)
        second = .5 * e2c * (ma * ma * torch.exp(da) + mb * mb * torch.exp(db))
        out_q[batch] = q.mean(0).cpu().numpy(); out_ipv[batch] = ipv.mean(0).cpu().numpy()
        out_mean[batch] = mean.mean(0).cpu().numpy(); out_second[batch] = second.mean(0).cpu().numpy()
    np.savez(path_out, q=out_q, ipv=out_ipv, mean=out_mean, second=out_second, lower=lower, upper=upper, g_contacts=g_contacts,
             device=str(torch.cuda.get_device_name(0)), torch_version=torch.__version__)


if __name__ == '__main__':
    run(sys.argv[1], sys.argv[2])
