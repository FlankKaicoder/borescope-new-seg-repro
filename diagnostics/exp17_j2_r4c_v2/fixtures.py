"""Six small tensor fixtures, exactly twelve budgeted engine calls."""
import contextlib
import numpy as np
from core import bridge_type, write

def verify(torch, budget, root):
    Bridge = bridge_type(torch); reports = []
    rng = np.random.default_rng(1742)
    for case in range(1, 7):
        dtype = torch.float64 if case <= 4 else torch.float32
        na, nn = (4, 3) if case != 2 else (5, 2)
        arrays = [rng.normal(size=(na, 2)), rng.normal(size=(nn, 2)),
                  np.array([-.8, 1.1]), np.array([.3, -.2])]
        h = rng.normal(size=(nn, 2))
        if case == 3:
            arrays[0][:, 0] = .4; arrays[2][1] = 0.; h[:, 0] -= h[:, 0].mean()
        if case == 5:
            arrays[0] += 1024.; arrays[1] += 1024.
        if case == 4:
            arrays.extend([np.array([[.7, -.2], [.3, 1.2]]), np.array([.9, -.6]), np.array([.1, .2])])
        eps = .001

        def objective(values, xp, bridged=False):
            a, n, gamma, beta = values[:4]
            if xp is np:
                mu = a.mean(0); var = ((a-mu)**2).mean(0)
            else: var, mu = torch.var_mean(a, dim=0, unbiased=False)
            def norm(x, m, v, g, b): return (x-m)/(v+eps)**.5*g+b
            def replay(x, m, v, g, b):
                if not bridged: return norm(x, m, v, g, b)
                base = norm(x, m.detach(), v.detach(), g, b)
                # Bridge expects [N,C,...], here [N,C].
                out = Bridge.apply(base, m, v, x.detach(), g.detach(), eps, None)
                assert torch.equal(out, base)
                return out
            out = replay(n, mu, var, gamma, beta)
            if case == 4:
                w, g2, b2 = values[4:]
                a2 = xp.tanh(norm(a, mu, var, gamma, beta)) @ w
                n2 = xp.tanh(out) @ w
                if xp is np: m2=a2.mean(0); v2=((a2-m2)**2).mean(0)
                else: v2,m2=torch.var_mean(a2,dim=0,unbiased=False)
                out = replay(n2,m2,v2,g2,b2)
            if case == 6:
                if xp is np: out=out/(1+np.exp(-out))
                else: out=torch.nn.functional.silu(out,inplace=True)
            hh = h if xp is np else torch.tensor(h,dtype=dtype)
            centered_a=a-(1024. if case==5 else 0.)
            return (out*hh).sum() + .07*(centered_a*centered_a).sum(), out

        observed=[]; outputs=[]
        for kind in ('bridge','reference'):
            values=[torch.tensor(x,dtype=dtype,requires_grad=True) for x in arrays]
            context=torch.autograd.graph.save_on_cpu(pin_memory=False) if case==6 else contextlib.nullcontext()
            with context: loss,out=objective(values,torch,kind=='bridge')
            budget.begin(f'cpu_{case}_{kind}')
            gradients=torch.autograd.grad(loss,values,allow_unused=False)
            budget.commit(loss=float(loss.detach()))
            assert all(x.grad is None for x in values)
            observed.append([g.detach().double().numpy() for g in gradients]);outputs.append(out.detach())
        atol,rtol=(1e-10,1e-9) if case<=4 else (1e-6,1e-5)
        for a,b in zip(*observed,strict=True): np.testing.assert_allclose(a,b,atol=atol,rtol=rtol)
        assert torch.equal(outputs[0],outputs[1])
        # Independent finite differences of float64 mathematical reference.
        numerical=[]; delta=1e-5
        for k,x in enumerate(arrays):
            grad=np.empty_like(x)
            for idx in np.ndindex(x.shape):
                plus=[v.copy() for v in arrays];minus=[v.copy() for v in arrays]
                plus[k][idx]+=delta;minus[k][idx]-=delta
                grad[idx]=(objective(plus,np)[0]-objective(minus,np)[0])/(2*delta)
            numerical.append(grad)
        if case<=4:
            for a,b in zip(observed[0],numerical,strict=True):np.testing.assert_allclose(a,b,atol=1e-6,rtol=1e-5)
        else:
            # Independent float64 closed form for the FP32 cases: do not compare
            # finite differences against rounded arithmetic derivatives.
            a,n,g,b=arrays;mu=a.mean(0);v=((a-mu)**2).mean(0);r=np.sqrt(v+eps)
            hh=h.copy();z=(n-mu)/r*g+b
            if case==6:
                sigmoid=1/(1+np.exp(-z));hh*=sigmoid*(1+z*(1-sigmoid))
            gm=-g/r*hh.sum(0);gv=-g/(2*r**3)*(hh*(n-mu)).sum(0)
            analytical=[.14*(a-(1024. if case==5 else 0.))+gm/na+2*(a-mu)*gv/na,
                        hh*g/r,(hh*(n-mu)/r).sum(0),hh.sum(0)]
            for x,y in zip(analytical,numerical,strict=True):np.testing.assert_allclose(x,y,atol=1e-6,rtol=1e-5)
        reports.append({'case':case,'autograd_calls':2,'dtype':str(dtype),
                        'bridge_reference_max_abs':max(float(np.max(np.abs(a-b))) for a,b in zip(*observed,strict=True)),
                        'finite_difference_max_abs':max(float(np.max(np.abs(a-b))) for a,b in zip(observed[0],numerical,strict=True)),
                        'finite_difference_asserted':True,'forward_equal':True})
    write(root/'cpu_tests.json',{'passed':True,'autograd_calls':12,'fixtures':reports,
                               'FD_scope':'float64 reference all6; FP32 cases5-6 additionally compared to closed-form NumPy derivatives; no model forwards'})
    return reports
