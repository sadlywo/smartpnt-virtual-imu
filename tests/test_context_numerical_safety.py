import torch,pytest
from virtual_imu.models.matched import ProbSparseAttention
from virtual_imu.context_study.numerical_safety import safe_sparse_forward,_ORIGINAL

@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA required')
def test_conditional_fp32_recovery_and_finite_path_identity():
    torch.manual_seed(932);module=ProbSparseAttention(32,factor=2).cuda()
    q=torch.randn(2,2,32,8,device='cuda',requires_grad=True);k=torch.randn_like(q,requires_grad=True);v=torch.randn_like(q,requires_grad=True)
    with torch.autocast('cuda'):
        expected=_ORIGINAL(module,q.half(),k.half(),v.half());actual=safe_sparse_forward(module,q.half(),k.half(),v.half())
    assert torch.equal(expected,actual)
    largeq=torch.full((2,2,32,8),120.,device='cuda',requires_grad=True);largek=torch.full_like(largeq,120.,requires_grad=True)
    with torch.autocast('cuda'):
        unsafe=_ORIGINAL(module,largeq.half(),largek.half(),v.half())
        safe=safe_sparse_forward(module,largeq.half(),largek.half(),v.half())
    assert not torch.isfinite(unsafe).all()
    assert torch.isfinite(safe).all()
    safe.float().square().mean().backward()
    assert all(torch.isfinite(t.grad).all() for t in (largeq,largek,v))
