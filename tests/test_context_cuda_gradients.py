"""Regression: eval-mode CUDA RNNs need a gradient-capable recurrent forward for explanations."""
import pytest
import torch
from virtual_imu.context_study.models import ContextModel
from virtual_imu.context_study.gradients import eval_forward_for_input_gradients

@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA required')
@pytest.mark.parametrize('backend,lookahead',[('lstm',0),('gru',0),('bigru',0),('bigru',6),('bigru','full')])
def test_eval_cuda_gradient_preserves_predictions_mode_and_future_bound(backend,lookahead):
    torch.set_num_threads(1);torch.manual_seed(91)
    model=ContextModel(backend=backend,lookahead=lookahead).cuda().eval()
    inputs=torch.randn(2,6,32,device='cuda',requires_grad=True)
    before={name:m.training for name,m in model.named_modules()}
    with torch.no_grad():reference=model(inputs)
    explained=eval_forward_for_input_gradients(model,inputs)
    torch.testing.assert_close(explained,reference,atol=2e-6,rtol=2e-5)
    grad=torch.autograd.grad(explained[:,:,16].sum(),inputs)[0]
    assert torch.isfinite(grad).all() and grad.abs().sum()>0
    if isinstance(lookahead,int):assert grad[:,:,17+lookahead:].abs().max()<1e-8
    assert before=={name:m.training for name,m in model.named_modules()}
