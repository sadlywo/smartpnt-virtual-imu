"""Independent explicit-time implementation for saved FIR/ARX states."""
import numpy as np


def reference_predict(state, x, *, center, block_size):
    channels = x.shape[-1]
    order = len(state['ar_weights'])
    result = []
    for anchor in np.asarray(x, dtype=np.float64):
        output = np.zeros((center+block_size, channels))
        first = 0 if order else center
        for t in range(first, center+block_size):
            positions = t+state['offsets']
            available = (positions >= 0) & (positions < len(anchor))
            features = np.zeros((len(positions), channels))
            features[available] = anchor[positions[available]]
            design = features.ravel()
            if state['delta']:
                previous = anchor[t-1] if t > 0 else np.zeros(channels)
                design = np.r_[design, anchor[t]-previous]
            output[t] = design@state['weights']+state['bias']
            for k, coefficients in enumerate(state['ar_weights'], 1):
                if t >= k:
                    output[t] += output[t-k]@coefficients
        result.append(output[center:center+block_size])
    return np.array(result)
