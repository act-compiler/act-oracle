import jax.numpy as jnp


def attention(kernel, api):
    @kernel(
        hbm=32768,  # 32 KB: enough for 3 inputs + 1 output
        input=[
            {'addr': 0, 'shape': (64, 64), 'dtype': jnp.bfloat16},
            {'addr': 8192, 'shape': (64, 64), 'dtype': jnp.bfloat16},
            {'addr': 16384, 'shape': (64, 64), 'dtype': jnp.bfloat16},
        ],
        constant=[],
        output=[{'addr': 24576, 'shape': (64, 64), 'dtype': jnp.bfloat16}],
    )
    def attention_():
        # S = Q @ K^T, by gemm_33
        api.load_01(n=64, addr_in=8192, addr_out=0)  # K
        api.transpose_13(addr_in=0, addr_out=64)  # K^T
        api.load_03(n=64, addr_in=0, addr_out=0)

        api.gemm_33(addr_1=0, addr_2=64, addr_out=0)
        # P = softmax(S)
        api.softmax(n=64, addr=0)  # In place softmax
        # P and V where the second product reads them
        api.mov_23(n=64, addr_in=0, addr_out=0)  # P is now in d3 addr=0
        api.load_03(n=64, addr_in=16384, addr_out=64)  # V is now in d3 addr=64
        # O = P @ V, by gemm_33
        api.gemm_33(addr_1=0, addr_2=64, addr_out=0)
        api.mov_21(n=64, addr_in=0, addr_out=0)
        # O to HBM
        api.store_10(n=64, addr_in=0, addr_out=24576)

    return attention_
