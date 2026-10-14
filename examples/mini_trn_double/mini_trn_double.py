"""mini_trn_double Accelerator ISA Definition"""

from taidl import Accelerator

from taidl_to import generate_oracle

mini_trn_double = Accelerator('mini_trn_double')


# Data models

# d1: 128 rows x 64 columns of bf16
mini_trn_double.add_data_model('d1', [128], [64], 'bf16')

# d3: 128 rows x 64 columns of bf16
mini_trn_double.add_data_model('d3', [128], [64], 'bf16')

# d2: 64 rows x 64 columns of bf16
mini_trn_double.add_data_model('d2', [64], [64], 'bf16')


# Instructions

# load_01: load d1 from HBM (d0), row-major
instr = mini_trn_double.add_instruction('load_01', ['n'], ['addr_in', 'addr_out'])
instr.set_inputs([['d0', ['@a.addr_in'], ['@c.n * 128']]])  # u8[@c.n * 128]
instr.set_outputs([['d1', ['@a.addr_out'], ['@c.n']]])  # bf16[@c.n, 64]
instr.add_semantics("""
ENTRY {
    %In0 = u8[`@c.n * 128`] parameter(0);
    %a = u8[`@c.n`,64,2] reshape(%In0);
    ROOT %Out0 = bf16[`@c.n`,64] bitcast_convert(%a);
}
""")


# load_03: load d3 from HBM (d0), row-major
instr = mini_trn_double.add_instruction('load_03', ['n'], ['addr_in', 'addr_out'])
instr.set_inputs([['d0', ['@a.addr_in'], ['@c.n * 128']]])  # u8[@c.n * 128]
instr.set_outputs([['d3', ['@a.addr_out'], ['@c.n']]])  # bf16[@c.n, 64]
instr.add_semantics("""
ENTRY {
    %In0 = u8[`@c.n * 128`] parameter(0);
    %a = u8[`@c.n`,64,2] reshape(%In0);
    ROOT %Out0 = bf16[`@c.n`,64] bitcast_convert(%a);
}
""")


# store_10: store d1 to HBM (d0), row-major
instr = mini_trn_double.add_instruction('store_10', ['n'], ['addr_in', 'addr_out'])
instr.set_inputs([['d1', ['@a.addr_in'], ['@c.n']]])  # bf16[@c.n, 64]
instr.set_outputs([['d0', ['@a.addr_out'], ['@c.n * 128']]])  # u8[@c.n * 128]
instr.add_semantics("""
ENTRY {
    %In0 = bf16[`@c.n`,64] parameter(0);
    %a = u8[`@c.n`,64,2] bitcast_convert(%In0);
    ROOT %Out0 = u8[`@c.n*128`] reshape(%a);
}
""")

# store_30: store d3 to HBM (d0), row-major
instr = mini_trn_double.add_instruction('store_30', ['n'], ['addr_in', 'addr_out'])
instr.set_inputs([['d3', ['@a.addr_in'], ['@c.n']]])  # bf16[@c.n, 64]
instr.set_outputs([['d0', ['@a.addr_out'], ['@c.n * 128']]])  # u8[@c.n * 128]
instr.add_semantics("""
ENTRY {
    %In0 = bf16[`@c.n`,64] parameter(0);
    %a = u8[`@c.n`,64,2] bitcast_convert(%In0);
    ROOT %Out0 = u8[`@c.n*128`] reshape(%a);
}
""")

# transpose_13: move 64 rows of d1 to d3, transposed
instr = mini_trn_double.add_instruction('transpose_13', [], ['addr_in', 'addr_out'])
instr.set_inputs([['d1', ['@a.addr_in'], ['64']]])  # bf16[64, 64]
instr.set_outputs([['d3', ['@a.addr_out'], ['64']]])  # bf16[64, 64]
instr.add_semantics("""
ENTRY {
    %In0 = bf16[64,64] parameter(0);
    %a = bf16[64,64] transpose(%In0), dimensions={1,0};
    ROOT %Out0 = bf16[64, 64] copy(%a);
}
""")

# mov_21: copy d2 to d1
instr = mini_trn_double.add_instruction('mov_21', ['n'], ['addr_in', 'addr_out'])
instr.set_inputs([['d2', ['@a.addr_in'], ['@c.n']]])  # bf16[@c.n, 64]
instr.set_outputs([['d1', ['@a.addr_out'], ['@c.n']]])  # bf16[@c.n, 64]
instr.add_semantics("""
ENTRY {
    %In0 = bf16[`@c.n`,64] parameter(0);
    ROOT %Out0 = bf16[`@c.n`,64] copy(%In0);
}
""")

# mov_23: copy d2 to d3
instr = mini_trn_double.add_instruction('mov_23', ['n'], ['addr_in', 'addr_out'])
instr.set_inputs([['d2', ['@a.addr_in'], ['@c.n']]])  # bf16[@c.n, 64]
instr.set_outputs([['d3', ['@a.addr_out'], ['@c.n']]])  # bf16[@c.n, 64]
instr.add_semantics("""
ENTRY {
    %In0 = bf16[`@c.n`,64] parameter(0);
    ROOT %Out0 = bf16[`@c.n`,64] copy(%In0);
}
""")

# gemm_33: multiply two d3 tiles into d2
instr = mini_trn_double.add_instruction('gemm_33', [], ['addr_1', 'addr_2', 'addr_out'])
instr.set_inputs([['d3', ['@a.addr_1'], ['64']], ['d3', ['@a.addr_2'], ['64']]])
instr.set_outputs([['d2', ['@a.addr_out'], ['64']]])  # bf16[64, 64]
instr.add_semantics("""
ENTRY {
    %In0 = bf16[64,64] parameter(0);
    %In1 = bf16[64,64] parameter(1);
    ROOT %Out0 = bf16[64,64] dot(%In0, %In1), lhs_contracting_dims={1}, rhs_contracting_dims={0};
}
""")

# gemm_13: multiply d1 by d3 into d2
instr = mini_trn_double.add_instruction('gemm_13', [], ['addr_1', 'addr_2', 'addr_out'])
instr.set_inputs([['d1', ['@a.addr_1'], ['64']], ['d3', ['@a.addr_2'], ['64']]])
instr.set_outputs([['d2', ['@a.addr_out'], ['64']]])  # bf16[64, 64]
instr.add_semantics("""
ENTRY {
    %In0 = bf16[64,64] parameter(0);
    %In1 = bf16[64,64] parameter(1);
    ROOT %Out0 = bf16[64,64] dot(%In0, %In1), lhs_contracting_dims={1}, rhs_contracting_dims={0};
}
""")

# softmax: softmax of each d2 row, in place
instr = mini_trn_double.add_instruction('softmax', ['n'], ['addr'])
instr.set_inputs([['d2', ['@a.addr'], ['@c.n']]])  # bf16[@c.n, 64]
instr.set_outputs([['d2', ['@a.addr'], ['@c.n']]])  # bf16[@c.n, 64]
instr.add_semantics("""
ENTRY {
    %In0 = bf16[`@c.n`,64] parameter(0);
    %a = bf16[`@c.n`,64] exponential(%In0);
    %reduced = bf16[`@c.n`] reduce_add(%a), dimensions={1};
    %b = bf16[`@c.n`,64] broadcast(%reduced), dimensions={0};
    ROOT %Out0 = bf16[`@c.n`,64] divide(%a, %b);
}
""")

generate_oracle(mini_trn_double)
