# Operations

The oracle lowers every operation word of TAIDL v2.13 to the StableHLO operation that means the same, or to a CHLO operation where StableHLO has none.
What each word takes, its operands and its attributes, is the language's and is documented in [TAIDL's operations reference](https://github.com/act-compiler/taidl/blob/master/docs/operations.md).
This page says what the oracle adds: the operation each word lowers to, the element types at which it lowers, what it refuses by name when an oracle is generated, and how closely the tests hold each word to a reference.

## Element types

The oracle lowers every element type the grammar lexes, 19 in all: the floats `f64`, `f32`, `f16`, `bf16`, `f8e4m3fn` and `f8e5m2`, the integers `u2`, `u4`, `u8`, `u16`, `u32`, `u64`, `s2`, `s4`, `s8`, `s16`, `s32` and `s64`, and `pred`.
At `s2` and `u2` it refuses `dot` and `concatenate` by name.
A word's type is the element type of its first operand, or of its result where it has no operand.
A `pred` moves to and from its byte by a `convert`, in a semantics and at a kernel's inputs and outputs.
A kernel declares a value of `s2`, `u2`, `s4`, `u4`, `f8e4m3fn` or `f8e5m2` by its jax dtype, `jnp.int2`, `jnp.uint2`, `jnp.int4`, `jnp.uint4`, `jnp.float8_e4m3fn` or `jnp.float8_e5m2`.
In HBM a value of `s2` or `u2` packs four elements to a byte and one of `s4` or `u4` two, by a `bitcast_convert`, as a semantics packs it; a kernel's input, constant or output of a sub-byte type whose elements fill no whole number of bytes is refused by name when the kernel is compiled.

## The words

In the table, every type is the 19 above, signed is `s2` to `s64`, and a word at a type the table does not list for it is refused by name.

| word                     | lowers to                                          | element types       |
| ------------------------ | -------------------------------------------------- | ------------------- |
| `parameter`              | a `dynamic_slice` of the input's region            | every type          |
| `constant`               | `stablehlo.constant`, its literal at every element | every type          |
| `iota`                   | `stablehlo.iota`                                   | floats and integers |
| `abs`                    | `stablehlo.abs`                                    | floats and signed   |
| `negate`                 | `stablehlo.negate`                                 | floats and integers |
| `sign`                   | `stablehlo.sign`                                   | floats and signed   |
| `add`                    | `stablehlo.add`                                    | every type          |
| `subtract`               | `stablehlo.subtract`                               | floats and integers |
| `multiply`               | `stablehlo.multiply`                               | every type          |
| `mulhi`                  | `chlo.mulhi`                                       | integers            |
| `divide`                 | `stablehlo.divide`                                 | floats and integers |
| `remainder`              | `stablehlo.remainder`                              | floats and integers |
| `maximum`                | `stablehlo.maximum`                                | every type          |
| `minimum`                | `stablehlo.minimum`                                | every type          |
| `power`                  | `stablehlo.power`                                  | floats and integers |
| `sqrt`                   | `stablehlo.sqrt`                                   | floats              |
| `rsqrt`                  | `stablehlo.rsqrt`                                  | floats              |
| `cbrt`                   | `stablehlo.cbrt`                                   | floats              |
| `exponential`            | `stablehlo.exponential`                            | floats              |
| `exponential_minus_one`  | `stablehlo.exponential_minus_one`                  | floats              |
| `log`                    | `stablehlo.log`                                    | floats              |
| `log_plus_one`           | `stablehlo.log_plus_one`                           | floats              |
| `logistic`               | `stablehlo.logistic`                               | floats              |
| `erf`                    | `chlo.erf`                                         | floats              |
| `sine`                   | `stablehlo.sine`                                   | floats              |
| `cosine`                 | `stablehlo.cosine`                                 | floats              |
| `tan`                    | `stablehlo.tan`                                    | floats              |
| `asin`                   | `chlo.asin`                                        | floats              |
| `acos`                   | `chlo.acos`                                        | floats              |
| `atan2`                  | `stablehlo.atan2`                                  | floats              |
| `sinh`                   | `chlo.sinh`                                        | floats              |
| `cosh`                   | `chlo.cosh`                                        | floats              |
| `tanh`                   | `stablehlo.tanh`                                   | floats              |
| `asinh`                  | `chlo.asinh`                                       | floats              |
| `acosh`                  | `chlo.acosh`                                       | floats              |
| `atanh`                  | `chlo.atanh`                                       | floats              |
| `floor`                  | `stablehlo.floor`                                  | floats              |
| `ceil`                   | `stablehlo.ceil`                                   | floats              |
| `round_nearest_afz`      | `stablehlo.round_nearest_afz`                      | floats              |
| `round_nearest_even`     | `stablehlo.round_nearest_even`                     | floats              |
| `and`                    | `stablehlo.and`                                    | integers and `pred` |
| `or`                     | `stablehlo.or`                                     | integers and `pred` |
| `xor`                    | `stablehlo.xor`                                    | integers and `pred` |
| `not`                    | `stablehlo.not`                                    | integers and `pred` |
| `shift_left`             | `stablehlo.shift_left`                             | integers            |
| `shift_right_logical`    | `stablehlo.shift_right_logical`                    | integers            |
| `shift_right_arithmetic` | `stablehlo.shift_right_arithmetic`                 | integers            |
| `popcnt`                 | `stablehlo.popcnt`                                 | integers            |
| `count_leading_zeros`    | `stablehlo.count_leading_zeros`                    | integers            |
| `compare`                | `stablehlo.compare`                                | every type          |
| `select`                 | `stablehlo.select`                                 | every type          |
| `clamp`                  | `stablehlo.clamp`                                  | every type          |
| `is_finite`              | `stablehlo.is_finite`                              | floats              |
| `convert`                | `stablehlo.convert`                                | every type          |
| `bitcast_convert`        | `stablehlo.bitcast_convert`                        | every type          |
| `reduce_precision`       | `stablehlo.reduce_precision`                       | floats              |
| `reshape`                | `stablehlo.reshape`                                | every type          |
| `transpose`              | `stablehlo.transpose`                              | every type          |
| `broadcast`              | `stablehlo.broadcast_in_dim`                       | every type          |
| `concatenate`            | `stablehlo.concatenate`                            | all but `s2`, `u2`  |
| `pad`                    | `stablehlo.pad`                                    | every type          |
| `reverse`                | `stablehlo.reverse`                                | every type          |
| `slice`                  | `stablehlo.slice`                                  | every type          |
| `dynamic_slice`          | `stablehlo.dynamic_slice`                          | every type          |
| `dynamic_update_slice`   | `stablehlo.dynamic_update_slice`                   | every type          |
| `gather`                 | `stablehlo.gather`                                 | every type          |
| `copy`                   | `stablehlo.reshape` to its own type                | every type          |
| `dot`                    | `stablehlo.dot_general`                            | all but `s2`, `u2`  |
| `cholesky`               | `stablehlo.cholesky`                               | floats              |
| `triangular_solve`       | `stablehlo.triangular_solve`                       | floats              |
| `batch_norm_inference`   | `stablehlo.batch_norm_inference`                   | floats              |
| `reduce_add`             | `stablehlo.reduce` applying `add`                  | every type          |
| `reduce_multiply`        | `stablehlo.reduce` applying `multiply`             | every type          |
| `reduce_max`             | `stablehlo.reduce` applying `maximum`              | every type          |
| `reduce_min`             | `stablehlo.reduce` applying `minimum`              | every type          |
| `reduce_and`             | `stablehlo.reduce` applying `and`                  | integers and `pred` |
| `reduce_or`              | `stablehlo.reduce` applying `or`                   | integers and `pred` |
| `reduce_xor`             | `stablehlo.reduce` applying `xor`                  | integers and `pred` |

A word means what XLA-HLO says it means, in particular:

- A `constant` writes a non-finite float by its bits, and its literal rounds to the nearest value of its type.
- A reduce starts from its reducer's identity at its type, a float sum from `+0.0` as XLA's does, so a reduce over an empty dimension gives that identity.
- `power` at an integer type is XLA's, which reads an exponent's low six bits, so an exponent past 63 does not give integer arithmetic's result.
- `select` takes a predicate of its branches' shape or of rank 0, as StableHLO allows.
- `cholesky` leaves the triangle it does not compute as the implementation writes it.
- `cholesky` and `triangular_solve` read a flag left out as `false`, as XLA-HLO does.
- `triangular_solve` at `f16` solves at `f32` and rounds its result to `f16`, since XLA's GPU backend fails a solve at `f16`.
- A list or a slice range may hold an expression in backticks, evaluated at each call; an attribute that is not a list, such as `iota_dimension` or `epsilon`, is a literal.

## What is refused

Generation refuses, by name and before it writes anything, a spec the oracle cannot lower or whose semantics contradicts itself, and leaves no oracle, not even the previous one.
A `NotImplementedError` names what the oracle does not lower, a `ValueError` a contradiction in the spec, and a `TypeError` a contradiction in a value's type.

| refused                                                                                                         | error                 |
| --------------------------------------------------------------------------------------------------------------- | --------------------- |
| an element type of the six the oracle does not lower                                                            | `NotImplementedError` |
| a word at an element type the table above does not list for it                                                  | `NotImplementedError` |
| an index of `dynamic_slice`, `dynamic_update_slice` or `gather` that is not an integer                          | `NotImplementedError` |
| an expression in an attribute that is not a list, such as `iota_dimension` or `epsilon`                         | `NotImplementedError` |
| a state update                                                                                                  | `NotImplementedError` |
| a name the generated oracle keeps for itself, as an instruction, attribute, data model or value                 | `NotImplementedError` |
| an output no semantics gives, or a value with no region of its kind                                             | `ValueError`          |
| a tile typed unlike its data model, or holding more elements than its region                                    | `ValueError`          |
| a line whose shapes, element types or attributes disagree, where the spec writes them as integers               | `ValueError`          |
| a `compare` whose `type` does not fit its operands                                                              | `ValueError`          |
| a `constant` literal past its type's range                                                                      | `ValueError`          |
| a `batch_norm_inference` epsilon past the range of `f32`, the type StableHLO gives it                           | `ValueError`          |
| a `constant` number that is not an integer at an integer type, or `true` or `false` at a type other than `pred` | `TypeError`           |

A tile is compared with its region at each call, at each iteration of its loops: one of the region's element count in another shape is read from or written to the region in row-major order where no loop sizes either, and any other tile whose shape differs is refused with a `ValueError`.
A shape or a list entry written as an expression is known only at the call, so a disagreement it hides fails when the kernel is compiled.
A `constant` given by an expression is checked against its type at each call.

## How the tests hold each word

Each word but `parameter` runs at each type the table lists for it, `s2`, `u2`, `s4`, `u4`, `f8e4m3fn` and `f8e5m2` aside, in a kernel whose output is compared with a NumPy reference computed in the test; `parameter` reads every data model the tests declare.
At those six types the tests compare `add` and `reduce_max` exactly with jax, hold a `constant` at the two 8-bit floats to its value, and pass a kernel's input, constant and output through HBM and a data model unchanged.
The transcendental words, `exponential`, `rsqrt`, `cbrt`, `exponential_minus_one`, `log`, `log_plus_one`, `logistic`, `erf`, `sine`, `cosine`, `tan`, `asin`, `acos`, `atan2`, `sinh`, `cosh`, `tanh`, `asinh`, `acosh`, `atanh`, and `power` at a float type, are held within a number of ulps of the reference at the result's type, an infinity or a nan held exactly.
An ulp is the unit in the last place of the result's type at the reference's magnitude.
On a GPU the bounds double.

| type   | CPU    | GPU     |
| ------ | ------ | ------- |
| `bf16` | 2 ulps | 4 ulps  |
| `f16`  | 2 ulps | 4 ulps  |
| `f32`  | 8 ulps | 16 ulps |
| `f64`  | 8 ulps | 16 ulps |

A word that reaches past its bound, on XLA's CPU or on an NVIDIA H100 NVL, is held on that platform alone to twice the most it was measured to reach:

| word                    | type  | platform | measured  | bound    |
| ----------------------- | ----- | -------- | --------- | -------- |
| `exponential_minus_one` | `f16` | CPU      | 2.05 ulps | 4.1 ulps |
| `sqrt`                  | `f32` | GPU      | 1 ulp     | 2 ulps   |
