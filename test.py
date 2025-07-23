from executorch import version
from executorch.backends.arm.ethosu_partitioner import EthosUPartitioner
from executorch.backends.arm.arm_backend import ArmCompileSpecBuilder
from executorch.exir.backend.backend_api import to_backend
from executorch.extension.export_util.utils import save_pte_program
from optimum.exporters.executorch.tasks.asr import load_seq2seq_speech_model
from executorch.exir import (
    EdgeCompileConfig,
    ExecutorchBackendConfig,
    to_edge_transform_and_lower,
)
import torch
from torch.export import Dim, export
from torchao.quantization.pt2e.quantize_pt2e import convert_pt2e, prepare_pt2e
from executorch.backends.arm.quantizer.arm_quantizer import (
    EthosUQuantizer,
    get_symmetric_quantization_config,
)
import executorch.kernels.quantized

# Requires this command to be run from the executorch repo before this script can be run:
# ./examples/arm/setup.sh --i-agree-to-the-contained-eula  --target-toolchain zephyr

speech_model = load_seq2seq_speech_model("openai/whisper-tiny")
exports = speech_model.export()

spec_builder = ArmCompileSpecBuilder().ethosu_compile_spec(
        "ethos-u55-128",
        system_config="Ethos_U55_High_End_Embedded",
        memory_mode="Shared_Sram",
        extra_flags="--output-format=raw --debug-force-regor",
        )

compile_spec = spec_builder.build()
#torch.ops.load_library(
#    "cmake-out-aot-lib/kernels/quantized/libquantized_ops_aot_lib.so"
#)
inputs = (torch.rand(1,80,3000),)
#exported_encoder = export(speech_model.encoder, inputs)
# Post training quantization
graph_module = torch.export.export_for_training(speech_model.encoder, inputs).module()
quantizer = EthosUQuantizer(compile_spec)
operator_config = get_symmetric_quantization_config(is_per_channel=False)
quantizer.set_global(operator_config)
graph_module = prepare_pt2e(graph_module, quantizer)
graph_module(*inputs)
graph_module = convert_pt2e(graph_module)
exported_program = torch.export.export_for_training(graph_module, inputs)
executorch_program = to_edge_transform_and_lower(
    exported_program,
    partitioner = [EthosUPartitioner(compile_spec)],
    compile_config=EdgeCompileConfig(
        _check_ir_validity=False,
    )
).to_executorch(config=ExecutorchBackendConfig(extract_delegate_segments=False))

with open("whisper_tiny_encoder_ethos_u55_128.pte", "wb") as file:
    file.write(executorch_program.buffer)

#encoder_ethosU = to_backend(
#    exports["encoder"],
#    EthosUPartitioner(compile_spec)
#    )
#exec_prog = encoder_ethosU.to_executorch(
#            config=ExecutorchBackendConfig(extract_delegate_segments=False)
#)
#save_pte_program(exec_prog, "whisper_tiny_encoder_ethos_u55_128.pte")

#edge = to_edge_transform_and_lower(
#    exports["encoder"],
#    partitioner=[EthosUPartitioner(compile_spec)],
#    compile_config=EdgeCompileConfig(
#            _check_ir_validity=False,
#    ),
#)
#exec_prog = edge.to_executorch(
#            config=ExecutorchBackendConfig(extract_delegate_segments=False)
#)
#save_pte_program(exec_prog, "whisper_tiny_encoder_ethos_u55_128.pte")
