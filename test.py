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

# Requires this command to be run from the executorch repo before this script can be run:
# ./examples/arm/setup.sh --i-agree-to-the-contained-eula  --target-toolchain zephyr

speech_model = load_seq2seq_speech_model("openai/whisper-tiny")
exports = speech_model.export()

spec_builder = ArmCompileSpecBuilder().ethosu_compile_spec(
        "ethos-u55-128",
        extra_flags="--verbose-operators --verbose-cycle-estimate"
        )
compile_spec = spec_builder.build()
inputs = (torch.rand(1,80,3000),)
exported_encoder = export(speech_model.encoder, inputs)
executorch_program = to_edge_transform_and_lower(
    exported_encoder,
    partitioner = [EthosUPartitioner(compile_spec)]
).to_executorch()

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
