#import logging
#import os
#import subprocess
#import tempfile
#import unittest

#import pytest
#from datasets import load_dataset
from executorch import version
from executorch.backends.arm.ethosu_partitioner import EthosUPartitioner
from executorch.backends.arm.arm_backend import ArmCompileSpecBuilder
from executorch.exir.backend.backend_api import to_backend
from executorch.extension.export_util.utils import save_pte_program
#from executorch.extension.pybindings.portable_lib import ExecuTorchModule
#from packaging.version import parse
#from transformers import AutoProcessor, AutoTokenizer
#from transformers.testing_utils import slow

#from optimum.executorch import ExecuTorchModelForSpeechSeq2Seq
#from optimum.utils.import_utils import is_transformers_version
from optimum.exporters.executorch.tasks.asr import load_seq2seq_speech_model

speech_model = load_seq2seq_speech_model("openai/whisper-tiny")
exports = speech_model.export()

spec_builder = ArmCompileSpecBuilder().ethosu_compile_spec(
        "ethos-u55-128",
        extra_flags="--verbose-operators --verbose-cycle-estimate"
        )
compile_spec = spec_builder.build()
encoder_ethosU = to_backend(
    exports["encoder"],
    EthosUPartitioner(compile_spec)
    )

save_pte_program(encoder_ethosU, "whisper_tiny_encoder_ethos_u55_128.pte")
