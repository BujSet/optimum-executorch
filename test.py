import logging
import os
import subprocess
import tempfile
import unittest

import pytest
from datasets import load_dataset
from executorch import version
from executorch.extension.pybindings.portable_lib import ExecuTorchModule
from packaging.version import parse
from transformers import AutoProcessor, AutoTokenizer
from transformers.testing_utils import slow

from optimum.executorch import ExecuTorchModelForSpeechSeq2Seq
from optimum.utils.import_utils import is_transformers_version
from optimum.exporters.executorch.tasks.seq2seq_lm import load_seq2seq_lm_model

model = load_seq2seq_lm_model("openai/whisper-tiny")
