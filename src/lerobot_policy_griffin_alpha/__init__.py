"""Griffin Alpha-S: a LeRobot policy plugin for the Qwen3-VL-4B vision-language-action model.

Two policy types, registered under distinct type strings and selected by a checkpoint's
``config.json`` ``type`` field:

- ``griffin_alpha``       -> flow-matching action expert (``GriffinAlphaPolicy``), the default head
- ``griffin_alpha_fast``  -> FAST action tokens (``GriffinAlphaFASTPolicy``)

Importing this package registers both types (and the processor steps their checkpoints reference).
LeRobot imports it automatically: its CLIs import every installed distribution whose name starts with
``lerobot_policy_``, which is why the distribution name must equal this package's import name.
"""

from .action_steps import (
    AbsoluteActionWithSE3ProcessorStep,
    GriffinAlphaAddBatchDimensionProcessorStep,
    RelativeActionWithSE3ProcessorStep,
    ResampleActionProcessorStep,
    SE3MatrixToXYZRot6DProcessorStep,
    XYZRot6DToSE3MatrixProcessorStep,
    reconnect_se3_steps,
)
from .backbone import GriffinAlphaBackboneConfig, GriffinAlphaBackbonePolicy
from .configuration_griffin_alpha import GriffinAlphaConfig
from .configuration_griffin_alpha_fast import GriffinAlphaFASTConfig
from .modeling_griffin_alpha import FlowMatchingExpert, GriffinAlphaPolicy
from .modeling_griffin_alpha_fast import GriffinAlphaFASTPolicy
from .processor_griffin_alpha import GriffinAlphaInputProcessorStep, make_griffin_alpha_pre_post_processors
from .processor_griffin_alpha_fast import (
    GriffinAlphaFASTInputProcessorStep,
    make_griffin_alpha_fast_pre_post_processors,
)

__all__ = [
    "GriffinAlphaBackboneConfig",
    "GriffinAlphaBackbonePolicy",
    "GriffinAlphaConfig",
    "GriffinAlphaPolicy",
    "FlowMatchingExpert",
    "GriffinAlphaInputProcessorStep",
    "make_griffin_alpha_pre_post_processors",
    "GriffinAlphaFASTConfig",
    "GriffinAlphaFASTPolicy",
    "GriffinAlphaFASTInputProcessorStep",
    "make_griffin_alpha_fast_pre_post_processors",
    "GriffinAlphaAddBatchDimensionProcessorStep",
    "ResampleActionProcessorStep",
    "RelativeActionWithSE3ProcessorStep",
    "AbsoluteActionWithSE3ProcessorStep",
    "SE3MatrixToXYZRot6DProcessorStep",
    "XYZRot6DToSE3MatrixProcessorStep",
    "reconnect_se3_steps",
]
