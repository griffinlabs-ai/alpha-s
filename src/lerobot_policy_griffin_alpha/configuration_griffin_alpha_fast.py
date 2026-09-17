"""Configuration for ``griffin_alpha_fast``: Griffin Alpha-S with the FAST action-token head.

The FAST head is the variant; the unqualified ``griffin_alpha`` type is the flow-matching head
(``configuration_griffin_alpha.py``). Both share every backbone/prompt/action-space field through
``GriffinAlphaBackboneConfig``.

Naming is load-bearing -- lerobot resolves the whole triple from this class
(``lerobot/policies/factory.py``): ``GriffinAlphaFASTConfig`` -> policy class
``GriffinAlphaFASTPolicy`` in ``modeling_griffin_alpha_fast`` (this module's name with
``configuration_`` -> ``modeling_``), and the registered type string ``griffin_alpha_fast`` ->
factory ``make_griffin_alpha_fast_pre_post_processors`` in ``processor_griffin_alpha_fast``.
Renaming any one of the three without the others breaks ``make_policy`` at runtime, not import time.
"""

from dataclasses import dataclass

from lerobot.configs.policies import PreTrainedConfig

from .backbone import GriffinAlphaBackboneConfig


@PreTrainedConfig.register_subclass("griffin_alpha_fast")
@dataclass
class GriffinAlphaFASTConfig(GriffinAlphaBackboneConfig):
    """Configuration class for GriffinAlphaFASTPolicy."""

    # The FAST tokenizer (DCT + BPE over the normalized action chunk) whose codes are appended to the
    # text vocabulary as ``<robot_action_i>``.
    fast_action_tokenizer_name: str = "lerobot/fast-action-tokenizer"
