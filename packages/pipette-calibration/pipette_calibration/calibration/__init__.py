from pipette_scores.types import EvalId

GROUP_OF = {
    EvalId.IFBENCH: lambda s: str(s.n_constraints),
    EvalId.IFSTRUCT: lambda s: s.entity_type,
    EvalId.GPQA_DIAMOND: lambda s: s.answer,
    EvalId.MATH_500: lambda _s: "math_500",
}

SPLITS = {
    EvalId.IFBENCH: ("train",),
    EvalId.IFSTRUCT: ("test",),
    EvalId.GPQA_DIAMOND: ("test",),
    EvalId.MATH_500: ("test",),
}

PRIMARY_SPLIT = {
    EvalId.IFBENCH: "train",
    EvalId.IFSTRUCT: "test",
    EvalId.GPQA_DIAMOND: "test",
    EvalId.MATH_500: "test",
}
