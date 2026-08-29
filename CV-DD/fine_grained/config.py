from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class FineGrainedDataset:
    name: str
    classes: int
    mean: tuple[float, float, float]
    std: tuple[float, float, float]
    recovery_iterations: int
    fkd_batch_size: int
    accumulation_steps: int
    teacher_reference_top1: float
    paper_targets: tuple[float, float, float]

    @property
    def student_forward_batch_size(self) -> int:
        return self.fkd_batch_size // self.accumulation_steps

    def to_dict(self) -> dict:
        payload = asdict(self)
        payload["student_forward_batch_size"] = self.student_forward_batch_size
        return payload


DATASETS = {
    "CUB_imsize224": FineGrainedDataset(
        name="CUB_imsize224",
        classes=200,
        mean=(0.4857, 0.4994, 0.4326),
        std=(0.2260, 0.2215, 0.2595),
        recovery_iterations=10_000,
        fkd_batch_size=20,
        accumulation_steps=2,
        teacher_reference_top1=71.6,
        paper_targets=(53.4, 60.0, 63.5),
    ),
    "A_imsize224": FineGrainedDataset(
        name="A_imsize224",
        classes=100,
        mean=(0.4865, 0.5177, 0.5425),
        std=(0.2124, 0.2051, 0.2375),
        recovery_iterations=4_000,
        fkd_batch_size=20,
        accumulation_steps=2,
        teacher_reference_top1=83.9,
        paper_targets=(52.6, 66.6, 68.3),
    ),
    "SC_imsize224": FineGrainedDataset(
        name="SC_imsize224",
        classes=196,
        mean=(0.4708, 0.4601, 0.4551),
        std=(0.2885, 0.2879, 0.2962),
        recovery_iterations=4_000,
        fkd_batch_size=14,
        accumulation_steps=2,
        teacher_reference_top1=85.2,
        paper_targets=(52.4, 68.2, 70.9),
    ),
}


ALIASES = {
    "cub": "CUB_imsize224",
    "aircraft": "A_imsize224",
    "a": "A_imsize224",
    "cars": "SC_imsize224",
    "sc": "SC_imsize224",
}


def get_dataset(name: str) -> FineGrainedDataset:
    canonical = ALIASES.get(name.lower(), name)
    try:
        return DATASETS[canonical]
    except KeyError as error:
        raise ValueError(
            f"Unsupported fine-grained dataset {name!r}; expected one of "
            f"{sorted(DATASETS)}"
        ) from error
