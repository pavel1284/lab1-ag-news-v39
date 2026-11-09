"""Validated configuration for the first lab."""
from pathlib import Path
from typing import Annotated

import yaml
from pydantic import BaseModel, ConfigDict, Field


class Lab1Config(BaseModel):
    model_config = ConfigDict(extra="forbid")

    variant: Annotated[int, Field(ge=1, le=45)] = 39
    seed: int = 39
    dataset_id: str = "fancyzhx/ag_news"
    expected_train: Annotated[int, Field(gt=0)] = 120000
    expected_test: Annotated[int, Field(gt=0)] = 7600
    input_dir: Path = Path("data/raw")
    report_dir: Path = Path("reports/LAB1")
    min_token_length: Annotated[int, Field(ge=1, le=15)] = 2
    max_top_terms: Annotated[int, Field(ge=3, le=30)] = 12
    plot_max_tokens: Annotated[int, Field(ge=50, le=2000)] = 250
    max_stage_bytes: Annotated[int, Field(ge=1000)] = 1048576
    use_mlflow: bool = True
    mlflow_experiment: str = "lab1_ag_news_v39"


def read_config(path: str | Path = "configs/lab1.yaml") -> Lab1Config:
    with Path(path).open(encoding="utf-8") as stream:
        return Lab1Config.model_validate(yaml.safe_load(stream))
