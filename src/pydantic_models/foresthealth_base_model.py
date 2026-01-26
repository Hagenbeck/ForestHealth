from pydantic import BaseModel, ConfigDict


class ForestHealthBaseModel(BaseModel):
    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        validate_assignment=True,
        extra="forbid",
        str_strip_whitespace=True,
    )
