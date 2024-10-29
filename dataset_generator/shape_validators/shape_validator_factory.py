from common.domain import Domain
from dataset_generator.shape_validators.shape_validator_interface import ShapeValidatorInterface
from dataset_generator.shape_validators.chair_validator import ChairValidator
from dataset_generator.shape_validators.vase_validator import VaseValidator
from dataset_generator.shape_validators.table_validator import TableValidator
from dataset_generator.shape_validators.ceiling_lamp_validator import CeilingLampValidator


class ShapeValidatorFactory:
    @staticmethod
    def create_validator(domain) -> ShapeValidatorInterface:
        if domain == Domain.chair:
            return ChairValidator()
        elif domain == Domain.vase:
            return VaseValidator()
        elif domain == Domain.table:
            return TableValidator()
        elif domain == Domain.ceiling_lamp:
            return CeilingLampValidator()
        else:
            raise Exception(f"Domain [{domain}] is not recognized.")
