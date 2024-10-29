from dataset_generator.shape_validators.shape_validator_interface import ShapeValidatorInterface


class CeilingLampValidator(ShapeValidatorInterface):
    def validate_shape(self, input_params_map) -> (bool, str):
        return True, "Valid"
