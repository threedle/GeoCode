from dataset_generator.shape_validators.shape_validator_interface import ShapeValidatorInterface
from common.intersection_util import find_self_intersections


class CeilingLampValidator(ShapeValidatorInterface):
    def validate_shape(self, input_params_map) -> (bool, str):
        frames_self_intersections = find_self_intersections('frames')
        if frames_self_intersections > 0:
            return False, "Self intersection between the frames"
        light_bulbs_self_intersections = find_self_intersections('light_bulbs')
        if light_bulbs_self_intersections > 0:
            return False, "Self intersection between the light bulbs"
        return True, "Valid"
