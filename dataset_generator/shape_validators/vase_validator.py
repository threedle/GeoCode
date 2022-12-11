from dataset_generator.shape_validators.shape_validator_interface import ShapeValidatorInterface
from common.intersection_util import find_self_intersections, find_cross_intersections


class VaseValidator(ShapeValidatorInterface):
    def validate_shape(self, input_params_map) -> (bool, str):
        body_self_intersections = find_self_intersections('Body Self Intersections')
        if body_self_intersections > 0:
            return False, "Self intersection in the body"
        if input_params_map['handle_count'].get_value() > 0:
            handle_self_intersections = find_self_intersections('Handle Self Intersections')
            if handle_self_intersections > 0:
                return False, "self intersection in the handle"
            base_handle_intersections = find_self_intersections('Base and Handle Intersections')
            if base_handle_intersections > 0:
                return False, "Base intersects with handles"
            floor_handle_intersections = find_self_intersections('Floor and Handle Intersections')
            if floor_handle_intersections > 0:
                return False, "Floor intersects with handles"
        return True, "Valid"
