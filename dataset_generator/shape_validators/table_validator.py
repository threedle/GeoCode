from dataset_generator.shape_validators.shape_validator_interface import ShapeValidatorInterface
from dataset_generator.shape_validators.common_validations import validate_monoleg
from common.intersection_util import find_self_intersections


class TableValidator(ShapeValidatorInterface):
    def validate_shape(self, input_params_map) -> (bool, str):
        table_top_and_legs_support_intersections = find_self_intersections('table_top_and_legs_support')
        if table_top_and_legs_support_intersections > 0:
            return False, "Table top intersects with the legs supports"
        floor_and_legs_support_intersections = find_self_intersections('floor_and_legs_support')
        if floor_and_legs_support_intersections > 0:
            return False, "Legs supports intersect with the floor"
        if input_params_map['is_monoleg'].get_value() > 0 and input_params_map['is_monoleg_tent'].get_value() == 0:
            if not validate_monoleg('monoleg', factor=0.16):
                # the factor is more restricting since the tables can be much wider than chairs
                return False, "Invalid monoleg"
        return True, "Valid"
