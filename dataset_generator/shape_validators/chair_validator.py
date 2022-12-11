from dataset_generator.shape_validators.shape_validator_interface import ShapeValidatorInterface
from dataset_generator.shape_validators.common_validations import validate_monoleg
from common.intersection_util import find_self_intersections, find_cross_intersections


class ChairValidator(ShapeValidatorInterface):
    def validate_shape(self, input_params_map) -> (bool, str):
        if not self.is_valid_chair(input_params_map):
            return False, "Collision"
        return True, "Valid"

    def is_valid_chair(self, input_params_map):
        if input_params_map['is_back_rest'].get_value() == 0 \
                and input_params_map['is_top_rail'].get_value() == 1 \
                and input_params_map['is_vertical_rail'].get_value() == 1:
            # reaching here means the vertical rails are visible
            # also note the assumption that the min vertical rails count is 3
            if find_self_intersections('vertical_rails_out') > 0:
                # try again, as we have vertical rails intersecting each other
                # print("vr, found collisions")
                return False
        if input_params_map['is_back_rest'].get_value() == 0 \
                and (input_params_map['is_top_rail'].get_value() == 0
                     or input_params_map['is_vertical_rail'].get_value() == 0):
            # reaching here means the cross rails are visible
            # also note the assumption that the min cross rails count is 3
            if find_self_intersections('cross_rails_and_top_rail_out') > 0:
                # try again, as we have cross rails intersecting each other or the top rail
                # print("cr, found collisions...")
                return False
        if input_params_map['handles_state'].get_value() == 1 and input_params_map['is_handles_support'].get_value():
            if find_self_intersections('handles_support_and_back_frame') > 0:
                return False
        if input_params_map['handles_state'].get_value() > 0:
            if find_cross_intersections('handles_left_side', 'handles_right_side') > 0:
                # the handles in both sides of the chair should never intersect
                return False
        if input_params_map['is_monoleg'].get_value() > 0 and input_params_map['is_monoleg_tent'].get_value() == 0:
            if not validate_monoleg('monoleg'):
                return False
        return True
