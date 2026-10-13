"""Optional EEF target filter for the fixed-left RTC client.

Like the Tianji client, filter before the existing execution constraints and
feed the resulting committed targets back as the next RTC prefix. Planning
uses explicit endpoint seeds, so there is no hidden state for discarded tails
or a reset at chunk boundaries. SDK-side filtering is independent.
"""
import math
import numpy as np

from .geometry import eef9_column_to_pose7, pose7_to_eef9_column, slerp
from .robot import Action58
from .rtc_fixed_left import FixedLeftController


class EEFFilteredFixedLeftController(FixedLeftController):
    def __init__(self, *args, eef_filter_alpha=1.0, **kwargs):
        alpha = float(eef_filter_alpha)
        if not math.isfinite(alpha) or not 0 < alpha <= 1:
            raise ValueError('EEF filter alpha must be finite and in (0, 1]')
        self.eef_filter_alpha = alpha
        super().__init__(*args, **kwargs)

    def prepare_from(self, actions, seed_poses, seed_hands):
        if self.eef_filter_alpha == 1.0:
            return super().prepare_from(actions, seed_poses, seed_hands)

        previous = tuple(p.copy() for p in seed_poses)
        hands = tuple(h.copy() for h in seed_hands)
        planned = []
        for raw in actions:
            value = np.asarray(raw, dtype=np.float32).copy()
            action = Action58.parse(value)
            target = eef9_column_to_pose7(action.right_eef)
            filtered = previous[1].copy()
            filtered[:3] += self.eef_filter_alpha * (target[:3] - previous[1][:3])
            filtered[3:] = slerp(previous[1][3:], target[3:], self.eef_filter_alpha)
            value[29:38] = pose7_to_eef9_column(filtered)

            # Reuse all original workspace, step, fixed-left and hand guards.
            # The next filter seed is the final constrained target, not the raw
            # prediction or a live position sampled before the committed tail.
            prepared = super().prepare_from([value], previous, hands)[0]
            planned.append(prepared)
            parsed = Action58.parse(prepared.raw)
            previous = (prepared.left, prepared.right)
            hands = (parsed.left_hand, parsed.right_hand)
        return planned
