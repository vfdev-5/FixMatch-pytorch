import json
from collections import OrderedDict

from ctaugment.ctaugment import *


class StorableCTAugment(CTAugment):
    def load_state_dict(self, state):
        for k in ["decay", "depth", "th", "rates"]:
            assert k in state, "{} not in {}".format(k, state.keys())
            setattr(self, k, state[k])

    def state_dict(self):
        return OrderedDict(
            [(k, getattr(self, k)) for k in ["decay", "depth", "th", "rates"]]
        )


def get_default_cta():
    return StorableCTAugment()


def cta_apply(pil_img, ops):
    if ops is None:
        return pil_img
    for op, args in ops:
        pil_img = OPS[op].f(pil_img, *args)
    return pil_img


def deserialize(policy_str):
    return [OP(f=x[0], bins=x[1]) for x in json.loads(policy_str)]


def stats(cta):
    return "\n".join(
        "%-16s    %s"
        % (
            k,
            " / ".join(
                " ".join("%.2f" % x for x in cta.rate_to_p(rate))
                for rate in cta.rates[k]
            ),
        )
        for k in sorted(OPS.keys())
    )
