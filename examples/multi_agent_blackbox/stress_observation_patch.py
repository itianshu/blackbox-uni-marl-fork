"""Enable transaction/sync diagnostics without changing scheduling decisions."""
import logging


def apply_patch():
    from examples.multi_agent_blackbox import verl_patch
    verl_patch.apply_patch()
    for suffix in ("controller", "executor", "patch"):
        logger = logging.getLogger("uni_agent.trainer.dynamic_inference." + suffix)
        logger.disabled = False
        logger.setLevel(logging.INFO)
        if not any(getattr(h, "_stress_observer", False) for h in logger.handlers):
            handler = logging.StreamHandler()
            handler._stress_observer = True
            handler.setFormatter(logging.Formatter("%(asctime)s %(name)s %(levelname)s %(message)s"))
            logger.addHandler(handler)
        logger.propagate = False
