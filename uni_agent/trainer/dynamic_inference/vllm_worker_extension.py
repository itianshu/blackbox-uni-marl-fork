"""Named vLLM RPC extension; preserves verl checkpoint and sleep support."""
from verl.workers.rollout.vllm_rollout.utils import vLLMColocateWorkerExtension


class CloneWorkerExtension(vLLMColocateWorkerExtension):
    def dynamic_inference_clone_worker(self, operation, payload):
        from .replica_clone import clone_worker
        return clone_worker(self, operation, payload)
