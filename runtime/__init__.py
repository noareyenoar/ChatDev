from typing import Any

__all__ = ["WorkflowMetaInfo", "WorkflowRunResult", "run_workflow"]


def __getattr__(name: str) -> Any:
	if name in __all__:
		from runtime.sdk import WorkflowMetaInfo, WorkflowRunResult, run_workflow

		exports = {
			"WorkflowMetaInfo": WorkflowMetaInfo,
			"WorkflowRunResult": WorkflowRunResult,
			"run_workflow": run_workflow,
		}
		return exports[name]
	raise AttributeError(f"module 'runtime' has no attribute '{name}'")
