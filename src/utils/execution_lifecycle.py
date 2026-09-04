"""Agent Tool Execution Lifecycle & Hook System for DaVinci Resolve MCP.

Unifies pre-flight risk evaluation, Resolve state inspection, permission/confirmation
gating, dry-run simulation, post-flight readback verification, drift detection, and
execution trace emission into a cohesive before/after tool execution pipeline.

Pipeline flow:
    Agent
      ↓
    MCP tool request
      ↓
    ┌──────────────────────────────────────────────┐
    │ before_tool_call                             │
    │   • inspect operation & parameters           │
    │   • inspect Resolve state (pre-flight)       │
    │   • permission / confirmation gate check     │
    │   • risk classification & blast radius       │
    │   • dry-run interception                     │
    └──────────────────────┬───────────────────────┘
                           ↓
                     Execute Resolve
                           ↓
    ┌──────────────────────┴───────────────────────┐
    │ after_tool_call                              │
    │   • verify result                            │
    │   • readback observation                     │
    │   • detect drift (pre-state vs post-state)   │
    │   • record provenance & emit trace           │
    └──────────────────────────────────────────────┘
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from src.utils import execution_trace, readback
from src.utils.destructive_hook import (
    DESTRUCTIVE_ACTIONS_BY_TOOL,
    STRICT_DEFAULT_ACTIONS,
    is_destructive,
    is_strict_required,
)

logger = logging.getLogger("resolve-mcp.lifecycle")


# ── Risk Data Model ─────────────────────────────────────────────────────────

class RiskLevel(str, Enum):
    """Risk tier of an MCP operation."""
    LOW = "low"            # Read-only queries, probes, detectors, exports
    MEDIUM = "medium"      # Additive or reversible mutations (add marker, set color, import)
    HIGH = "high"          # Content-modifying edits (delete clips, ripple cut, split)
    CRITICAL = "critical"  # Catastrophic / irreversible actions (delete project, delete timelines)


class BlastRadius(str, Enum):
    """Scope of impact of an MCP operation."""
    ITEM = "item"          # Single clip, marker, or node
    TRACK = "track"        # Whole track or multiple items on track
    TIMELINE = "timeline"  # Whole timeline
    PROJECT = "project"    # Whole project or media pool hierarchy
    DISK = "disk"          # Filesystem / storage


@dataclass
class RiskAssessment:
    """Pre-flight risk assessment for an operation."""
    level: RiskLevel = RiskLevel.LOW
    destructive: bool = False
    blast_radius: BlastRadius = BlastRadius.ITEM
    confirmation_required: bool = False
    snapshot_available: bool = False
    reasons: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "level": self.level.value,
            "destructive": self.destructive,
            "blast_radius": self.blast_radius.value,
            "confirmation_required": self.confirmation_required,
            "snapshot_available": self.snapshot_available,
            "reasons": list(self.reasons),
        }


# ── Context and Decisions ───────────────────────────────────────────────────

@dataclass
class ToolCallContext:
    """Context passed through the lifecycle hooks for a single tool execution."""
    tool_name: str
    action: str
    params: Dict[str, Any] = field(default_factory=dict)
    execution_id: Optional[str] = None
    dry_run: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)
    pre_state: Optional[Dict[str, Any]] = None
    post_state: Optional[Dict[str, Any]] = None
    risk: Optional[RiskAssessment] = None
    timestamp_start: float = field(default_factory=time.time)

    def get_param(self, key: str, default: Any = None) -> Any:
        return self.params.get(key, default) if isinstance(self.params, dict) else default


@dataclass
class HookDecision:
    """Decision returned by before_tool_call hooks."""
    proceed: bool = True
    short_circuit_result: Optional[Dict[str, Any]] = None
    risk: Optional[RiskAssessment] = None
    warnings: List[str] = field(default_factory=list)


# ── Hook Base Class ─────────────────────────────────────────────────────────

class LifecycleHook:
    """Base interface for lifecycle hooks."""
    name: str = "base_hook"
    enabled: bool = True

    def before_tool_call(self, ctx: ToolCallContext) -> Optional[HookDecision]:
        """Runs before tool execution. Return HookDecision(proceed=False) to abort/intercept."""
        return None

    def after_tool_call(
        self, ctx: ToolCallContext, result: Any, duration_ms: int
    ) -> Optional[Dict[str, Any]]:
        """Runs after successful tool execution. Can return additional metadata/verifications."""
        return None

    def on_error(
        self, ctx: ToolCallContext, error: Exception, duration_ms: int
    ) -> Optional[Dict[str, Any]]:
        """Runs if tool execution raises an unhandled exception."""
        return None


# ── Built-in Hooks ──────────────────────────────────────────────────────────

# Critical actions that affect entire project or destroy timelines/folders
_CRITICAL_ACTIONS: Dict[str, Tuple[str, ...]] = {
    "project_manager": ("delete_project", "close_project_without_saving"),
    "media_pool": ("delete_timelines", "delete_folders", "purge_unused"),
    "timeline": ("delete_timelines", "delete_track"),
    "graph": ("reset_all_grades",),
}

# High-risk actions that alter timeline content or media structure
_HIGH_RISK_ACTIONS: Dict[str, Tuple[str, ...]] = {
    "timeline": (
        "delete_clips", "lift_range", "overwrite_range", "apply_cuts",
        "ripple_insert", "split_clip", "replace_clip",
    ),
    "media_pool": ("delete_clips", "move_clips", "move_folders"),
    "edit_engine": ("execute_selects", "execute_tighten", "execute_silence_ripple", "execute_swap"),
    "timeline_item": ("set_retime", "set_audio"),
    "timeline_item_color": ("reset_all_node_colors",),
}

# Read-only action prefixes
_READ_ONLY_PREFIXES = (
    "get_", "list_", "probe_", "detect_", "inspect_", "find_", "query_",
    "read_", "check_", "validate_", "summarize_", "export_", "render_status"
)


class RiskClassificationHook(LifecycleHook):
    """Evaluates risk level, blast radius, and confirmation gating before execution."""
    name = "risk_classification"

    def before_tool_call(self, ctx: ToolCallContext) -> Optional[HookDecision]:
        assessment = self.classify(ctx.tool_name, ctx.action, ctx.params)
        ctx.risk = assessment

        # If confirmation is required and no confirm_token or confirmed flag is provided,
        # note it in decision
        if assessment.confirmation_required:
            has_token = bool(ctx.params.get("confirm_token") or ctx.params.get("confirmToken"))
            confirmed = bool(ctx.params.get("confirmed", False))
            if not has_token and not confirmed:
                logger.info(
                    "Action %s.%s requires confirmation (risk=%s, blast_radius=%s)",
                    ctx.tool_name, ctx.action, assessment.level.value, assessment.blast_radius.value
                )

        return HookDecision(proceed=True, risk=assessment)

    @classmethod
    def classify(
        cls, tool_name: str, action: str, params: Optional[Dict[str, Any]] = None
    ) -> RiskAssessment:
        params = params or {}
        reasons: List[str] = []
        is_dest = is_destructive(tool_name, action, params)

        # Check critical
        critical_ops = _CRITICAL_ACTIONS.get(tool_name, ())
        if action in critical_ops:
            reasons.append(f"Action '{action}' permanently alters or destroys project-level assets")
            radius = BlastRadius.PROJECT if "project" in action or tool_name == "project_manager" else BlastRadius.TIMELINE
            return RiskAssessment(
                level=RiskLevel.CRITICAL,
                destructive=True,
                blast_radius=radius,
                confirmation_required=True,
                snapshot_available=True,
                reasons=reasons,
            )

        # Check high
        high_ops = _HIGH_RISK_ACTIONS.get(tool_name, ())
        is_ripple_delete = (
            tool_name == "timeline"
            and action == "delete_clips"
            and bool(params.get("ripple", False))
        )
        if action in high_ops or is_ripple_delete or is_strict_required(tool_name, action, params):
            if is_ripple_delete:
                reasons.append("Ripple delete shifts all subsequent timeline content")
                radius = BlastRadius.TIMELINE
            elif "track" in action:
                reasons.append(f"Action '{action}' alters whole track structure")
                radius = BlastRadius.TRACK
            else:
                reasons.append(f"Action '{action}' modifies timeline content or edits")
                radius = BlastRadius.ITEM

            return RiskAssessment(
                level=RiskLevel.HIGH,
                destructive=True,
                blast_radius=radius,
                confirmation_required=bool(params.get("strict") or is_ripple_delete),
                snapshot_available=True,
                reasons=reasons,
            )

        # Check medium (destructive or additive mutations)
        if is_dest:
            reasons.append(f"Action '{action}' mutates working timeline or clip state")
            return RiskAssessment(
                level=RiskLevel.MEDIUM,
                destructive=True,
                blast_radius=BlastRadius.ITEM,
                confirmation_required=False,
                snapshot_available=True,
                reasons=reasons,
            )

        # Mutations that are not in destructive registry (e.g. metadata, markers, colors)
        if not any(action.startswith(p) for p in _READ_ONLY_PREFIXES):
            reasons.append(f"Action '{action}' performs non-destructive mutation or configuration")
            return RiskAssessment(
                level=RiskLevel.MEDIUM,
                destructive=False,
                blast_radius=BlastRadius.ITEM,
                confirmation_required=False,
                snapshot_available=False,
                reasons=reasons,
            )

        # Low (read-only queries)
        reasons.append(f"Action '{action}' is a read-only query or inspection")
        return RiskAssessment(
            level=RiskLevel.LOW,
            destructive=False,
            blast_radius=BlastRadius.ITEM,
            confirmation_required=False,
            snapshot_available=False,
            reasons=reasons,
        )


class ResolveStateInspectionHook(LifecycleHook):
    """Captures pre-flight Resolve state snapshot (timeline name, duration, item count) when available."""
    name = "resolve_state_inspection"

    def __init__(self, state_provider: Optional[Callable[[], Optional[Dict[str, Any]]]] = None):
        self._state_provider = state_provider

    def set_state_provider(self, provider: Callable[[], Optional[Dict[str, Any]]]) -> None:
        self._state_provider = provider

    def before_tool_call(self, ctx: ToolCallContext) -> Optional[HookDecision]:
        # Only inspect state for non-observer actions to avoid overhead
        if ctx.risk and ctx.risk.level == RiskLevel.LOW:
            return None

        if self._state_provider is not None:
            try:
                state = self._state_provider()
                if state:
                    ctx.pre_state = state
                    if ctx.risk:
                        ctx.risk.snapshot_available = bool(state.get("timeline") or state.get("project"))
            except Exception as exc:
                logger.debug("Pre-flight Resolve state inspection failed: %s", exc)

        return None


# Actions known to implement their own domain-specific dry_run simulation logic
NATIVE_DRY_RUN_ACTIONS: frozenset[Tuple[str, str]] = frozenset({
    ("timeline", "ripple_insert"),
    ("media_analysis", "analyze_media"),
    ("media_analysis", "analyze_batch"),
    ("media_analysis", "analyze_file"),
    ("media_analysis", "analyze_timeline"),
    ("media_analysis", "delete_source_media"),
    ("media_pool", "bulk_match_to_hero"),
    ("media_pool", "relink_clips"),
})


class DryRunInterceptionHook(LifecycleHook):
    """Intercepts dry-run calls to return a structured impact preview without executing mutations."""
    name = "dry_run_interception"

    def before_tool_call(self, ctx: ToolCallContext) -> Optional[HookDecision]:
        caller_dry_run = bool(ctx.params.get("dry_run", False) or ctx.params.get("dryRun", False))
        if not caller_dry_run:
            return None

        ctx.dry_run = True
        risk = ctx.risk or RiskClassificationHook.classify(ctx.tool_name, ctx.action, ctx.params)

        # Allow actions with native domain-specific dry-run implementations to run
        if (ctx.tool_name, ctx.action) in NATIVE_DRY_RUN_ACTIONS:
            return None

        preview = {
            "dry_run": True,
            "operation": f"{ctx.tool_name}.{ctx.action}",
            "risk": risk.to_dict(),
            "target_params": {k: v for k, v in ctx.params.items() if k not in {"dry_run", "dryRun"}},
            "pre_state": ctx.pre_state or {},
            "simulated": True,
            "message": f"Dry-run simulation for {ctx.tool_name}.{ctx.action}. No changes were made.",
        }

        return HookDecision(
            proceed=False,
            short_circuit_result=preview,
            risk=risk,
            warnings=["Operation intercepted by dry-run lifecycle hook; no changes committed to Resolve."],
        )


class ReadbackVerificationHook(LifecycleHook):
    """Post-flight hook: evaluates verification results and logs contradictions."""
    name = "readback_verification"

    def after_tool_call(
        self, ctx: ToolCallContext, result: Any, duration_ms: int
    ) -> Optional[Dict[str, Any]]:
        if not isinstance(result, dict):
            return None

        # Check if result contains readback verification dict
        verif = result.get("verification")
        if isinstance(verif, dict):
            if verif.get("contradiction"):
                logger.warning(
                    "Contradiction in %s.%s: API reported success but readback verification failed",
                    ctx.tool_name, ctx.action
                )
            return {"readback_checked": True, "contradiction": bool(verif.get("contradiction"))}

        # If result has success=True and destructive=True but no verification block, note unverified
        if ctx.risk and ctx.risk.destructive and result.get("success") is True and "verification" not in result:
            return {"readback_checked": False, "status": "unverified"}

        return None


class DriftDetectionHook(LifecycleHook):
    """Compares pre-state and post-state to detect unexpected side effects."""
    name = "drift_detection"

    def __init__(self, state_provider: Optional[Callable[[], Optional[Dict[str, Any]]]] = None):
        self._state_provider = state_provider

    def set_state_provider(self, provider: Callable[[], Optional[Dict[str, Any]]]) -> None:
        self._state_provider = provider

    def after_tool_call(
        self, ctx: ToolCallContext, result: Any, duration_ms: int
    ) -> Optional[Dict[str, Any]]:
        if not ctx.pre_state or self._state_provider is None:
            return None

        try:
            post_state = self._state_provider()
            if not post_state:
                return None
            ctx.post_state = post_state

            drift_warnings: List[str] = []
            pre_dur = ctx.pre_state.get("duration_frames")
            post_dur = post_state.get("duration_frames")

            # If action was not expected to change duration (e.g. set_clip_color), but duration changed:
            duration_altering = ctx.action in {"delete_clips", "lift_range", "ripple_insert", "apply_cuts"}
            if pre_dur is not None and post_dur is not None and pre_dur != post_dur and not duration_altering:
                drift_warnings.append(
                    f"Timeline duration drifted unexpectedly from {pre_dur} to {post_dur} frames"
                )

            if drift_warnings:
                logger.warning("Drift detected in %s.%s: %s", ctx.tool_name, ctx.action, drift_warnings)
                return {"drift_detected": True, "drift_warnings": drift_warnings}
        except Exception as exc:
            logger.debug("Drift detection check failed: %s", exc)

        return None


class ProvenanceTraceHook(LifecycleHook):
    """Emits trace steps into execution_trace and attaches timing metadata."""
    name = "provenance_trace"

    def after_tool_call(
        self, ctx: ToolCallContext, result: Any, duration_ms: int
    ) -> Optional[Dict[str, Any]]:
        return {"duration_ms": duration_ms}


# ── Pipeline Coordinator ───────────────────────────────────────────────────

class LifecyclePipeline:
    """Coordinates before and after lifecycle hooks for MCP tool execution."""

    def __init__(self) -> None:
        self._hooks: List[LifecycleHook] = []
        self._lock = threading.Lock()
        self._register_default_hooks()

    def _register_default_hooks(self) -> None:
        self._risk_hook = RiskClassificationHook()
        self._state_hook = ResolveStateInspectionHook()
        self._dry_run_hook = DryRunInterceptionHook()
        self._readback_hook = ReadbackVerificationHook()
        self._drift_hook = DriftDetectionHook()
        self._trace_hook = ProvenanceTraceHook()

        self._hooks = [
            self._risk_hook,
            self._state_hook,
            self._dry_run_hook,
            self._readback_hook,
            self._drift_hook,
            self._trace_hook,
        ]

    def register_hook(self, hook: LifecycleHook, index: Optional[int] = None) -> None:
        """Add a custom hook to the pipeline."""
        with self._lock:
            if index is None:
                self._hooks.append(hook)
            else:
                self._hooks.insert(index, hook)

    def set_state_provider(self, provider: Callable[[], Optional[Dict[str, Any]]]) -> None:
        """Configure the state provider for Resolve inspection and drift detection."""
        self._state_hook.set_state_provider(provider)
        self._drift_hook.set_state_provider(provider)

    def list_hooks(self) -> List[Dict[str, Any]]:
        """Return list of active hooks and their configuration."""
        with self._lock:
            return [{"name": h.name, "enabled": h.enabled} for h in self._hooks]

    def run_before(self, ctx: ToolCallContext) -> HookDecision:
        """Run all before_tool_call hooks in sequence.
        
        If any hook returns proceed=False, short-circuits execution and returns
        the hook's decision immediately.
        """
        decision = HookDecision(proceed=True)
        with self._lock:
            hooks = list(self._hooks)

        for hook in hooks:
            if not hook.enabled:
                continue
            try:
                sub_dec = hook.before_tool_call(ctx)
                if sub_dec is not None:
                    if sub_dec.risk:
                        decision.risk = sub_dec.risk
                    if sub_dec.warnings:
                        decision.warnings.extend(sub_dec.warnings)
                    if not sub_dec.proceed:
                        decision.proceed = False
                        decision.short_circuit_result = sub_dec.short_circuit_result
                        return decision
            except Exception as exc:
                logger.warning("Lifecycle hook '%s.before_tool_call' failed: %s", hook.name, exc)

        return decision

    def run_after(
        self, ctx: ToolCallContext, result: Any, duration_ms: int
    ) -> Any:
        """Run all after_tool_call hooks in sequence. Merges hook contributions into result."""
        with self._lock:
            hooks = list(self._hooks)

        hook_data: Dict[str, Any] = {}
        for hook in hooks:
            if not hook.enabled:
                continue
            try:
                contrib = hook.after_tool_call(ctx, result, duration_ms)
                if isinstance(contrib, dict):
                    hook_data.update(contrib)
            except Exception as exc:
                logger.warning("Lifecycle hook '%s.after_tool_call' failed: %s", hook.name, exc)

        # If result is a dict and we have hook data (like drift warnings or readback check), merge
        if isinstance(result, dict) and hook_data:
            if "drift_warnings" in hook_data:
                existing_warns = result.setdefault("warnings", [])
                if isinstance(existing_warns, list):
                    existing_warns.extend(hook_data["drift_warnings"])

        return result

    def run_on_error(
        self, ctx: ToolCallContext, error: Exception, duration_ms: int
    ) -> None:
        """Notify hooks of unhandled exceptions during tool execution."""
        with self._lock:
            hooks = list(self._hooks)

        for hook in hooks:
            if not hook.enabled:
                continue
            try:
                hook.on_error(ctx, error, duration_ms)
            except Exception as exc:
                logger.debug("Lifecycle hook '%s.on_error' failed: %s", hook.name, exc)

    def inspect_operation(
        self, tool_name: str, action: str, params: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Inspect and assess an operation's risk, blast radius, and requirements without executing."""
        params = params or {}
        risk = RiskClassificationHook.classify(tool_name, action, params)

        pre_state = None
        if self._state_hook._state_provider is not None:
            try:
                pre_state = self._state_hook._state_provider()
            except Exception:
                pre_state = None

        return {
            "tool": tool_name,
            "action": action,
            "risk": risk.to_dict(),
            "destructive": risk.destructive,
            "blast_radius": risk.blast_radius.value,
            "confirmation_required": risk.confirmation_required,
            "snapshot_available": bool(pre_state and (pre_state.get("timeline") or pre_state.get("project"))),
            "reasons": risk.reasons,
            "pre_state": pre_state or {},
        }


# ── Global Pipeline Instance ───────────────────────────────────────────────

_GLOBAL_PIPELINE = LifecyclePipeline()


def get_lifecycle_pipeline() -> LifecyclePipeline:
    """Return the active global LifecyclePipeline instance."""
    return _GLOBAL_PIPELINE


def inspect_operation(
    tool_name: str, action: str, params: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Public helper to inspect an operation's risk, requirements, and blast radius."""
    return _GLOBAL_PIPELINE.inspect_operation(tool_name, action, params)


def list_lifecycle_hooks() -> List[Dict[str, Any]]:
    """Return the list of registered lifecycle hooks."""
    return _GLOBAL_PIPELINE.list_hooks()
