"""Tests for Agent Tool Execution Lifecycle & Hook System."""

from __future__ import annotations

import unittest
from unittest.mock import MagicMock, patch

from src.utils import execution_lifecycle as el
from src.utils.execution_lifecycle import (
    BlastRadius,
    DryRunInterceptionHook,
    HookDecision,
    LifecycleHook,
    LifecyclePipeline,
    ReadbackVerificationHook,
    ResolveStateInspectionHook,
    RiskAssessment,
    RiskClassificationHook,
    RiskLevel,
    ToolCallContext,
    inspect_operation,
    list_lifecycle_hooks,
)
from src import server


class TestRiskClassification(unittest.TestCase):
    """Test risk classification and blast radius calculation."""

    def test_critical_project_action(self):
        assessment = RiskClassificationHook.classify("project_manager", "delete_project", {"project_name": "Test"})
        self.assertEqual(assessment.level, RiskLevel.CRITICAL)
        self.assertTrue(assessment.destructive)
        self.assertEqual(assessment.blast_radius, BlastRadius.PROJECT)
        self.assertTrue(assessment.confirmation_required)

    def test_critical_media_pool_delete_timelines(self):
        assessment = RiskClassificationHook.classify("media_pool", "delete_timelines", {})
        self.assertEqual(assessment.level, RiskLevel.CRITICAL)
        self.assertTrue(assessment.destructive)
        self.assertEqual(assessment.blast_radius, BlastRadius.TIMELINE)
        self.assertTrue(assessment.confirmation_required)

    def test_high_risk_ripple_delete(self):
        assessment = RiskClassificationHook.classify(
            "timeline", "delete_clips", {"clip_ids": ["c1", "c2"], "ripple": True}
        )
        self.assertEqual(assessment.level, RiskLevel.HIGH)
        self.assertTrue(assessment.destructive)
        self.assertEqual(assessment.blast_radius, BlastRadius.TIMELINE)
        self.assertTrue(assessment.confirmation_required)

    def test_high_risk_edit_engine(self):
        assessment = RiskClassificationHook.classify("edit_engine", "execute_selects", {})
        self.assertEqual(assessment.level, RiskLevel.HIGH)
        self.assertTrue(assessment.destructive)

    def test_medium_risk_mutation(self):
        assessment = RiskClassificationHook.classify("timeline_item", "set_color", {"color": "Blue"})
        self.assertEqual(assessment.level, RiskLevel.MEDIUM)
        self.assertEqual(assessment.blast_radius, BlastRadius.ITEM)
        self.assertFalse(assessment.confirmation_required)

    def test_low_risk_read_only(self):
        for act in ["get_timeline_info", "list_clips", "probe_track_items", "query_markers"]:
            assessment = RiskClassificationHook.classify("timeline", act, {})
            self.assertEqual(assessment.level, RiskLevel.LOW, f"Action {act} should be LOW risk")
            self.assertFalse(assessment.destructive)


class TestStateInspectionAndDrift(unittest.TestCase):
    """Test pre-flight Resolve state capture and post-flight drift detection."""

    def test_state_inspection_success(self):
        mock_state = {
            "project": "Demo Project",
            "timeline": "Cut 1",
            "duration_frames": 1200,
            "track_count_video": 2,
        }
        hook = ResolveStateInspectionHook(state_provider=lambda: mock_state)
        ctx = ToolCallContext(
            tool_name="timeline",
            action="delete_clips",
            risk=RiskAssessment(level=RiskLevel.HIGH),
        )
        hook.before_tool_call(ctx)
        self.assertEqual(ctx.pre_state, mock_state)
        self.assertTrue(ctx.risk.snapshot_available)

    def test_drift_detection_duration_shift_warning(self):
        provider = MagicMock(return_value={"duration_frames": 950})
        drift_hook = el.DriftDetectionHook(state_provider=provider)

        ctx = ToolCallContext(
            tool_name="timeline_item",
            action="set_color",  # non-duration altering
            pre_state={"duration_frames": 1000},
        )
        res = {"success": True}
        contrib = drift_hook.after_tool_call(ctx, res, duration_ms=10)
        self.assertIsNotNone(contrib)
        self.assertTrue(contrib.get("drift_detected"))
        self.assertTrue(any("drifted unexpectedly" in w for w in contrib.get("drift_warnings", [])))


class TestDryRunInterception(unittest.TestCase):
    """Test safe dry-run interception for non-native actions."""

    def test_intercept_destructive_dry_run(self):
        hook = DryRunInterceptionHook()
        ctx = ToolCallContext(
            tool_name="timeline",
            action="delete_clips",
            params={"clip_ids": ["c1", "c2"], "dry_run": True},
        )
        decision = hook.before_tool_call(ctx)
        self.assertIsNotNone(decision)
        self.assertFalse(decision.proceed)
        self.assertIsNotNone(decision.short_circuit_result)
        self.assertTrue(decision.short_circuit_result.get("dry_run"))
        self.assertTrue(decision.short_circuit_result.get("simulated"))
        self.assertEqual(decision.short_circuit_result.get("operation"), "timeline.delete_clips")

    def test_pass_through_native_dry_run(self):
        hook = DryRunInterceptionHook()
        ctx = ToolCallContext(
            tool_name="timeline",
            action="ripple_insert",
            params={"clip_id": "c1", "dry_run": True},
        )
        decision = hook.before_tool_call(ctx)
        self.assertIsNone(decision)  # allowed to execute native implementation


class TestReadbackVerificationHook(unittest.TestCase):
    """Test readback verification evaluation and contradiction alerting."""

    def test_contradiction_detection(self):
        hook = ReadbackVerificationHook()
        ctx = ToolCallContext(
            tool_name="timeline_item",
            action="set_color",
            risk=RiskAssessment(level=RiskLevel.MEDIUM, destructive=True),
        )
        result = {
            "success": True,
            "verification": {
                "verified": False,
                "contradiction": True,
                "expected": "Blue",
                "actual": "Red",
            },
        }
        contrib = hook.after_tool_call(ctx, result, duration_ms=15)
        self.assertIsNotNone(contrib)
        self.assertTrue(contrib.get("readback_checked"))
        self.assertTrue(contrib.get("contradiction"))

    def test_unverified_flag(self):
        hook = ReadbackVerificationHook()
        ctx = ToolCallContext(
            tool_name="timeline",
            action="delete_clips",
            risk=RiskAssessment(level=RiskLevel.HIGH, destructive=True),
        )
        result = {"success": True}  # No verification block attached
        contrib = hook.after_tool_call(ctx, result, duration_ms=15)
        self.assertIsNotNone(contrib)
        self.assertEqual(contrib.get("status"), "unverified")


class TestPipelineCoordinator(unittest.TestCase):
    """Test LifecyclePipeline hook coordination and execution flow."""

    def setUp(self):
        self.pipeline = LifecyclePipeline()

    def test_custom_hook_registration(self):
        called = []

        class CustomHook(LifecycleHook):
            name = "custom_test"

            def before_tool_call(self, ctx):
                called.append("before")
                return None

            def after_tool_call(self, ctx, result, duration_ms):
                called.append("after")
                return {"custom": True}

        self.pipeline.register_hook(CustomHook())
        ctx = ToolCallContext(tool_name="timeline", action="list_clips")
        dec = self.pipeline.run_before(ctx)
        self.assertTrue(dec.proceed)
        self.assertIn("before", called)

        res = self.pipeline.run_after(ctx, {"success": True}, 5)
        self.assertIn("after", called)

    def test_inspect_operation_query(self):
        info = self.pipeline.inspect_operation("timeline", "delete_clips", {"ripple": True})
        self.assertEqual(info["tool"], "timeline")
        self.assertEqual(info["action"], "delete_clips")
        self.assertEqual(info["risk"]["level"], "high")
        self.assertEqual(info["blast_radius"], "timeline")
        self.assertTrue(info["destructive"])
        self.assertTrue(info["confirmation_required"])


class TestServerResolveControlIntegration(unittest.TestCase):
    """Test resolve_control actions for lifecycle inspection and listing."""

    def test_resolve_control_inspect_operation(self):
        res = server.resolve_control(
            action="inspect_operation",
            params={
                "tool": "timeline",
                "target_action": "delete_clips",
                "target_params": {"ripple": True},
            },
        )
        self.assertEqual(res.get("tool"), "timeline")
        self.assertEqual(res.get("action"), "delete_clips")
        self.assertEqual(res.get("risk", {}).get("level"), "high")
        self.assertEqual(res.get("blast_radius"), "timeline")

    def test_resolve_control_list_lifecycle_hooks(self):
        res = server.resolve_control(action="list_lifecycle_hooks")
        self.assertTrue(res.get("success"))
        hooks = res.get("hooks", [])
        hook_names = [h["name"] for h in hooks]
        self.assertIn("risk_classification", hook_names)
        self.assertIn("resolve_state_inspection", hook_names)
        self.assertIn("dry_run_interception", hook_names)
        self.assertIn("readback_verification", hook_names)
        self.assertIn("drift_detection", hook_names)
        self.assertIn("provenance_trace", hook_names)


if __name__ == "__main__":
    unittest.main()
