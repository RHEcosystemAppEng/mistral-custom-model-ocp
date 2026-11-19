"""Kubelet probe evidence must belong to the current runtime process."""

import unittest
from unittest.mock import Mock


from mistral_validation import serving


class HealthObservationTests(unittest.TestCase):
    def setUp(self):
        self.pod = {"metadata": {"uid": "current-pod"}, "status": {
            "containerStatuses": [{"name": "kserve-container", "state": {
                "running": {"startedAt": "2026-11-17T19:00:00Z"}}}]}}
        self.event = {"reason": "Unhealthy", "involvedObject": {
            "uid": "current-pod", "fieldPath": "spec.containers{kserve-container}"},
            "lastTimestamp": "2026-11-17T19:00:30Z",
            "message": 'Startup probe failed: Get "http://10.0.0.1:3000/health": '
                       'dial tcp 10.0.0.1:3000: connect: connection refused'}
        self.oc = Mock()
        self.oc.get.return_value = {"items": [self.event]}

    def observed(self):
        return serving.failed_health_probe(self.oc, "validation", self.pod, RuntimeError)

    def test_unbound_runtime_port_is_evidence_of_health_unready(self):
        self.assertTrue(self.observed())

    def test_other_pod_or_sidecar_or_endpoint_is_not_loading_evidence(self):
        for field, value in (("uid", "old-pod"),
                             ("fieldPath", "spec.containers{kube-rbac-proxy}")):
            with self.subTest(field=field):
                original = self.event["involvedObject"][field]
                self.event["involvedObject"][field] = value
                self.assertFalse(self.observed())
                self.event["involvedObject"][field] = original
        self.event["message"] = 'Startup probe failed: Get "http://10.0.0.1:8080/health"'
        self.assertFalse(self.observed())

    def test_probe_failure_before_container_restart_does_not_count(self):
        self.event["lastTimestamp"] = "2026-11-17T18:59:00Z"
        self.assertFalse(self.observed())

    def test_missing_or_inaccessible_events_do_not_establish_transition(self):
        self.oc.get.return_value = {"items": []}
        self.assertFalse(self.observed())
        self.oc.get.side_effect = RuntimeError()
        self.assertFalse(self.observed())


if __name__ == "__main__":
    unittest.main()
