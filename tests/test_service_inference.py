"""The protected KServe Service requires authenticated, verified TLS."""

import io
import json
import ssl
import unittest
from unittest.mock import Mock, patch


from mistral_validation import serving


class ProtectedServiceTests(unittest.TestCase):
    def setUp(self):
        self.oc = Mock(binary="oc")
        self.oc.get.side_effect = [{"spec": {"ports": [{"port": 8443}]}},
                                   {"data": {"service-ca.crt": "cluster-service-ca"}}]
        self.oc.run.return_value = "operator-token\n"

    def test_protected_service_uses_cluster_ca_dns_name_and_operator_token(self):
        response = io.StringIO(json.dumps({"model": serving.SERVED_MODEL,
                                          "choices": [{"message": {"content": "ready"}}]}))
        response.status = 200
        connection = Mock()
        connection.getresponse.return_value = response
        forward = Mock()
        forward.poll.return_value = None
        with patch.object(serving.ssl, "create_default_context") as context, \
                patch.object(serving, "ForwardedServiceTLS", return_value=connection) as tls, \
                patch.object(serving, "start_forward", return_value=(forward, 12345)) as start:
            self.assertTrue(serving.service_inference(self.oc, "validation", RuntimeError))
        context.assert_called_once_with(cadata="cluster-service-ca")
        tls.assert_called_once_with(
            "mistral-large-instruct-2411-predictor.validation.svc", 12345, context.return_value)
        self.assertEqual(start.call_args.args[-1], 8443)
        self.assertEqual(connection.request.call_args.args[3]["Authorization"], "Bearer operator-token")
        connection.close.assert_called_once()
        forward.terminate.assert_called_once()

    def test_invalid_ca_fails_before_retrieving_or_sending_credentials(self):
        with patch.object(serving.ssl, "create_default_context", side_effect=ssl.SSLError()), \
                patch.object(serving, "start_forward") as start:
            self.assertFalse(serving.service_inference(self.oc, "validation", RuntimeError))
        self.oc.run.assert_not_called()
        start.assert_not_called()

    def test_empty_operator_token_fails_before_opening_forward(self):
        self.oc.run.return_value = ""
        with patch.object(serving.ssl, "create_default_context"), \
                patch.object(serving, "start_forward") as start:
            self.assertFalse(serving.service_inference(self.oc, "validation", RuntimeError))
        start.assert_not_called()

    def test_tls_hostname_validation_failure_closes_socket(self):
        context = ssl.create_default_context()
        self.assertTrue(context.check_hostname)
        self.assertEqual(context.verify_mode, ssl.CERT_REQUIRED)
        connection = serving.ForwardedServiceTLS("predictor.validation.svc", 12345, context)
        sock = Mock()
        with patch.object(serving.socket, "create_connection", return_value=sock), \
                patch.object(context, "wrap_socket", side_effect=ssl.SSLCertVerificationError()) as wrap:
            with self.assertRaises(ssl.SSLCertVerificationError):
                connection.connect()
        wrap.assert_called_once_with(sock, server_hostname="predictor.validation.svc")
        sock.close.assert_called_once()


if __name__ == "__main__":
    unittest.main()
