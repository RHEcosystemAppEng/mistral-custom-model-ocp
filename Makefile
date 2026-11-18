PYTHON ?= python3
OC ?= oc
NAMESPACE ?= mistral-custom-model-ocp
PULL_SECRET ?= ibm-entitlement
PVC_NAME ?= mistral-weights
PVC_SIZE ?= 140Gi
STORAGE_CLASS ?=
GPU_COUNT ?= 8
LAYOUT ?= .local/2411-layout.json
PREPARE_TIMEOUT ?= 14400s
SERVING_TIMEOUT ?= 3600s
REPORT_DIR ?= reports
ROUTE_URL ?=
CLEANUP_TIMEOUT ?= 300s

OPERATOR_ARGS = --namespace "$(NAMESPACE)" --pull-secret "$(PULL_SECRET)" --pvc-name "$(PVC_NAME)" --pvc-size "$(PVC_SIZE)" --storage-class "$(STORAGE_CLASS)" --gpu-count "$(GPU_COUNT)"

.PHONY: preflight inventory render-config render-deployment render-discovery render-preparation prepare install status validate restart-verify runtime-check uninstall purge

preflight:
	$(PYTHON) scripts/operator.py preflight --oc "$(OC)" $(OPERATOR_ARGS)

inventory:
	$(PYTHON) scripts/operator.py inventory --oc "$(OC)" $(OPERATOR_ARGS) --layout "$(LAYOUT)" --prepare-timeout "$(PREPARE_TIMEOUT)"

render-config:
	$(PYTHON) scripts/operator.py render-config $(OPERATOR_ARGS)

render-deployment:
	$(PYTHON) scripts/operator.py render-deployment $(OPERATOR_ARGS) --serving-timeout "$(SERVING_TIMEOUT)"

render-discovery:
	$(PYTHON) scripts/operator.py render-discovery $(OPERATOR_ARGS)

render-preparation:
	$(PYTHON) scripts/operator.py render-preparation $(OPERATOR_ARGS) --layout "$(LAYOUT)"

prepare:
	$(PYTHON) scripts/operator.py prepare --oc "$(OC)" $(OPERATOR_ARGS) --layout "$(LAYOUT)" --prepare-timeout "$(PREPARE_TIMEOUT)" --report-dir "$(REPORT_DIR)"

install:
	$(PYTHON) scripts/operator.py install --oc "$(OC)" $(OPERATOR_ARGS) --serving-timeout "$(SERVING_TIMEOUT)" --report-dir "$(REPORT_DIR)"

status:
	$(PYTHON) scripts/operator.py status --oc "$(OC)" $(OPERATOR_ARGS)

uninstall purge:
	$(PYTHON) scripts/operator.py $@ --oc "$(OC)" $(OPERATOR_ARGS) --cleanup-timeout "$(CLEANUP_TIMEOUT)"

runtime-check:
	$(PYTHON) scripts/operator.py runtime-check --oc "$(OC)" $(OPERATOR_ARGS) --serving-timeout "$(SERVING_TIMEOUT)" --report-dir "$(REPORT_DIR)"

restart-verify:
	$(PYTHON) scripts/operator.py restart-verify --oc "$(OC)" $(OPERATOR_ARGS) --serving-timeout "$(SERVING_TIMEOUT)" --report-dir "$(REPORT_DIR)"

validate:
	$(PYTHON) scripts/operator.py validate --oc "$(OC)" $(OPERATOR_ARGS) --route-url "$(ROUTE_URL)" --report-dir "$(REPORT_DIR)"

.PHONY: concurrency-check
concurrency-check:
	$(PYTHON) scripts/operator.py concurrency-check --oc "$(OC)" $(OPERATOR_ARGS) --route-url "$(ROUTE_URL)" --report-dir "$(REPORT_DIR)"

.PHONY: context-check
context-check:
	$(PYTHON) scripts/operator.py context-check --oc "$(OC)" $(OPERATOR_ARGS) --route-url "$(ROUTE_URL)" --report-dir "$(REPORT_DIR)"

.PHONY: verify-run
verify-run:
	$(PYTHON) scripts/operator.py verify-run --oc "$(OC)" $(OPERATOR_ARGS) --layout "$(LAYOUT)" --prepare-timeout "$(PREPARE_TIMEOUT)" --serving-timeout "$(SERVING_TIMEOUT)" --cleanup-timeout "$(CLEANUP_TIMEOUT)" --route-url "$(ROUTE_URL)" --report-dir "$(REPORT_DIR)"

RUN_REPORT ?=
.PHONY: purge-run
purge-run:
	$(PYTHON) scripts/operator.py purge-run --oc "$(OC)" $(OPERATOR_ARGS) --run-report "$(RUN_REPORT)" --cleanup-timeout "$(CLEANUP_TIMEOUT)"

.PHONY: test
test:
	$(PYTHON) -m unittest discover -s tests -v
